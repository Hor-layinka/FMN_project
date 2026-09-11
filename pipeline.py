# preprocessing pipeline for new data
import pandas as pd
import numpy as np

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes schema, dates, and sorts chronologically per SKU.
    """
    df = df.copy()

    # remove duplicates and null values
    df = df.drop_duplicates()
    df = df.dropna()
    # reformat category columns to title case for consistency
    df['category'] = df['category'].str.title()

    # make sku_id and category categorical types for memory efficiency and model compatibility
    df['sku_id'] = df['sku_id'].astype('category')
    df['category'] = df['category'].astype('category')

    # Enforce Datetime and clean column naming
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure numeric columns are strictly float/int
    num_cols = ['units_sold', 'units_received', 'closing_stock', 'lead_time_days']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Sort strictly by SKU and Date to guarantee sequential validity
    df = df.sort_values(by=['date', 'sku_id']).reset_index(drop=True)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling velocity, volatility, hierarchy shares, and calendar signals.
    Applies shift(1) to past consumption to prevent lookahead data leakage.
    """
    df = clean_raw_data(df)
    
    # ---------------------------------------------------------
    # 1. Calendar & Temporal Features
    # ---------------------------------------------------------
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # ---------------------------------------------------------
    # 2. Autoregressive Lags & Rolling Consumption (SKU Level)
    # ---------------------------------------------------------
    # shift(1) guarantees rolling values use strictly historical data up to yesterday
    demand = df.groupby('sku_id')['units_sold'].shift(1)

    df['lag1_units_sold'] = df.groupby('sku_id')['units_sold'].shift(1)
    df['lag7_units_sold'] = df.groupby('sku_id')['units_sold'].shift(7)

    # create moving averages for 7, 14, shifted by 1 to prevent leakage
    df['ma7_units_sold'] = df.groupby('sku_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['ma14_units_sold'] = df.groupby('sku_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )
    # 14 days volatility
    df['vol14_units_sold'] = df.groupby('sku_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).std()
)

    # demand momentum
    eps = 1e-5  # Prevents division by zero
    df['demand_momentum'] = (df['ma7_units_sold'] + eps) / (df['ma14_units_sold'] + eps)

    # ---------------------------------------------------------
    # 3. Volatility & Uncertainty (SKU Level)
    # ---------------------------------------------------------
    df['vol14_units_sold'] = df.groupby('sku_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).std()
    )

    # Coefficient of Variation: Volatility scaled by mean
    df['demand_cv'] = df['vol14_units_sold'] / (df['ma14_units_sold'] + 1e-5)

    # ---------------------------------------------------------
    # 4. Category-Level Aggregates (Cross-SKU Context)
    # ---------------------------------------------------------
    # Aggregate total units sold by category per date
    cat_daily_dem = df.groupby(['date', 'category'])['units_sold'].sum().reset_index()
    cat_daily_dem = cat_daily_dem.sort_values(by=['category', 'date']).reset_index(drop=True)
    
    # 7-day category rolling sales
    # 7-day category moving average (shifted by 1 to prevent leakage)
    cat_daily_dem['cat_dem_ma7'] = (
        cat_daily_dem.groupby('category')['units_sold']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(0.0)
    )

    # Merge back to primary dataframe
    df = df.merge(
        cat_daily_dem[['date', 'category', 'cat_dem_ma7']],
        on=['date', 'category'],
        how='left'
    )

    # SKU's proportional share within its broader category
    df['sku_category_share'] = (df['ma7_units_sold'] + 1e-5) / (df['cat_dem_ma7'] + 1e-5)

    # drop any rows with NaN values that may have resulted from rolling calculations
    df = df.dropna().reset_index(drop=True)

    return df


def build_pipeline_from_raw(raw_input) -> pd.DataFrame:
    """
    Convenience wrapper to ingest raw data and produce a fully engineered frame.
    Accepts either a CSV path/buffer (str) or an already-loaded DataFrame.
    """
    if isinstance(raw_input, pd.DataFrame):
        raw_df = raw_input.copy()
    else:
        raw_df = pd.read_csv(raw_input)
    processed_df = engineer_features(raw_df)
    return processed_df