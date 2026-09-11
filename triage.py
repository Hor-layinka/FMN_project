import numpy as np
import pandas as pd

FEATURE_COLS = [
    'sku_id', 'category', 'lead_time_days',
    'day_of_week', 'day_of_month', 'month', 'is_weekend',
    'lag1_units_sold', 'lag7_units_sold',
    'ma7_units_sold', 'ma14_units_sold', 'vol14_units_sold',
    'demand_momentum', 'demand_cv', 'cat_dem_ma7', 'sku_category_share',
]

MAX_HORIZON_CAP = 14  # safety cap even if a SKU's lead_time_days is unexpectedly large
LOW_CONFIDENCE_SKUS = {'SKU-1011', 'SKU-1000', 'SKU-1007', 'SKU-1014', 'SKU-1020'}

def _apply_trained_dtypes(df, trained_categories):
    """
    Pins sku_id/category to the exact categories seen at training time.
    Prevents a batch missing some SKUs from silently remapping category
    codes and corrupting predictions with no error thrown.
    """
    df = df.copy()
    if trained_categories is not None:
        df['sku_id'] = df['sku_id'].astype('category').cat.set_categories(trained_categories['sku_id'])
        df['category'] = df['category'].astype('category').cat.set_categories(trained_categories['category'])
    else:
        df['sku_id'] = df['sku_id'].astype('category')
        df['category'] = df['category'].astype('category')
    return df


def fit_historical(df: pd.DataFrame, model, feature_cols: list = FEATURE_COLS,
                    trained_categories: dict = None) -> pd.DataFrame:
    """
    Runs the model on rows that already have real outcomes.
    This is IN-SAMPLE FIT, not a forecast — useful only for showing
    'model vs actual' on the historical portion of a chart. Never used
    for triage decisions (that's what generate_forecast does).
    """
    df = _apply_trained_dtypes(df, trained_categories)
    X = df[feature_cols]
    preds = model.predict(X)
    df = df.copy()
    df['fitted_demand'] = np.clip(preds, a_min=0, a_max=None).round(1)
    return df


def generate_forecast(df: pd.DataFrame, model, feature_cols: list = FEATURE_COLS,
                       k: float = 1.35, max_horizon_cap: int = MAX_HORIZON_CAP,
                       trained_categories: dict = None) -> pd.DataFrame:
    df = _apply_trained_dtypes(df, trained_categories)
    forecast_rows = []

    for sku_id, g in df.groupby('sku_id', observed=True):
        g = g.sort_values('date').reset_index(drop=True)
        if g.empty:
            continue

        last_row = g.iloc[-1]
        last_date = last_row['date']
        category = last_row['category']
        lead_time = int(last_row['lead_time_days'])
        closing_stock = float(last_row['closing_stock'])
        demand_cv = float(last_row['demand_cv'])
        rop_days = lead_time * (1 + np.clip(demand_cv, 0.1, 0.8))
        max_stock_days = rop_days * k
        last_cat_dem_ma7 = float(last_row['cat_dem_ma7'])

        history = g['units_sold'].tail(14).tolist()

        # ── TODAY'S TRUTH — computed once, from real data, independent
        # of any forecast. This is what 'inventory_status' should
        # actually show — exactly the original day-0 formula, before
        # any projection assumptions enter the picture.
        current_avg_demand = np.mean(history[-7:])
        current_doi = closing_stock / (current_avg_demand + 1e-5)
        current_status = (
            'understock' if current_doi < rop_days else
            'overstock' if current_doi > max_stock_days else
            'optimal'
        )
        
        projected_stock = closing_stock
        breach_day = None
        overstock_days = 0
        horizon = min(lead_time, max_horizon_cap)

        # inside generate_forecast, right after current_status is computed
        if current_status == 'understock':
            breach_day = 0   # already breached — not a future event
        else:
            breach_day = None  # forward loop below will set this if it finds one

        for h in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=h)
            lag1 = history[-1]
            lag7 = history[-7] if len(history) >= 7 else np.mean(history)
            roll_mean7 = np.mean(history[-7:])
            roll_mean14 = np.mean(history[-14:])
            roll_std7 = np.std(history[-7:]) if len(history[-7:]) > 1 else 0.0
            eps = 1e-5
            momentum = (roll_mean7 + eps) / (roll_mean14 + eps)
            sku_cat_share = (roll_mean7 + eps) / (last_cat_dem_ma7 + eps)

            X_future = pd.DataFrame([{
                'sku_id': sku_id, 'category': category, 'lead_time_days': lead_time,
                'day_of_week': future_date.dayofweek, 'day_of_month': future_date.day,
                'month': future_date.month, 'is_weekend': int(future_date.dayofweek >= 5),
                'lag1_units_sold': lag1, 'lag7_units_sold': lag7,
                'ma7_units_sold': roll_mean7, 'ma14_units_sold': roll_mean14,
                'vol14_units_sold': roll_std7, 'demand_momentum': momentum,
                'demand_cv': demand_cv, 'cat_dem_ma7': last_cat_dem_ma7,
                'sku_category_share': sku_cat_share,
            }])
            X_future['sku_id'] = X_future['sku_id'].astype(df['sku_id'].dtype)
            X_future['category'] = X_future['category'].astype(df['category'].dtype)
            X_future = X_future[feature_cols]

            pred_demand = max(0.0, float(model.predict(X_future)[0]))
            history.append(pred_demand)
            projected_stock -= pred_demand
            projected_doi = projected_stock / (roll_mean7 + eps)

            if breach_day is None and projected_doi < rop_days:
                breach_day = h
            if projected_doi > max_stock_days:
                overstock_days += 1

            forecast_rows.append({
                'sku_id': sku_id, 'category': category, 'date': future_date,
                'horizon_step': h, 'lead_time_days': lead_time,
                'forecasted_demand': round(pred_demand, 1),
                'projected_closing_stock': round(max(projected_stock, 0), 1),
                'forecasted_days_of_inventory': round(max(projected_doi, 0), 2),
                'ROP_days': round(rop_days, 2), 'max_stock_days': round(max_stock_days, 2),
                'days_to_breach': breach_day,
                'overstock_days_in_horizon': overstock_days,
                'closing_stock_today': closing_stock,
                # carried constant across every row for this SKU —
                # today's truth, not affected by how far the walk has gone
                'current_days_of_inventory': round(current_doi, 2),
                'current_status': current_status,
                'is_forecast': True,
            })

    return pd.DataFrame(forecast_rows)


def build_sku_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()

    last_rows = (
        forecast_df.sort_values(['sku_id', 'horizon_step'])
        .groupby('sku_id', as_index=False, observed=True)
        .last()
    )

    # this is the actual fix: inventory_status now comes from TODAY'S
    # real position, not the last day of a depleting-to-zero simulation
    last_rows['inventory_status'] = last_rows['current_status']

    def action_for(row):
        if row['current_status'] == 'understock':
            return 'error', "🚨 Already below reorder point — order needed now."
        elif pd.notna(row['days_to_breach']) and row['days_to_breach'] > 0:
            days = int(row['days_to_breach'])
            urgency = 'error' if days <= 3 else 'warning'
            return urgency, f"⚠️ Currently optimal, but projected to breach reorder point in {days} day{'s' if days != 1 else ''}."
        elif row['overstock_days_in_horizon'] >= 3:
            return 'warning', f"📦 Projected to stay overstocked for {int(row['overstock_days_in_horizon'])} of the next {int(row['horizon_step'])} days — capital tied up."
        else:
            return 'success', "✅ Stock level healthy across the forecast horizon."

    alert_types, actions = zip(*last_rows.apply(action_for, axis=1))
    last_rows['alert_type'] = alert_types
    last_rows['action'] = actions

    last_rows['target_stock_units'] = (last_rows['max_stock_days'] * last_rows['forecasted_demand']).round(0)
    last_rows['recommended_order'] = 0
    understock_mask = last_rows['inventory_status'] == 'understock'
    overstock_mask = last_rows['inventory_status'] == 'overstock'
    last_rows.loc[understock_mask, 'recommended_order'] = np.maximum(
        0, np.ceil(last_rows.loc[understock_mask, 'target_stock_units'] - last_rows.loc[understock_mask, 'closing_stock_today'])
    ).astype(int)
    last_rows['excess_stock'] = 0
    last_rows.loc[overstock_mask, 'excess_stock'] = np.maximum(
        0, np.floor(last_rows.loc[overstock_mask, 'closing_stock_today'] - last_rows.loc[overstock_mask, 'target_stock_units'])
    ).astype(int)

    last_rows['forecast_confidence'] = np.where(
        last_rows['sku_id'].isin(LOW_CONFIDENCE_SKUS), 'low', 'normal'
    )

    return last_rows


def enrich_warehouse_snapshot(df: pd.DataFrame, model, feature_cols: list = FEATURE_COLS,
                               k: float = 1.35, trained_categories: dict = None) -> dict:
    """
    The executive-summary entry point. Returns a dict with three pieces —
    app.py needs small updates to consume this shape (see notes below).
    """
    hist_fitted = fit_historical(df, model, feature_cols, trained_categories)
    forecast_df = generate_forecast(df, model, feature_cols, k, MAX_HORIZON_CAP, trained_categories)
    summary_df = build_sku_summary(forecast_df)

    return {
        'historical': hist_fitted,
        'forecast': forecast_df,
        'summary': summary_df,
    }


def forecast_single_sku(enriched: dict, target_sku_id: str) -> dict:
    """
    Pulls one SKU's summary + builds chart data combining real history
    with genuine future forecast rows (previously the chart had no
    future dates at all — this is the fix for that).
    """
    hist = enriched['historical']
    forecast = enriched['forecast']
    summary = enriched['summary']

    sku_summary = summary[summary['sku_id'] == target_sku_id]
    if sku_summary.empty:
        raise ValueError(f"SKU {target_sku_id} not found.")
    sku_summary = sku_summary.iloc[0]

    hist_slice = hist[hist['sku_id'] == target_sku_id].sort_values('date')
    forecast_slice = forecast[forecast['sku_id'] == target_sku_id].sort_values('date')

    hist_chart = hist_slice[['date', 'units_sold', 'fitted_demand']].rename(
        columns={'fitted_demand': 'forecasted_demand'})
    hist_chart['units_sold_actual'] = hist_chart['units_sold']
    future_chart = forecast_slice[['date', 'forecasted_demand']].copy()
    future_chart['units_sold_actual'] = np.nan  # no actual exists yet for future dates

    chart_df = pd.concat([
        hist_chart[['date', 'units_sold_actual', 'forecasted_demand']],
        future_chart[['date', 'units_sold_actual', 'forecasted_demand']],
    ], ignore_index=True)

    return {
        'sku_id': target_sku_id,
        'category': sku_summary['category'],
        'closing_stock': float(sku_summary['closing_stock_today']),
        'lead_time_days': int(sku_summary['lead_time_days']),
        'forecasted_demand': float(sku_summary['forecasted_demand']),
        'runway_days': float(sku_summary['forecasted_days_of_inventory']),
        'status': sku_summary['inventory_status'],
        'action': sku_summary['action'],
        'alert_type': sku_summary['alert_type'],
        'recommended_order': int(sku_summary['recommended_order']),
        'days_to_breach': sku_summary['days_to_breach'],
        'history': chart_df,
    }