import streamlit as st
import pandas as pd
import joblib
import gdown
import os
from triage import enrich_warehouse_snapshot, forecast_single_sku
from pipeline import build_pipeline_from_raw

DRIVE_ARTIFACTS = {
    "model": {
        "id": "1ClElZybePUlAr6yrOUXqxXZ9Gr_b3aRH",
        "path": "fmn_demand_forecast_model.pkl"
    },
    "features": {
        "id": "1B7tBjrOqFBT6x-nfjULifGq3pazr2Ov0",
        "path": "demand_feature_schema.pkl"
    },
    "data": {
        "id": "1c4ju-VN5FipKPeVdhXrr5H-ZsCv2PA_O",
        "path": "project1_supply_chain_demand.csv"
    }
}

@st.cache_resource(show_spinner="Fetching model artifacts from Drive...")
def load_all_artifacts():
    os.makedirs("artifacts", exist_ok=True)
    for name, meta in DRIVE_ARTIFACTS.items():
        if not os.path.exists(meta["path"]):
            download_url = f"https://drive.google.com/uc?id={meta['id']}"
            gdown.download(download_url, meta["path"], quiet=False)

    model = joblib.load(DRIVE_ARTIFACTS["model"]["path"])
    feature_cols = joblib.load(DRIVE_ARTIFACTS["features"]["path"])
    base_df = pd.read_csv(DRIVE_ARTIFACTS["data"]["path"])
    base_df['date'] = pd.to_datetime(base_df['date'])
    return model, feature_cols, base_df


@st.cache_data
def get_warehouse_data():
    model, features, base_df = load_all_artifacts()
    df = build_pipeline_from_raw(base_df)
    df['date'] = pd.to_datetime(df['date'])
    enriched = enrich_warehouse_snapshot(df, model, features, k=1.35)
    return enriched


st.set_page_config(page_title="FMN Inventory Optimization", layout="wide", page_icon="🌾")

model, feature_cols, base_df = load_all_artifacts()
enriched = get_warehouse_data()
summary_df = enriched['summary']

st.title("🌾 Flour Mills of Nigeria — Inventory Triage Engine")

# ---------------------------------------------------------
# Section 1: Plant-Wide Overview
# ---------------------------------------------------------
n_under = (summary_df['inventory_status'] == 'understock').sum()
n_over  = (summary_df['inventory_status'] == 'overstock').sum()
n_opt   = (summary_df['inventory_status'] == 'optimal').sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Active SKUs", len(summary_df))
col2.metric("Stockout Risk (Understock)", n_under, delta=f"-{n_under}", delta_color="inverse")
col3.metric("Optimal Healthy Lines", n_opt)
col4.metric("Capital Tied (Overstock)", n_over)

st.divider()

# ---------------------------------------------------------
# Section 2: SKU Filter & Detailed Drill-Down
# ---------------------------------------------------------
st.subheader("🔍 SKU Operational Inspector")

selected_sku = st.selectbox("Select SKU to inspect:", options=summary_df['sku_id'].unique())

sku_info = forecast_single_sku(enriched, selected_sku)

if sku_info['alert_type'] == 'error':
    st.error(sku_info['action'])
elif sku_info['alert_type'] == 'warning':
    st.warning(sku_info['action'])
else:
    st.success(sku_info['action'])

if sku_info.get('forecast_confidence') == 'low':
    st.caption("⚠️ This SKU's demand forecast has historically been less reliable than average — treat the flag above as directional, and confirm manually before acting.")
# Metric Tiles — rewritten: lead_time_dma and cvr no longer exist
# (both were derived from the old flat-rate extrapolation, removed
# when the recursive walk-forward replaced it). days_to_breach is
# the new, more directly useful number in their place.
m1, m2, m3, m4 = st.columns(4)
m1.metric("Closing Stock", f"{int(sku_info['closing_stock']):,} units")
m2.metric("Forecasted Daily Demand", f"{sku_info['forecasted_demand']} units")
m3.metric("Runway Remaining", f"{sku_info['runway_days']} days")

days_to_breach = sku_info['days_to_breach']
breach_display = f"{int(days_to_breach)} days" if pd.notna(days_to_breach) else "None projected"
m4.metric("Days to Reorder-Point Breach", breach_display)

# Historical Actuals vs. Forward Forecast Chart
st.markdown("##### Historical Sales vs. Forward Demand Forecast")
chart_data = sku_info['history'].set_index('date')[['units_sold_actual', 'forecasted_demand']]
st.line_chart(chart_data)

st.divider()

# ---------------------------------------------------------
# Section 3: Master Warehouse Triage Board
# ---------------------------------------------------------
st.subheader("📋 Warehouse Triage Board (All SKUs)")

def color_status(val):
    if val == 'understock':
        return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;'
    elif val == 'overstock':
        return 'background-color: #bbdefb; color: #0d47a1; font-weight: bold;'
    return 'background-color: #c8e6c9; color: #1b5e20;'

display_cols = [
    'sku_id', 'category', 'closing_stock_today', 'lead_time_days',
    'forecasted_demand', 'current_days_of_inventory',   # was: forecasted_days_of_inventory
    'inventory_status', 'days_to_breach', 'overstock_days_in_horizon',   # added
    'recommended_order', 'forecast_confidence',
]

st.dataframe(
    summary_df[display_cols].style.map(color_status, subset=['inventory_status']),
    use_container_width=True,
    height=320
)