# Flour Mills of Nigeria — Inventory Triage Engine

## Problem Understanding

The sponsor's problem, in their own words: *"We keep getting caught off guard — some SKUs run out and delay production, others sit overstocked and tie up working capital. I don't have anything today that tells me, ahead of time, which SKUs need attention and why."*

Two failure modes, both needing attention, at two different timescales:

1. **Stockouts that delay production** — the mill needs to know a SKU is heading toward a stockout *before* it happens, with enough lead time to actually order more, not just a same-day alert once it's already too late.
2. **Overstock tying up working capital** — SKUs sitting on more inventory than their replenishment cycle justifies, quietly costing money without anyone noticing until someone asks why.

The deliverable requested was a **self-service tool** the Supply Chain team can open themselves, not a one-off analysis — showing upcoming demand per SKU, with each SKU that needs attention flagged in plain language, before it becomes a problem.

## Approach

The system is a two-tier pipeline, deliberately separating the part that genuinely needs machine learning from the part that doesn't:

**Tier 1 — Demand Forecasting (LightGBM regressor).** A single model, pooled across all SKUs, predicts daily `units_sold`. Trained on leakage-safe lag and rolling-window features (`lag1`, `lag7`, 7/14-day moving averages, volatility, momentum), calendar features, and category-hierarchy signals. Evaluated with MAE and WAPE (chosen over RMSE, which over-penalizes sporadic zero-demand days). LightGBM was selected after a head-to-head comparison against XGBoost, CatBoost, and a classical per-SKU exponential smoothing (ETS) baseline, the three ML models were statistically tied and all clearly outperformed ETS, justifying the added complexity of a pooled ML approach over a simpler classical one.

**Tier 2 — Inventory Risk Triage (deterministic logic, not a second model).** Rather than training a classifier to predict inventory status, this tier applies a transparent reorder-point formula directly to Tier 1's forecast:

- `ROP_days = lead_time_days × (1 + demand_cv.clip(0.1, 0.8))` — the reorder point, in days of cover, with a floor to protect new SKUs and a ceiling to guard against volatility outliers.
- `max_stock_days = ROP_days × 1.35` — the healthy ceiling, sized on standard FMCG replenishment-cycle guidance (0.5–1.0× lead time above the reorder point), not an arbitrary multiplier.
- Tier 1's forecast is walked forward, day by day, per SKU, out to that SKU's own lead time (not a fixed horizon for every SKU) — projecting stock depletion under a conservative no-incoming-receipts assumption — to determine **whether and when** a SKU is projected to breach its reorder point, or how long it will sit above the healthy ceiling.

This tier was deliberately kept as explicit, auditable arithmetic rather than a second trained model, the label a classifier would learn to predict is fully determined by known thresholds, so training a black-box model on it would only reproduce the same rule with less transparency, not add predictive value. This is also how real supply chain platforms are structured: a statistical forecasting engine feeding a deterministic policy engine.

**Output:** each SKU gets a status computed from **today's real stock position** (not a forward-projected snapshot, which would be misleadingly depleted by the projection's own no-replenishment assumption), plus a separate forward-looking `days_to_breach`- the actual early-warning signal and an `overstock_days_in_horizon` count for capital-tied-up risk.

**Delivery:** a Streamlit web app, deployed from this repo, giving the Supply Chain team a plant-wide overview, a per-SKU inspector with forecast chart, and a full triage board across all active SKUs.

## How to Run

### Local setup

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

### Dependencies

See `requirements.txt`. Core: `streamlit`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `joblib`.

### Model artifacts

The app loads a trained model, its feature schema, trained category mappings, and the source dataset. Depending on the current repo configuration, these are either committed directly to the repo or pulled at runtime from Google Drive via `gdown` — check `app.py`'s `DRIVE_ARTIFACTS` block / repo root to confirm which.

### Deployed link

`[TODO — add the live Streamlit Community Cloud URL here once deployment is confirmed working]`

## Limitations & Next Steps

**Model accuracy is moderate, not exceptional.** WAPE ~17.9% on the held-out test window — a real, validated improvement over a naive rolling-average baseline (~12% relative), but not precise enough to fully automate order *quantities* without human review. Treat flags as directional prioritization, not an auto-pilot.

**Five SKUs are consistently hard to forecast**, confirmed across four different modeling approaches (LightGBM, XGBoost, CatBoost, ETS all converged on the same weak set): `SKU-1011`, `SKU-1000`, `SKU-1007`, `SKU-1014`, `SKU-1020`. These are flagged in the app as lower-confidence, but the underlying cause (genuinely volatile demand vs. a data quality issue specific to those SKUs) hasn't been root-caused yet — worth a manual investigation before fully trusting their flags.

**The low-confidence SKU list is a static snapshot**, not a live mechanism — it won't update as new data comes in or as other SKUs' forecast accuracy drifts. A proper next step is a rolling per-SKU accuracy tracker that updates this dynamically.

**No incoming-receipts modeling.** The forward projection assumes zero replenishment during the forecast window — a deliberately conservative (worst-case) simplification. It doesn't account for a PO already in transit, which means `days_to_breach` is a floor estimate, not a precise date.

**AI-generated explanations and the grounded Q&A feature are not yet implemented.** The current SKU-level alert text is templated, not LLM-generated — this doesn't yet meet the "AI-generated explanations" and "grounded Q&A" requirements from the project brief. Designed but pending integration (requires an LLM API key and `anthropic` package).

**No action-confirmation loop.** Once a SKU is flagged, there's currently no way in the app to mark it as "ordered" or "addressed" — the same flag will keep showing until the underlying stock/forecast data changes. Worth adding for real day-to-day use.

**Cache refresh is manual.** The app's data pipeline is cached without a TTL — it won't pick up new source data automatically; someone has to clear the cache or restart the app. Fine for active development, not yet production-ready for unattended daily use.

**`trained_categories` wiring status should be reconfirmed** before relying on this in production — a mismatch between the categories a fresh data batch presents and what the model was actually trained on can silently produce wrong predictions for affected SKUs, with no error thrown.
