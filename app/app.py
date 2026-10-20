from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.inference import MenuDemandForecaster
from src.utils import save_forecast_result


st.set_page_config(
    page_title="MenuDemand Forecast",
    page_icon="☕",
    layout="wide",
)


st.title("☕ MenuDemand Forecast")

st.caption(
    "Next-day menu-item demand forecasting using a validation-selected "
    "Rolling-28 operational baseline."
)


with st.sidebar:
    st.header("Project")

    st.markdown(
        """
        **Operational Champion**  
        Rolling-28

        **Best Learned Model**  
        LightGBM L1

        **Granularity**  
        Product × Day

        **Target**  
        Units sold
        """
    )

    st.info(
        "The default operational forecast uses Rolling-28 because it achieved "
        "the lowest validation MAE."
    )


forecaster = MenuDemandForecaster(
    ROOT / "models" / "menudemand_best_learned_lgbm.joblib"
)


forecast_path = ROOT / "reports" / "next_day_forecast.csv"

if not forecast_path.exists():
    st.error(
        f"Forecast artifact not found: {forecast_path}"
    )
    st.stop()


forecast_df = forecaster.load_precomputed_forecast(forecast_path)


st.subheader("Next-Day Forecast")

forecast_date = None

if "forecast_date" in forecast_df.columns:
    forecast_date = forecast_df["forecast_date"].iloc[0]

if forecast_date is not None:
    st.write(f"Forecast date: **{forecast_date}**")


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Products",
        len(forecast_df),
    )

with col2:
    st.metric(
        "Forecast Units",
        f"{forecast_df['forecast_units'].sum():.1f}",
    )

with col3:
    st.metric(
        "Rounded Units",
        int(forecast_df["forecast_units_rounded"].sum()),
    )


st.divider()


top_n = st.slider(
    "Top products to display",
    min_value=5,
    max_value=min(30, len(forecast_df)),
    value=min(15, len(forecast_df)),
)


top_df = (
    forecast_df
    .sort_values("forecast_units", ascending=False)
    .head(top_n)
)


st.subheader("Highest Forecast Demand")

chart_df = (
    top_df[
        [
            "ProductDisplay",
            "forecast_units",
        ]
    ]
    .set_index("ProductDisplay")
)

st.bar_chart(chart_df)


st.subheader("Forecast Table")

display_cols = [
    "ProductDisplay",
    "Category",
    "forecast_units",
    "forecast_units_rounded",
]

if "forecast_date" in forecast_df.columns:
    display_cols.insert(0, "forecast_date")

st.dataframe(
    forecast_df[display_cols],
    use_container_width=True,
    hide_index=True,
)


st.divider()


st.subheader("Product Lookup")

product_names = forecast_df["ProductDisplay"].tolist()

selected_product = st.selectbox(
    "Select a menu item",
    product_names,
)

row = forecast_df[
    forecast_df["ProductDisplay"] == selected_product
].iloc[0]


c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Forecast",
        f"{row['forecast_units']:.2f}",
    )

with c2:
    st.metric(
        "Rounded",
        int(row["forecast_units_rounded"]),
    )

with c3:
    st.metric(
        "Category",
        row["Category"],
    )


st.divider()


st.subheader("Save Result")

if st.button("Save Forecast to tests/results"):
    output_path = save_forecast_result(
        forecast_df,
        ROOT / "tests" / "results",
        "menu_demand_forecast.csv",
    )

    st.success(
        f"Saved to: {output_path.relative_to(ROOT)}"
    )


with st.expander("Model & Methodology"):
    st.markdown(
        """
        ### Operational method

        The deployment default is **Rolling-28**:

        > Forecast next-day product demand using the mean demand observed
        > during the previous 28 days.

        This baseline achieved the best validation MAE.

        ### Learned model

        A LightGBM L1 model is also preserved as the best learned model.

        It is not automatically used as the operational champion because
        model selection was locked using validation results before viewing
        the held-out test set.

        ### Limitation

        This MVP does not model:

        - stock availability;
        - promotions;
        - weather;
        - special events;
        - inventory constraints;
        - outlet-specific demand.

        Forecasts should therefore be interpreted as historical-demand
        estimates, not guaranteed future sales.
        """
    )