from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.inference import (
    MenuDemandForecaster,
    validate_history,
)
from src.utils import (
    dataframe_to_csv_bytes,
    save_forecast_result,
)


MODEL_PATH = (
    ROOT
    / "models"
    / "menudemand_best_learned_lgbm.joblib"
)


st.set_page_config(
    page_title="MenuDemand Forecast",
    page_icon="☕",
    layout="wide",
)


@st.cache_resource
def load_forecaster():
    return MenuDemandForecaster(
        model_path=MODEL_PATH
    )


forecaster = load_forecaster()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("MenuDemand")

    st.markdown(
        """
        **Operational Champion**  
        Rolling-28

        **Best Learned Model**  
        LightGBM L1

        **Task**  
        Next-day product demand

        **Granularity**  
        Product × Day
        """
    )

    st.divider()

    st.caption(
        "Rolling-28 is the default because "
        "it achieved the lowest validation MAE."
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("☕ MenuDemand Forecast")

st.write(
    """
    Upload historical daily product sales and generate
    a **next-day demand forecast**.
    """
)


# ---------------------------------------------------------
# FORMAT EXPLANATION
# ---------------------------------------------------------

with st.expander(
    "📄 Required CSV format",
    expanded=False,
):
    st.markdown(
        """
        Your CSV must contain:
        """
    )

    example = pd.DataFrame(
        {
            "Date": [
                "2025-09-01",
                "2025-09-02",
                "2025-09-03",
            ],
            "ProductId": [
                3,
                3,
                3,
            ],
            "ProductDisplay": [
                "Air Mineral",
                "Air Mineral",
                "Air Mineral",
            ],
            "Category": [
                "Other",
                "Other",
                "Other",
            ],
            "Price": [
                5000,
                5000,
                5000,
            ],
            "demand": [
                1,
                3,
                2,
            ],
        }
    )

    st.dataframe(
        example,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Every product must have at least 28 daily rows."
    )


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

st.subheader("1. Upload Sales History")

uploaded_file = st.file_uploader(
    "Upload historical daily sales CSV",
    type=["csv"],
)


if uploaded_file is None:
    st.info(
        "Upload a CSV file to start forecasting."
    )

    st.stop()


try:
    history_df = pd.read_csv(
        uploaded_file
    )
except Exception as exc:
    st.error(
        f"Could not read CSV: {exc}"
    )
    st.stop()


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

st.subheader("2. Validate Data")

validation = validate_history(
    history_df
)


if not validation.valid:
    st.error(
        validation.message
    )

    st.stop()


st.success(
    "✅ Historical data is valid."
)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Rows",
        f"{validation.rows:,}",
    )

with c2:
    st.metric(
        "Products",
        validation.products,
    )

with c3:
    st.metric(
        "Start",
        validation.min_date.date(),
    )

with c4:
    st.metric(
        "Last Date",
        validation.max_date.date(),
    )


with st.expander(
    "Preview historical data",
):
    st.dataframe(
        history_df.tail(100),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# METHOD
# ---------------------------------------------------------

st.subheader("3. Choose Forecast Method")


methods = [
    "Rolling-28",
]

if forecaster.model is not None:
    methods.append(
        "LightGBM"
    )


method = st.radio(
    "Forecast method",
    methods,
    horizontal=True,
)


if method == "Rolling-28":
    st.info(
        """
        **Rolling-28 — Operational Champion**

        Forecast = mean demand from the previous
        28 days for each product.
        """
    )

else:
    st.warning(
        """
        **LightGBM L1 — Best Learned Model**

        Preserved for comparison.

        It was not the operational champion because
        Rolling-28 achieved lower validation MAE.
        """
    )


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------

st.subheader("4. Generate Forecast")


generate = st.button(
    "🚀 Generate Next-Day Forecast",
    type="primary",
    use_container_width=True,
)


if not generate:
    st.stop()


try:
    with st.spinner(
        "Generating forecast..."
    ):
        forecast_df = forecaster.forecast(
            history_df,
            method=method,
        )

except Exception as exc:
    st.error(
        f"Forecast failed: {exc}"
    )
    st.stop()


# Keep forecast in session state.
st.session_state[
    "forecast_df"
] = forecast_df


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

st.success(
    "Forecast generated successfully."
)


forecast_date = pd.to_datetime(
    forecast_df["forecast_date"].iloc[0]
)


st.subheader(
    f"Forecast for {forecast_date.date()}"
)


m1, m2, m3, m4 = st.columns(4)


with m1:
    st.metric(
        "Products",
        len(forecast_df),
    )


with m2:
    st.metric(
        "Forecast Units",
        f"{forecast_df['forecast_units'].sum():.1f}",
    )


with m3:
    st.metric(
        "Rounded Units",
        int(
            forecast_df[
                "forecast_units_rounded"
            ].sum()
        ),
    )


with m4:
    st.metric(
        "Method",
        forecast_df[
            "method"
        ].iloc[0],
    )


# ---------------------------------------------------------
# TOP PRODUCTS
# ---------------------------------------------------------

st.divider()

st.subheader(
    "Highest Forecast Demand"
)


max_top = min(
    30,
    len(forecast_df),
)


top_n = st.slider(
    "Products to display",
    min_value=5,
    max_value=max_top,
    value=min(15, max_top),
)


top_df = forecast_df.head(
    top_n
)


chart = (
    top_df[
        [
            "ProductDisplay",
            "forecast_units",
        ]
    ]
    .set_index(
        "ProductDisplay"
    )
)


st.bar_chart(
    chart,
    use_container_width=True,
)


# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------

st.subheader(
    "Forecast Table"
)


table = forecast_df[
    [
        "ProductDisplay",
        "Category",
        "Price",
        "forecast_units",
        "forecast_units_rounded",
        "method",
    ]
].copy()


table = table.rename(
    columns={
        "ProductDisplay": "Product",
        "Category": "Category",
        "Price": "Price",
        "forecast_units": "Forecast",
        "forecast_units_rounded": "Rounded",
        "method": "Method",
    }
)


st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# PRODUCT INSPECTOR
# ---------------------------------------------------------

st.subheader(
    "Product Inspector"
)


selected_product = st.selectbox(
    "Choose product",
    forecast_df[
        "ProductDisplay"
    ].tolist(),
)


product = forecast_df[
    forecast_df[
        "ProductDisplay"
    ]
    == selected_product
].iloc[0]


p1, p2, p3, p4 = st.columns(4)


with p1:
    st.metric(
        "Forecast",
        f"{product['forecast_units']:.2f}",
    )


with p2:
    st.metric(
        "Rounded",
        int(
            product[
                "forecast_units_rounded"
            ]
        ),
    )


with p3:
    st.metric(
        "Category",
        product["Category"],
    )


with p4:
    st.metric(
        "Price",
        f"{float(product['Price']):,.0f}",
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

st.divider()

st.subheader(
    "5. Export Result"
)


col_save, col_download = st.columns(
    2
)


with col_save:
    if st.button(
        "💾 Save to tests/results",
        use_container_width=True,
    ):
        output_path = save_forecast_result(
            forecast_df,
            ROOT
            / "tests"
            / "results",
        )

        st.success(
            f"Saved: "
            f"{output_path.relative_to(ROOT)}"
        )


with col_download:
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=dataframe_to_csv_bytes(
            forecast_df
        ),
        file_name=(
            f"menudemand_"
            f"{forecast_date.date()}"
            f".csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

with st.expander(
    "⚠️ How to interpret this forecast",
):
    st.markdown(
        """
        Forecast values estimate **next-day visible product demand**
        based on historical sales.

        They are **not guaranteed sales**.

        The MVP does not currently account for:

        - stockouts;
        - promotions;
        - weather;
        - public events;
        - holidays;
        - inventory capacity;
        - outlet-specific demand.

        The operational default is Rolling-28 because it achieved
        the lowest validation MAE in the project evaluation.
        """
    )