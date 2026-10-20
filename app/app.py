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


# =========================================================
# SIDEBAR
# =========================================================

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


# =========================================================
# HEADER
# =========================================================

st.title("☕ MenuDemand Forecast")

st.write(
    """
    Upload historical daily product sales and generate
    a **next-day demand forecast**.
    """
)


# =========================================================
# REQUIRED FORMAT
# =========================================================

with st.expander(
    "📄 Required CSV format",
    expanded=False,
):
    st.markdown(
        """
        Required columns:
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
        "Every product must contain at least "
        "28 days of historical demand."
    )


# =========================================================
# STEP 1 — UPLOAD
# =========================================================

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


# =========================================================
# STEP 2 — VALIDATION
# =========================================================

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
    start_date = (
        validation.min_date.strftime(
            "%Y-%m-%d"
        )
        if validation.min_date is not None
        else "-"
    )

    st.metric(
        "Start",
        start_date,
    )


with c4:
    last_date = (
        validation.max_date.strftime(
            "%Y-%m-%d"
        )
        if validation.max_date is not None
        else "-"
    )

    st.metric(
        "Last Date",
        last_date,
    )


with st.expander(
    "Preview historical data",
):
    st.dataframe(
        history_df.tail(100),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STEP 3 — METHOD
# =========================================================

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

        Preserved as a learned-model comparison.

        It is not the operational champion because
        Rolling-28 achieved lower validation MAE.
        """
    )


# =========================================================
# STEP 4 — GENERATE FORECAST
# =========================================================

st.subheader("4. Generate Forecast")


generate = st.button(
    "🚀 Generate Next-Day Forecast",
    type="primary",
    use_container_width=True,
)


if generate:
    try:
        with st.spinner(
            "Generating forecast..."
        ):
            forecast_df = forecaster.forecast(
                history_df,
                method=method,
            )

        st.session_state[
            "forecast_df"
        ] = forecast_df

        st.session_state[
            "forecast_method"
        ] = method

    except Exception as exc:
        st.error(
            f"Forecast failed: {exc}"
        )

        st.stop()


if "forecast_df" not in st.session_state:
    st.stop()


forecast_df = st.session_state[
    "forecast_df"
]


# =========================================================
# FORECAST RESULT
# =========================================================

st.success(
    "✅ Forecast generated successfully."
)


forecast_date = pd.to_datetime(
    forecast_df[
        "forecast_date"
    ].iloc[0]
)


st.subheader(
    f"Forecast for {forecast_date.strftime('%Y-%m-%d')}"
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
        str(
            forecast_df[
                "method"
            ].iloc[0]
        ),
    )


# =========================================================
# TOP PRODUCTS
# =========================================================

st.divider()

st.subheader(
    "Highest Forecast Demand"
)


max_top = min(
    30,
    len(forecast_df),
)


if max_top >= 5:
    top_n = st.slider(
        "Products to display",
        min_value=5,
        max_value=max_top,
        value=min(
            15,
            max_top,
        ),
    )

else:
    top_n = max_top


top_df = (
    forecast_df
    .sort_values(
        "forecast_units",
        ascending=False,
    )
    .head(
        top_n
    )
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


# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader(
    "Forecast Table"
)


table_columns = [
    "ProductDisplay",
    "Category",
    "forecast_units",
    "forecast_units_rounded",
    "method",
]


if "Price" in forecast_df.columns:
    table_columns.insert(
        2,
        "Price",
    )


table = forecast_df[
    table_columns
].copy()


rename_map = {
    "ProductDisplay": "Product",
    "Category": "Category",
    "Price": "Price",
    "forecast_units": "Forecast",
    "forecast_units_rounded": "Rounded",
    "method": "Method",
}


table = table.rename(
    columns=rename_map
)


st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# PRODUCT INSPECTOR
# =========================================================

st.subheader(
    "Product Inspector"
)


product_names = (
    forecast_df[
        "ProductDisplay"
    ]
    .astype(str)
    .tolist()
)


selected_product = st.selectbox(
    "Choose product",
    product_names,
)


product = forecast_df[
    forecast_df[
        "ProductDisplay"
    ].astype(str)
    == selected_product
].iloc[0]


p1, p2, p3, p4 = st.columns(4)


with p1:
    st.metric(
        "Forecast",
        f"{float(product['forecast_units']):.2f}",
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
        str(
            product[
                "Category"
            ]
        ),
    )


with p4:
    if "Price" in product.index:
        price = pd.to_numeric(
            product["Price"],
            errors="coerce",
        )

        if pd.notna(price):
            st.metric(
                "Price",
                f"{float(price):,.0f}",
            )
        else:
            st.metric(
                "Price",
                "-",
            )

    else:
        st.metric(
            "Price",
            "-",
        )


# =========================================================
# OPTIONAL PRODUCT HISTORY
# =========================================================

st.subheader(
    "Recent Product History"
)


selected_history = history_df[
    history_df[
        "ProductDisplay"
    ].astype(str)
    == selected_product
].copy()


if not selected_history.empty:
    selected_history[
        "Date"
    ] = pd.to_datetime(
        selected_history[
            "Date"
        ],
        errors="coerce",
    )

    selected_history = (
        selected_history
        .sort_values(
            "Date"
        )
        .tail(28)
    )

    history_chart = (
        selected_history[
            [
                "Date",
                "demand",
            ]
        ]
        .set_index(
            "Date"
        )
    )

    st.line_chart(
        history_chart,
        use_container_width=True,
    )


# =========================================================
# STEP 5 — EXPORT
# =========================================================

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
        try:
            output_path = save_forecast_result(
                forecast_df,
                ROOT
                / "tests"
                / "results",
            )

            st.success(
                "Saved: "
                f"{output_path.relative_to(ROOT)}"
            )

        except Exception as exc:
            st.error(
                f"Could not save result: {exc}"
            )


with col_download:
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=dataframe_to_csv_bytes(
            forecast_df
        ),
        file_name=(
            "menudemand_"
            f"{forecast_date.strftime('%Y-%m-%d')}"
            ".csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# INTERPRETATION
# =========================================================

with st.expander(
    "⚠️ How to interpret this forecast",
):
    st.markdown(
        """
        The forecast estimates **next-day product demand**
        using historical daily sales.

        ### Rolling-28

        The operational default uses the mean demand
        from the previous 28 days for each product.

        ### LightGBM

        The LightGBM model is preserved as the best
        learned-model experiment.

        It is shown as a comparison option rather than
        replacing the operational champion.

        ### Important limitations

        This MVP currently does not explicitly model:

        - stock availability;
        - stockouts;
        - promotions;
        - discounts;
        - weather;
        - holidays;
        - public events;
        - inventory constraints;
        - outlet-level demand differences.

        Forecast values are therefore estimates based on
        historical demand patterns, not guaranteed sales.
        """
    )