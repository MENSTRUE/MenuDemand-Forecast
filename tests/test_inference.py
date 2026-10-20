import numpy as np
import pandas as pd

from src.inference import (
    MenuDemandForecaster,
    validate_history,
)


def create_history():
    dates = pd.date_range(
        "2025-01-01",
        periods=35,
        freq="D",
    )

    rows = []

    for product_id, demand in [
        (1, 2.0),
        (2, 4.0),
    ]:
        for date in dates:
            rows.append(
                {
                    "Date": date,
                    "ProductId": product_id,
                    "ProductDisplay": (
                        f"Product {product_id}"
                    ),
                    "Category": "Coffee",
                    "Price": 20000,
                    "demand": demand,
                }
            )

    return pd.DataFrame(rows)


def test_validate_history():
    df = create_history()

    result = validate_history(df)

    assert result.valid
    assert result.products == 2


def test_rolling28_forecast():
    df = create_history()

    forecaster = (
        MenuDemandForecaster(
            model_path="does_not_exist.joblib"
        )
    )

    result = (
        forecaster.forecast_rolling_28(
            df
        )
    )

    assert len(result) == 2

    first = result[
        result["ProductId"] == 1
    ].iloc[0]

    second = result[
        result["ProductId"] == 2
    ].iloc[0]

    assert np.isclose(
        first["forecast_units"],
        2.0,
    )

    assert np.isclose(
        second["forecast_units"],
        4.0,
    )

    assert (
        first["forecast_units_rounded"]
        == 2
    )

    assert (
        second["forecast_units_rounded"]
        == 4
    )


def test_next_forecast_date():
    df = create_history()

    forecaster = (
        MenuDemandForecaster(
            model_path="does_not_exist.joblib"
        )
    )

    result = forecaster.forecast(
        df,
        method="Rolling-28",
    )

    expected = (
        pd.Timestamp("2025-02-04")
        + pd.Timedelta(days=1)
    )

    actual = pd.to_datetime(
        result[
            "forecast_date"
        ].iloc[0]
    )

    assert actual == expected