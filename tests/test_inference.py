import pandas as pd

from src.inference import MenuDemandForecaster


def test_rolling_28_forecast():
    rows = []

    for product_id in [1, 2]:
        for day in range(30):
            rows.append(
                {
                    "ProductId": product_id,
                    "ProductDisplay": f"Product {product_id}",
                    "Category": "Coffee",
                    "Date": f"2025-01-{day + 1:02d}"
                    if day < 31
                    else None,
                    "demand": float(product_id),
                }
            )

    df = pd.DataFrame(rows)

    result = MenuDemandForecaster.rolling_28_forecast(df)

    assert len(result) == 2
    assert result["forecast_units"].notna().all()
    assert set(result["method"]) == {"Rolling-28"}