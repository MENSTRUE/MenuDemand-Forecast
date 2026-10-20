from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/menudemand_best_learned_lgbm.joblib")
DEFAULT_FORECAST_CONFIG = Path("reports/forecast_config.json")
DEFAULT_NEXT_FORECAST = Path("reports/next_day_forecast.csv")

OperationalMode = Literal["rolling28", "learned"]


@dataclass
class ForecastResult:
    product_id: int
    product_name: str
    category: str
    forecast_units: float
    forecast_units_rounded: int
    method: str


class MenuDemandForecaster:
    """
    Inference helper for MenuDemand Forecast.

    Default operational method:
        Rolling-28

    Optional comparison method:
        Best learned LightGBM model
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = None

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    @staticmethod
    def rolling_28_forecast(history: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast next-day demand using the mean of the previous 28 observed
        daily demand values for each product.

        Required columns:
            ProductId
            ProductDisplay
            Category
            Date
            demand
        """

        required = {
            "ProductId",
            "ProductDisplay",
            "Category",
            "Date",
            "demand",
        }

        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = history.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df["demand"] = pd.to_numeric(df["demand"], errors="raise")

        rows = []

        for product_id, group in df.groupby("ProductId"):
            group = group.sort_values("Date")

            if len(group) < 28:
                raise ValueError(
                    f"ProductId={product_id} only has {len(group)} daily rows. "
                    "Rolling-28 requires at least 28 days."
                )

            recent = group.tail(28)
            forecast = float(recent["demand"].mean())

            last = group.iloc[-1]

            rows.append(
                {
                    "ProductId": int(product_id),
                    "ProductDisplay": str(last["ProductDisplay"]),
                    "Category": str(last["Category"]),
                    "forecast_units": forecast,
                    "forecast_units_rounded": int(np.rint(forecast)),
                    "method": "Rolling-28",
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("forecast_units", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def load_precomputed_forecast(
        path: str | Path = DEFAULT_NEXT_FORECAST,
    ) -> pd.DataFrame:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Precomputed forecast not found: {path}"
            )

        df = pd.read_csv(path)

        expected = {
            "ProductId",
            "ProductDisplay",
            "Category",
            "forecast_units",
            "forecast_units_rounded",
        }

        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"Forecast CSV missing columns: {sorted(missing)}"
            )

        return df

    def has_learned_model(self) -> bool:
        return self.model is not None