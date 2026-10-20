from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd


ForecastMethod = Literal["Rolling-28", "LightGBM"]

DEFAULT_MODEL_PATH = Path("models/menudemand_best_learned_lgbm.joblib")

REQUIRED_HISTORY_COLUMNS = {
    "Date",
    "ProductId",
    "ProductDisplay",
    "Category",
    "Price",
    "demand",
}

FEATURE_COLUMNS = [
    "ProductId",
    "Category",
    "Price",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "expanding_mean",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_COLUMNS = [
    "ProductId",
    "Category",
]


@dataclass
class ValidationResult:
    valid: bool
    message: str
    rows: int
    products: int
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None


def validate_history(df: pd.DataFrame) -> ValidationResult:
    missing = REQUIRED_HISTORY_COLUMNS - set(df.columns)

    if missing:
        return ValidationResult(
            valid=False,
            message=f"Missing columns: {', '.join(sorted(missing))}",
            rows=len(df),
            products=0,
            min_date=None,
            max_date=None,
        )

    if df.empty:
        return ValidationResult(
            valid=False,
            message="CSV is empty.",
            rows=0,
            products=0,
            min_date=None,
            max_date=None,
        )

    work = df.copy()

    try:
        work["Date"] = pd.to_datetime(work["Date"], errors="raise")
    except Exception:
        return ValidationResult(
            valid=False,
            message="Column 'Date' contains invalid dates.",
            rows=len(work),
            products=0,
            min_date=None,
            max_date=None,
        )

    work["demand"] = pd.to_numeric(work["demand"], errors="coerce")
    work["Price"] = pd.to_numeric(work["Price"], errors="coerce")

    if work["demand"].isna().any():
        return ValidationResult(
            valid=False,
            message="Column 'demand' contains non-numeric values.",
            rows=len(work),
            products=work["ProductId"].nunique(),
            min_date=work["Date"].min(),
            max_date=work["Date"].max(),
        )

    if (work["demand"] < 0).any():
        return ValidationResult(
            valid=False,
            message="Demand cannot be negative.",
            rows=len(work),
            products=work["ProductId"].nunique(),
            min_date=work["Date"].min(),
            max_date=work["Date"].max(),
        )

    product_counts = work.groupby("ProductId")["Date"].nunique()
    insufficient = product_counts[product_counts < 28]

    if len(insufficient) > 0:
        return ValidationResult(
            valid=False,
            message=(
                f"{len(insufficient)} product(s) have fewer than 28 days "
                "of history. Rolling-28 requires at least 28 days/product."
            ),
            rows=len(work),
            products=work["ProductId"].nunique(),
            min_date=work["Date"].min(),
            max_date=work["Date"].max(),
        )

    return ValidationResult(
        valid=True,
        message="History is valid.",
        rows=len(work),
        products=work["ProductId"].nunique(),
        min_date=work["Date"].min(),
        max_date=work["Date"].max(),
    )


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    result = validate_history(df)

    if not result.valid:
        raise ValueError(result.message)

    work = df.copy()

    work["Date"] = pd.to_datetime(work["Date"], errors="raise")
    work["ProductId"] = pd.to_numeric(
        work["ProductId"],
        errors="raise",
    ).astype(int)

    work["Price"] = pd.to_numeric(
        work["Price"],
        errors="coerce",
    )

    work["demand"] = pd.to_numeric(
        work["demand"],
        errors="raise",
    ).astype(float)

    work["ProductDisplay"] = work["ProductDisplay"].astype(str)
    work["Category"] = work["Category"].astype(str)

    duplicate_mask = work.duplicated(
        subset=["ProductId", "Date"],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate ProductId + Date rows found. "
            "Aggregate demand to one row per product per day first."
        )

    return work.sort_values(
        ["ProductId", "Date"]
    ).reset_index(drop=True)


def _rolling_mean(y: np.ndarray, n: int) -> float:
    return float(np.mean(y[-n:]))


def _rolling_std(y: np.ndarray, n: int) -> float:
    values = y[-n:]

    if len(values) < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def build_next_day_features(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    history = prepare_history(history)

    next_date = history["Date"].max() + pd.Timedelta(days=1)

    rows = []

    for product_id, group in history.groupby("ProductId"):
        group = group.sort_values("Date")

        y = group["demand"].to_numpy(dtype=float)
        last = group.iloc[-1]

        if len(y) < 28:
            raise ValueError(
                f"ProductId={product_id} has fewer than 28 observations."
            )

        def lag(n: int) -> float:
            return float(y[-n])

        rows.append(
            {
                "ProductId": int(product_id),
                "ProductDisplay": str(last["ProductDisplay"]),
                "Category": str(last["Category"]),
                "Price": float(last["Price"]),
                "forecast_date": next_date,

                "lag_1": lag(1),
                "lag_2": lag(2),
                "lag_3": lag(3),
                "lag_7": lag(7),
                "lag_14": lag(14),
                "lag_28": lag(28),

                "rolling_mean_3": _rolling_mean(y, 3),
                "rolling_mean_7": _rolling_mean(y, 7),
                "rolling_mean_14": _rolling_mean(y, 14),
                "rolling_mean_28": _rolling_mean(y, 28),

                "rolling_std_7": _rolling_std(y, 7),
                "rolling_std_28": _rolling_std(y, 28),

                "expanding_mean": float(np.mean(y)),

                "day_of_week": int(next_date.dayofweek),
                "day_of_month": int(next_date.day),
                "week_of_year": int(next_date.isocalendar().week),
                "month": int(next_date.month),
                "is_weekend": int(next_date.dayofweek in [5, 6]),

                "dow_sin": float(
                    np.sin(2 * np.pi * next_date.dayofweek / 7)
                ),
                "dow_cos": float(
                    np.cos(2 * np.pi * next_date.dayofweek / 7)
                ),
                "month_sin": float(
                    np.sin(2 * np.pi * next_date.month / 12)
                ),
                "month_cos": float(
                    np.cos(2 * np.pi * next_date.month / 12)
                ),
            }
        )

    return pd.DataFrame(rows), next_date


class MenuDemandForecaster:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        self.model_path = Path(model_path)
        self.model = None

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def forecast_rolling_28(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        features, next_date = build_next_day_features(history)

        result = features[
            [
                "ProductId",
                "ProductDisplay",
                "Category",
                "Price",
            ]
        ].copy()

        result["forecast_date"] = next_date

        result["forecast_units"] = features[
            "rolling_mean_28"
        ].clip(lower=0)

        result["forecast_units_rounded"] = np.rint(
            result["forecast_units"]
        ).astype(int)

        result["method"] = "Rolling-28"

        return result.sort_values(
            "forecast_units",
            ascending=False,
        ).reset_index(drop=True)

    def forecast_lightgbm(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.model is None:
            raise FileNotFoundError(
                f"LightGBM model not found: {self.model_path}"
            )

        features, next_date = build_next_day_features(history)

        X = features[FEATURE_COLUMNS].copy()

        for column in CATEGORICAL_COLUMNS:
            X[column] = X[column].astype("category")

        prediction = self.model.predict(X)

        prediction = np.clip(
            np.asarray(prediction, dtype=float),
            0,
            None,
        )

        result = features[
            [
                "ProductId",
                "ProductDisplay",
                "Category",
                "Price",
            ]
        ].copy()

        result["forecast_date"] = next_date
        result["forecast_units"] = prediction

        result["forecast_units_rounded"] = np.rint(
            prediction
        ).astype(int)

        result["method"] = "LightGBM-L1"

        return result.sort_values(
            "forecast_units",
            ascending=False,
        ).reset_index(drop=True)

    def forecast(
        self,
        history: pd.DataFrame,
        method: ForecastMethod = "Rolling-28",
    ) -> pd.DataFrame:
        if method == "Rolling-28":
            return self.forecast_rolling_28(history)

        if method == "LightGBM":
            return self.forecast_lightgbm(history)

        raise ValueError(
            f"Unknown forecast method: {method}"
        )