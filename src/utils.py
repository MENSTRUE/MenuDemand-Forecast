from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_forecast_result(
    forecast_df: pd.DataFrame,
    output_dir: str | Path = "tests/results",
    filename: str = "latest_forecast.csv",
) -> Path:
    output_dir = ensure_directory(output_dir)
    output_path = output_dir / filename

    forecast_df.to_csv(output_path, index=False)

    return output_path


def top_n_forecast(
    forecast_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    return (
        forecast_df
        .sort_values("forecast_units", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )