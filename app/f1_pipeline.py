"""Reusable preprocessing components for the F1 finish prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_FEATURES = [
    "grid",
    "driver_age",
    "driver_race_count",
    "driver_prev_finish_rate",
    "driver_last5_finish_rate",
    "constructor_race_count",
    "constructor_prev_finish_rate",
    "constructor_prev_avg_grid",
    "constructor_last5_finish_rate",
    "circuit_finish_rate",
    "circuit_avg_grid",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "best_q_time",
    "grid_normalized",
    "has_qualifying",
    "front_row_start",
    "top10_start",
    "driver_nationality_encoded",
    "circuitRef_encoded",
]


LEAKAGE_COLUMNS = ["laps"]


DERIVED_FEATURES = ["experience_ratio", "grid_above_avg", "avg_finish_rate"]


DEFAULT_SELECTED_FEATURES = [
    "grid",
    "driver_race_count",
    "driver_prev_finish_rate",
    "driver_last5_finish_rate",
    "constructor_race_count",
    "constructor_prev_finish_rate",
    "constructor_prev_avg_grid",
    "constructor_last5_finish_rate",
    "circuit_finish_rate",
    "circuit_avg_grid",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "has_qualifying",
    "top10_start",
    "driver_nationality_encoded",
    "circuitRef_encoded",
    "experience_ratio",
    "grid_above_avg",
    "avg_finish_rate",
]


CATEGORICAL_FEATURES = [
    "has_qualifying",
    "top10_start",
    "driver_nationality_encoded",
    "circuitRef_encoded",
    "grid_above_avg",
]


WINSORIZE_COLUMNS = ["driver_age", "constructor_race_count", "driver_race_count"]


def smotenc_feature_indices(features: list[str] | None = None) -> list[int]:
    """Return categorical feature indices expected by SMOTENC after selection."""

    selected = features or DEFAULT_SELECTED_FEATURES
    return [i for i, col in enumerate(selected) if col in CATEGORICAL_FEATURES]


class LeakageColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are unavailable before the race outcome is known."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or LEAKAGE_COLUMNS

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        frame = _to_dataframe(X)
        return frame.drop(columns=[col for col in self.columns if col in frame.columns])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create deterministic pre-race features used by both training and inference."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        frame = _to_dataframe(X).copy()
        frame["experience_ratio"] = frame["driver_race_count"] / (
            frame["constructor_race_count"] + 1
        )
        frame["grid_above_avg"] = (
            frame["grid"] > frame["constructor_prev_avg_grid"]
        ).astype(int)
        frame["avg_finish_rate"] = (
            frame["driver_prev_finish_rate"] + frame["constructor_prev_finish_rate"]
        ) / 2
        return frame


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select and order the final model features."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or DEFAULT_SELECTED_FEATURES

    def fit(self, X, y=None):
        frame = _to_dataframe(X)
        missing = [col for col in self.columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.columns_ = list(self.columns)
        return self

    def transform(self, X):
        frame = _to_dataframe(X)
        return frame.loc[:, self.columns].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns, dtype=object)


@dataclass
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip selected numeric columns using quantiles learned only from training data."""

    columns: list[str] | None = None
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    def fit(self, X, y=None):
        frame = _to_dataframe(X)
        cols = self.columns or WINSORIZE_COLUMNS
        self.columns_ = [col for col in cols if col in frame.columns]
        self.bounds_ = {
            col: (
                frame[col].quantile(self.lower_quantile),
                frame[col].quantile(self.upper_quantile),
            )
            for col in self.columns_
        }
        return self

    def transform(self, X):
        frame = _to_dataframe(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in frame.columns:
                frame[col] = frame[col].clip(lower=lower, upper=upper)
        return frame

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array(input_features, dtype=object)


def _to_dataframe(X):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)
