from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    data_path: Path = Path("data/winequality-red.csv")
    output_path: Path = Path("data/wine_preprocessed.csv")


WINE_TARGET_COLUMN = "quality"
WINE_FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def read_csv_auto_sep(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Read a CSV while auto-detecting common separators.

    Wine Quality datasets are typically ';' separated.
    """
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df, sep
        except Exception:
            continue
    # Fallback: pandas default
    return pd.read_csv(path), ","


def preprocess_wine_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = WINE_FEATURE_COLUMNS + [WINE_TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[required]
    df = df.dropna()

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    return df.reset_index(drop=True)


def split_xy_binary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    X = all feature columns
    y = binary quality (1 = good wine if quality >= 7 else 0)
    """
    df = preprocess_wine_dataframe(df)
    X = df[WINE_FEATURE_COLUMNS]
    y_raw = df[WINE_TARGET_COLUMN]
    y = (y_raw >= 7).astype(int)
    y.name = "good_wine"
    return X, y


def run(cfg: PreprocessConfig) -> Path:
    df, _sep = read_csv_auto_sep(cfg.data_path)
    X, y = split_xy_binary(df)
    df_out = X.copy()
    df_out[y.name] = y
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.output_path, index=False)
    return cfg.output_path


if __name__ == "__main__":
    out = run(PreprocessConfig())
    print(f"Wrote {out}")
