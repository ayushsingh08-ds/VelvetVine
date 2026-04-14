from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix

from src.preprocess import WINE_FEATURE_COLUMNS, read_csv_auto_sep, split_xy_binary


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("data/winequality-red.csv")
    version: str = "v1"
    model_path: Path | None = None
    scaler_path: Path | None = None
    info_path: Path | None = None
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    n_estimators: int = 300
    max_depth: int | None = None
    n_jobs: int = 1
    calibrate: bool = True
    calibration_method: str = "sigmoid"


def train(cfg: TrainConfig) -> dict:
    df, sep = read_csv_auto_sep(cfg.data_path)
    X_df, y_s = split_xy_binary(df)
    X = X_df.to_numpy()
    y = y_s.to_numpy()

    version = cfg.version.strip()
    if not version:
        raise ValueError("version must be a non-empty string like 'v1'")
    model_path = cfg.model_path or Path(f"models/model_{version}.pkl")
    scaler_path = cfg.scaler_path or Path(f"models/scaler_{version}.pkl")
    info_path = cfg.info_path or Path(f"models/model_info_{version}.json")

    unique_classes = sorted(set(int(v) for v in y))
    do_stratify = cfg.stratify and (len(unique_classes) > 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if do_stratify else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        class_weight="balanced",
        n_jobs=cfg.n_jobs,
    )
    model.fit(X_train_s, y_train)
    if cfg.calibrate and (len(set(int(v) for v in y_train)) > 1):
        model = CalibratedClassifierCV(model, method=cfg.calibration_method, cv=5)
        model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s) if len(y_test) else []
    accuracy = float(accuracy_score(y_test, y_pred)) if len(y_test) else float("nan")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist() if len(y_test) else None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    def _pos_rate(arr) -> float:
        return float(sum(int(v) for v in arr) / len(arr)) if len(arr) else float("nan")

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    cfg_dict["model_path"] = str(model_path)
    cfg_dict["scaler_path"] = str(scaler_path)
    cfg_dict["info_path"] = str(info_path)

    info = {
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(cfg.data_path),
        "csv_sep_detected": sep,
        "features": WINE_FEATURE_COLUMNS,
        "target": "good_wine",
        "target_rule": "quality >= 7 => 1 else 0",
        "model": "CalibratedClassifierCV(RandomForestClassifier)" if cfg.calibrate else "RandomForestClassifier",
        "metrics": {
            "accuracy": accuracy,
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": cm,
        },
        "calibration": {"enabled": bool(cfg.calibrate), "method": cfg.calibration_method if cfg.calibrate else None},
        "split": {
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "stratified": do_stratify,
            "n_samples": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "pos_rate_overall": _pos_rate(y),
            "pos_rate_train": _pos_rate(y_train),
            "pos_rate_test": _pos_rate(y_test),
        },
        "config": cfg_dict,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    info = train(TrainConfig())
    print(json.dumps(info, indent=2))
