from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocess import WINE_FEATURE_COLUMNS, WINE_TARGET_COLUMN, read_csv_auto_sep, preprocess_wine_dataframe


@dataclass(frozen=True)
class RetrainConfig:
    base_data_path: Path = Path("data/winequality-red.csv")
    # Optional labeled new data to improve the model over time (must include `quality` + all features).
    new_data_path: Path | None = Path("data/new_wine_data.csv")
    models_dir: Path = Path("models")
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    n_estimators: int = 300
    max_depth: int | None = None
    n_jobs: int = 1


def _next_version(models_dir: Path) -> str:
    models_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in models_dir.glob("model_v*.pkl"):
        name = p.stem  # model_v12
        if not name.startswith("model_v"):
            continue
        try:
            n = int(name.split("model_v", 1)[1])
        except Exception:
            continue
        max_n = max(max_n, n)
    return f"v{max_n + 1}"


def _split_xy_binary_from_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = preprocess_wine_dataframe(df)
    X = df[WINE_FEATURE_COLUMNS]
    y = (df[WINE_TARGET_COLUMN] >= 7).astype(int)
    y.name = "good_wine"
    return X, y


def retrain(cfg: RetrainConfig) -> dict:
    base_df, sep = read_csv_auto_sep(cfg.base_data_path)
    dfs = [base_df]

    if cfg.new_data_path is not None and Path(cfg.new_data_path).exists():
        new_df, _new_sep = read_csv_auto_sep(Path(cfg.new_data_path))
        dfs.append(new_df)

    df_all = pd.concat(dfs, ignore_index=True)
    X_df, y_s = _split_xy_binary_from_quality(df_all)
    X = X_df.to_numpy()
    y = y_s.to_numpy()

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

    y_pred = model.predict(X_test_s) if len(y_test) else []
    accuracy = float(accuracy_score(y_test, y_pred)) if len(y_test) else float("nan")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist() if len(y_test) else None

    version = _next_version(cfg.models_dir)
    model_path = cfg.models_dir / f"model_{version}.pkl"
    scaler_path = cfg.models_dir / f"scaler_{version}.pkl"
    info_path = cfg.models_dir / f"model_info_{version}.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    def _pos_rate(arr) -> float:
        return float(sum(int(v) for v in arr) / len(arr)) if len(arr) else float("nan")

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    cfg_dict["version"] = version
    cfg_dict["model_path"] = str(model_path)
    cfg_dict["scaler_path"] = str(scaler_path)
    cfg_dict["info_path"] = str(info_path)

    info = {
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(cfg.base_data_path),
        "csv_sep_detected": sep,
        "new_data_used": str(cfg.new_data_path) if (cfg.new_data_path and Path(cfg.new_data_path).exists()) else None,
        "n_samples_total": int(len(df_all)),
        "features": WINE_FEATURE_COLUMNS,
        "target": "good_wine",
        "target_rule": "quality >= 7 => 1 else 0",
        "model": "RandomForestClassifier",
        "metrics": {
            "accuracy": accuracy,
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": cm,
        },
        "split": {
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "stratified": do_stratify,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "pos_rate_train": _pos_rate(y_train),
            "pos_rate_test": _pos_rate(y_test),
        },
        "artifacts": {"model": str(model_path), "scaler": str(scaler_path), "info": str(info_path)},
        "config": cfg_dict,
    }

    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    out = retrain(RetrainConfig())
    print(json.dumps(out, indent=2))

