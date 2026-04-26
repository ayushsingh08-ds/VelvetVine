from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_model_info(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    max_val = float(cm.max()) if cm.size else 0.0
    threshold = max_val / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_bar(versions: list[str], accuracies: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    colors = ["#4E79A7", "#F28E2B"][: len(versions)]
    bars = ax.bar(versions, accuracies, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_evaluation_artifacts(models_dir: Path = Path("models"), assets_dir: Path = Path("web/assets")) -> dict:
    info_v1 = _load_model_info(models_dir / "model_info_v1.json")
    info_v2 = _load_model_info(models_dir / "model_info_v2.json")

    versions = [info_v1["version"], info_v2["version"]]
    accuracies = [
        float(info_v1["metrics"]["accuracy"]),
        float(info_v2["metrics"]["accuracy"]),
    ]
    cm_v1 = np.array(info_v1["metrics"]["confusion_matrix"], dtype=int)
    cm_v2 = np.array(info_v2["metrics"]["confusion_matrix"], dtype=int)

    accuracy_chart = assets_dir / "model_accuracy_comparison.png"
    cm_v1_chart = assets_dir / "confusion_matrix_v1.png"
    cm_v2_chart = assets_dir / "confusion_matrix_v2.png"

    _plot_accuracy_bar(versions, accuracies, accuracy_chart)
    _plot_confusion_matrix(cm_v1, "Confusion Matrix (v1)", cm_v1_chart)
    _plot_confusion_matrix(cm_v2, "Confusion Matrix (v2)", cm_v2_chart)

    return {
        "accuracy_chart": str(accuracy_chart),
        "confusion_matrix_v1_chart": str(cm_v1_chart),
        "confusion_matrix_v2_chart": str(cm_v2_chart),
        "accuracy_v1": accuracies[0],
        "accuracy_v2": accuracies[1],
    }


if __name__ == "__main__":
    output = generate_evaluation_artifacts()
    print(json.dumps(output, indent=2))
