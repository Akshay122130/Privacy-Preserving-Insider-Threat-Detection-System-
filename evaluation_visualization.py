"""
Standalone script to generate model evaluation visualizations.
Run with: python evaluation_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Model evaluation metrics ---
ACCURACY = 0.9455
PRECISION = 0.8636
RECALL = 0.9681
F1_SCORE = 0.9046

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-score"]
METRIC_VALUES = [ACCURACY, PRECISION, RECALL, F1_SCORE]

# --- Confusion matrix (Actual x Predicted) ---
CONFUSION_MATRIX = np.array([
    [44, 3],
    [0, 8],
])
CLASS_LABELS = ["Normal", "Threat"]

# --- Output settings ---
DPI = 300


def plot_metrics_bar_chart(save_path: str = "metrics_bar_chart.png") -> None:
    """Create bar chart of evaluation metrics and save to file."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METRIC_NAMES))
    bars = ax.bar(x, METRIC_VALUES, color="steelblue", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Evaluation Metrics")

    for bar, val in zip(bars, METRIC_VALUES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_matrix_heatmap(save_path: str = "confusion_matrix.png") -> None:
    """Create confusion matrix heatmap and save to file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        CONFUSION_MATRIX,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    plot_metrics_bar_chart()
    plot_confusion_matrix_heatmap()
    print("Done.")


if __name__ == "__main__":
    main()
