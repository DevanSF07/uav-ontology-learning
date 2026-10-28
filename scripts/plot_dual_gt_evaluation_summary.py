#!/usr/bin/env python3
"""Plot the dual GT evaluation summary CSV as a grouped bar chart.

The script reads a CSV produced by the precision-first ontology pipeline and
renders precision, recall, and F1 for broad, structural, and local-after-
seed-similarity runs across the strict, safe_extended, and context_extended
levels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/sseshagi/onto_gen_product/ontology_experiment_framework/stage_outputs/precision_first_ontology_all_extractors/dual_gt_evaluation_summary.csv"
)

LEVEL_ORDER = {"strict": 0, "safe_extended": 1, "context_extended": 2}
EVALUATION_ORDER = [
    "broad_accepted",
    "structural_accepted",
    "local_after_broad_strict_accepted",
    "local_after_structural_strict_accepted",
]
METRICS = ["precision", "recall", "f1"]

DISPLAY_NAMES = {
    "broad_accepted": "Broad Accepted",
    "structural_accepted": "Structural Accepted",
    "local_after_broad_strict_accepted": "Local After Broad Seed Similarity",
    "local_after_structural_strict_accepted": "Local After Structural Seed Similarity",
}


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"evaluation_name", "level", "precision", "recall", "f1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame = frame[frame["evaluation_name"].isin(EVALUATION_ORDER)].copy()
    frame["level_order"] = frame["level"].map(LEVEL_ORDER).fillna(99).astype(int)
    frame["evaluation_order"] = frame["evaluation_name"].map({name: index for index, name in enumerate(EVALUATION_ORDER)}).fillna(99).astype(int)
    frame = frame.sort_values(["evaluation_order", "level_order", "level"]).reset_index(drop=True)
    return frame


def render_grouped_bars(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    levels = ["strict", "safe_extended", "context_extended"]
    metric_colors = {"precision": "#2F6BFF", "recall": "#FF9F1C", "f1": "#2CA58D"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes_flat = list(axes.flat)
    for axis in axes_flat[len(EVALUATION_ORDER):]:
        axis.axis("off")

    for axis, evaluation_name in zip(axes_flat, EVALUATION_ORDER, strict=True):
        subset = frame[frame["evaluation_name"] == evaluation_name].copy()
        subset = subset.set_index("level").reindex(levels)

        x = range(len(levels))
        width = 0.24

        for offset, metric in enumerate(METRICS):
            values = subset[metric].fillna(0.0).to_list()
            positions = [index + (offset - 1) * width for index in x]
            axis.bar(
                positions,
                values,
                width=width,
                label=metric.capitalize(),
                color=metric_colors[metric],
                edgecolor="white",
                linewidth=0.8,
            )
            for xpos, value in zip(positions, values):
                axis.text(
                    xpos,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

        axis.set_title(DISPLAY_NAMES.get(evaluation_name, evaluation_name.replace("_", " ").title()), fontsize=12, fontweight="bold")
        axis.set_xticks(list(x))
        axis.set_xticklabels([level.replace("_", " ") for level in levels], rotation=0)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_alpha(0.3)
        axis.spines["bottom"].set_alpha(0.3)

    axes_flat[0].set_ylabel("Score")
    axes_flat[0].legend(loc="upper right", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a visual summary plot from dual_gt_evaluation_summary.csv."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output image path. Defaults to the input CSV name with a .png suffix.",
    )
    parser.add_argument(
        "--title",
        default="Broad, Structural, and Local After Seed Similarity",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the rendered plot in a window after saving.",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)

    output_path = args.output_png or args.input_csv.with_suffix(".png")
    summary = load_summary(args.input_csv)
    if summary.empty:
        raise ValueError(
            "No supported broad, structural, or local-after-seed-similarity rows were found in the CSV."
        )
    render_grouped_bars(summary, output_path, args.title)

    print(f"Saved plot to: {output_path}")

    if args.show:
        image = plt.imread(output_path)
        plt.figure(figsize=(14, 5.8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()