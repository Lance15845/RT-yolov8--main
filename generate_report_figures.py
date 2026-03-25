from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "voc"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = RESULTS_DIR / "report_figures"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

MODEL_ORDER = ["baseline", "mobilenetv2", "pcg_ghost"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "mobilenetv2": "MobileNetV2",
    "pcg_ghost": "PCG-Ghost",
}
COLORS = {
    "baseline": "#1f4e79",
    "mobilenetv2": "#d97a1d",
    "pcg_ghost": "#3d8b5a",
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(COMPARISON_CSV)
    if not rows:
        raise FileNotFoundError(f"No comparison rows found in {COMPARISON_CSV}")

    ordered = [row for key in MODEL_ORDER for row in rows if row["model_key"] == key]
    generate_scatter(ordered)
    generate_bar_dashboard(ordered)
    generate_delta_chart(ordered)
    generate_qualitative_panel(ordered)

    print("Generated report figures:")
    for path in sorted(REPORT_DIR.glob("*.png")):
        print(path)
    return 0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_scatter(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for row in rows:
        key = row["model_key"]
        latency = to_float(row["latency_ms"])
        map50 = to_float(row["map50"])
        params_m = to_float(row["params_m"])
        size = 900 * params_m
        ax.scatter(
            latency,
            map50,
            s=size,
            c=COLORS[key],
            alpha=0.88,
            edgecolors="black",
            linewidths=0.8,
            label=MODEL_LABELS[key],
        )
        ax.annotate(
            f"{MODEL_LABELS[key]}\n{map50:.3f}, {latency:.2f} ms",
            (latency, map50),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("VOC Accuracy-Speed Tradeoff", fontsize=14, weight="bold")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP50")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True)

    fig.savefig(REPORT_DIR / "01_accuracy_speed_scatter.png", dpi=220)
    plt.close(fig)


def generate_bar_dashboard(rows: list[dict[str, str]]) -> None:
    labels = [MODEL_LABELS[row["model_key"]] for row in rows]
    map50 = [to_float(row["map50"]) for row in rows]
    map5095 = [to_float(row["map5095"]) for row in rows]
    latency = [to_float(row["latency_ms"]) for row in rows]
    fps = [to_float(row["fps"]) for row in rows]
    colors = [COLORS[row["model_key"]] for row in rows]
    x = range(len(rows))
    width = 0.34

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    ax1.bar([i - width / 2 for i in x], map50, width=width, color=colors, alpha=0.95, label="mAP50")
    ax1.bar([i + width / 2 for i in x], map5095, width=width, color=colors, alpha=0.45, label="mAP50-95")
    ax1.set_title("Detection Accuracy", fontsize=13, weight="bold")
    ax1.set_ylabel("Score")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(map50) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=True)

    for idx, value in enumerate(map50):
        ax1.text(idx - width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    for idx, value in enumerate(map5095):
        ax1.text(idx + width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)

    ax2.bar([i - width / 2 for i in x], latency, width=width, color=colors, alpha=0.95, label="Latency (ms)")
    ax2.bar([i + width / 2 for i in x], fps, width=width, color=colors, alpha=0.45, label="FPS")
    ax2.set_title("Runtime Performance", fontsize=13, weight="bold")
    ax2.set_ylabel("Value")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(max(latency), max(fps)) * 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(frameon=True)

    for idx, value in enumerate(latency):
        ax2.text(idx - width / 2, value + 1.2, f"{value:.2f}", ha="center", fontsize=9)
    for idx, value in enumerate(fps):
        ax2.text(idx + width / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)

    fig.savefig(REPORT_DIR / "02_accuracy_runtime_bars.png", dpi=220)
    plt.close(fig)


def generate_delta_chart(rows: list[dict[str, str]]) -> None:
    labels = [MODEL_LABELS[row["model_key"]] for row in rows]
    deltas = [to_float(row["delta_map50_vs_baseline"]) for row in rows]
    colors = [COLORS[row["model_key"]] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.bar(labels, deltas, color=colors, alpha=0.9)
    ax.axhline(3.0, color="#b22222", linestyle="--", linewidth=2, label="3% threshold")
    ax.set_title("Relative mAP50 Drop vs Baseline", fontsize=14, weight="bold")
    ax.set_ylabel("Drop (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True)

    for idx, value in enumerate(deltas):
        ax.text(idx, value + 0.8, f"{value:.2f}%", ha="center", fontsize=10)

    fig.savefig(REPORT_DIR / "03_delta_vs_baseline.png", dpi=220)
    plt.close(fig)


def generate_qualitative_panel(rows: list[dict[str, str]]) -> None:
    figure_paths = [
        FIGURES_DIR / "baseline_predictions.png",
        FIGURES_DIR / "mobilenetv2_predictions.png",
        FIGURES_DIR / "pcg_ghost_predictions.png",
    ]
    missing = [str(path) for path in figure_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing qualitative figure(s): {', '.join(missing)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, row, path in zip(axes, rows, figure_paths, strict=True):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(MODEL_LABELS[row["model_key"]], fontsize=13, weight="bold")
        ax.axis("off")

    fig.suptitle("Qualitative Prediction Comparison", fontsize=16, weight="bold")
    fig.savefig(REPORT_DIR / "04_qualitative_comparison.png", dpi=220)
    plt.close(fig)


def to_float(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
