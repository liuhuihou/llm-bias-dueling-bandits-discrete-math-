from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_regret_curves(
    summary: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    title: str = "Cumulative Regret Comparison",
    dpi: int = 160,
    figsize: tuple[int, int] = (9, 5),
) -> None:
    plt.figure(figsize=figsize)
    for algo_name, stats in summary.items():
        mean = stats["mean"]
        std = stats["std"]
        x = np.arange(1, len(mean) + 1)

        plt.plot(x, mean, label=algo_name, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_robustness_bars(
    table: list[dict],
    output_path: Path,
    title: str = "Relative Final Regret Across Bias Scenarios",
    dpi: int = 160,
) -> None:
    scenario_names = sorted({row["scenario"] for row in table})
    algo_names = sorted({row["algorithm"] for row in table})

    x = np.arange(len(scenario_names))
    width = 0.22

    plt.figure(figsize=(10, 5))
    for idx, algo in enumerate(algo_names):
        vals = []
        for scenario in scenario_names:
            matched = [r["relative_final_regret"] for r in table if r["scenario"] == scenario and r["algorithm"] == algo]
            vals.append(matched[0] if matched else np.nan)
        offset = (idx - (len(algo_names) - 1) / 2.0) * width
        plt.bar(x + offset, vals, width=width, label=algo)

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.xticks(x, scenario_names, rotation=15)
    plt.ylabel("Relative Final Regret (vs scenario best)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
