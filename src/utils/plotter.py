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
    plt.xlabel("Voting Session (Round)")
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


def plot_bias_diagnostics(
    table: list[dict],
    output_path: Path,
    title: str = "Bias Diagnostics",
    dpi: int = 160,
) -> None:
    metric_names = [
        "vote_share_mae",
        "probability_mae",
        "preference_mae",
        "mean_rank_error",
        "decision_flip",
    ]
    metric_names = [m for m in metric_names if any(row["metric"] == m for row in table)]
    algo_names = sorted({row["algorithm"] for row in table})

    if not metric_names or not algo_names:
        return

    x = np.arange(len(metric_names))
    width = min(0.24, 0.75 / max(len(algo_names), 1))

    plt.figure(figsize=(11, 5))
    for idx, algo in enumerate(algo_names):
        vals = []
        for metric in metric_names:
            matched = [
                float(row["value"])
                for row in table
                if row["algorithm"] == algo and row["metric"] == metric
            ]
            vals.append(matched[0] if matched else np.nan)
        offset = (idx - (len(algo_names) - 1) / 2.0) * width
        plt.bar(x + offset, vals, width=width, label=algo)

    plt.xticks(x, metric_names, rotation=18)
    plt.ylabel("Mean value")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_bias_effect_bars(
    table: list[dict],
    output_path: Path,
    title: str = "Vote Bias and Ranking Deviation",
    dpi: int = 160,
) -> None:
    scenario_names = sorted({row["scenario"] for row in table})
    algo_names = sorted({row["algorithm"] for row in table})

    if not scenario_names or not algo_names:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(scenario_names))
    width = min(0.24, 0.75 / max(len(algo_names), 1))

    for idx, algo in enumerate(algo_names):
        offset = (idx - (len(algo_names) - 1) / 2.0) * width
        vote_vals = []
        rank_vals = []
        for scenario in scenario_names:
            matched = [
                row
                for row in table
                if row["scenario"] == scenario and row["algorithm"] == algo
            ]
            vote_vals.append(float(matched[0]["vote_share_mae"]) if matched else np.nan)
            rank_vals.append(float(matched[0]["mean_rank_error"]) if matched else np.nan)
        axes[0].bar(x + offset, vote_vals, width=width, label=algo)
        axes[1].bar(x + offset, rank_vals, width=width, label=algo)

    axes[0].set_title("Vote-share deviation")
    axes[0].set_ylabel("Mean absolute error")
    axes[1].set_title("Final ranking deviation")
    axes[1].set_ylabel("Mean rank error")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=18)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_movie_ratings(
    movie_names: list[str],
    rating_data: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Estimated Movie Ratings (1–10 scale)",
    dpi: int = 160,
) -> None:
    """Bar chart comparing estimated vs true movie ratings on a 1–10 scale.

    rating_data keys are algorithm names or "True Quality"; values are arrays
    of length len(movie_names) with scores in [1, 10].
    """
    n_movies = len(movie_names)
    series_names = list(rating_data.keys())
    x = np.arange(n_movies)
    width = min(0.18, 0.8 / max(len(series_names), 1))

    # Sort movies by true quality (descending) for readability.
    sort_key = "True Quality" if "True Quality" in rating_data else series_names[0]
    order = np.argsort(-rating_data[sort_key])
    sorted_names = [movie_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(14, n_movies * 0.8), 5))

    for idx, series in enumerate(series_names):
        vals = rating_data[series][order]
        offset = (idx - (len(series_names) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=series, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 11)
    ax.set_ylabel("Rating (1–10 scale)")
    ax.set_title(title)
    ax.axhline(5.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
