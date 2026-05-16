from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from config.config import DEFAULT_CONFIG, FIGURES_DIR, PLOT_CONFIG, RAW_DATA_DIR, ensure_runtime_dirs
from src.algorithms.baselines import BSUCBAlgorithm, RUCBAlgorithm
from src.algorithms.dbs_ucb import DBSUCBAlgorithm
from src.data.synthetic_generator import BiasedDuelingEnvironment, SyntheticDuelingGenerator
from src.utils.metrics import (
    aggregate_regret,
    auc_regret,
    cumulative_regret,
    final_regret,
    instantaneous_regret,
    paired_permutation_test,
)
from src.utils.plotter import plot_regret_curves, plot_robustness_bars
from src.utils.reporting import build_latex_table_from_robustness, build_markdown_summary


def run_single_experiment(
    algorithm,
    environment: BiasedDuelingEnvironment,
    true_preference_matrix,
    best_arm: int,
    horizon: int,
):
    regret_trace = []
    for _ in range(horizon):
        i, j = algorithm.select_pair()
        winner = environment.duel(i, j)
        algorithm.update(i, j, winner)
        regret_trace.append(instantaneous_regret(best_arm, i, j, true_preference_matrix))
    return cumulative_regret(regret_trace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bias-robust dueling bandit benchmark.")
    parser.add_argument("--horizon", type=int, default=None, help="Override number of rounds per run.")
    parser.add_argument("--runs", type=int, default=None, help="Override number of Monte Carlo runs.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick debug mode: fewer runs and shorter horizon.",
    )
    args = parser.parse_args()
    if args.horizon is not None and args.horizon <= 0:
        parser.error("--horizon must be a positive integer.")
    if args.runs is not None and args.runs <= 0:
        parser.error("--runs must be a positive integer.")
    return args


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    if args.quick:
        cfg["horizon"] = 1000
        cfg["n_runs"] = 8
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.runs is not None:
        cfg["n_runs"] = args.runs

    ensure_runtime_dirs()

    algo_registry = {
        "RUCB": (RUCBAlgorithm, cfg["rucb_params"]),
        "BS-UCB": (BSUCBAlgorithm, cfg["bsucb_params"]),
        "DBS-UCB": (DBSUCBAlgorithm, cfg["dbs_params"]),
    }

    n_runs = int(cfg["n_runs"])
    horizon = int(cfg["horizon"])
    n_arms = int(cfg["n_arms"])
    seed = int(cfg["random_seed"])

    generator = SyntheticDuelingGenerator(random_state=seed)

    rows = []
    scenario_report = {}
    robustness_rows = []

    for scenario_idx, scenario in enumerate(cfg["scenarios"]):
        scenario_name = str(scenario["name"])
        all_curves: dict[str, list] = {name: [] for name in cfg["algorithms"]}
        final_regret_by_algo: dict[str, list[float]] = {name: [] for name in cfg["algorithms"]}
        auc_by_algo: dict[str, list[float]] = {name: [] for name in cfg["algorithms"]}

        for run_id in range(n_runs):
            true_pref_matrix = generator.generate_preference_matrix(
                n_arms=n_arms,
                temperature=float(cfg["preference_temperature"]),
                base_noise=float(cfg["base_noise"]),
            )

            best_arm = generator.find_condorcet_winner(true_pref_matrix)
            if best_arm is None:
                best_arm = int(true_pref_matrix.mean(axis=1).argmax())

            for algo_idx, name in enumerate(cfg["algorithms"]):
                env_seed = seed + scenario_idx * 10000 + run_id * 100 + algo_idx
                environment = BiasedDuelingEnvironment(
                    preference_matrix=true_pref_matrix,
                    position_bias=float(scenario["position_bias"]),
                    conformity_bias=float(scenario["conformity_bias"]),
                    selective_bias=float(scenario["selective_bias"]),
                    selective_threshold=float(scenario["selective_threshold"]),
                    random_state=env_seed,
                )

                algo_cls, algo_kwargs = algo_registry[name]
                algo = algo_cls(n_arms=n_arms, random_state=seed + run_id, **algo_kwargs)
                curve = run_single_experiment(
                    algorithm=algo,
                    environment=environment,
                    true_preference_matrix=true_pref_matrix,
                    best_arm=best_arm,
                    horizon=horizon,
                )

                all_curves[name].append(curve)
                final_regret_by_algo[name].append(final_regret(curve))
                auc_by_algo[name].append(auc_regret(curve))

        summary = {}
        for algo_name, run_curves in all_curves.items():
            mean, std = aggregate_regret(run_curves)
            summary[algo_name] = {"mean": mean, "std": std}
            for step_idx, (m, s) in enumerate(zip(mean, std), start=1):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "algorithm": algo_name,
                        "step": step_idx,
                        "mean_regret": m,
                        "std_regret": s,
                    }
                )

        plot_regret_curves(
            summary=summary,
            output_path=FIGURES_DIR / f"regret_{scenario_name}.png",
            title=f"{PLOT_CONFIG['title']} - {scenario_name}",
            dpi=PLOT_CONFIG["dpi"],
            figsize=PLOT_CONFIG["figsize"],
        )

        best_final = min(float(np.mean(v)) for v in final_regret_by_algo.values())
        for algo_name, values in final_regret_by_algo.items():
            robustness_rows.append(
                {
                    "scenario": scenario_name,
                    "algorithm": algo_name,
                    "relative_final_regret": float(np.mean(values) / max(best_final, 1e-12)),
                }
            )

        scenario_metrics = {
            "mean_final_regret": {k: float(np.mean(v)) for k, v in final_regret_by_algo.items()},
            "std_final_regret": {k: float(np.std(v)) for k, v in final_regret_by_algo.items()},
            "mean_auc_regret": {k: float(np.mean(v)) for k, v in auc_by_algo.items()},
            "significance_vs_dbs": {},
        }

        dbs = np.asarray(final_regret_by_algo["DBS-UCB"], dtype=float)
        for baseline in ["RUCB", "BS-UCB"]:
            base = np.asarray(final_regret_by_algo[baseline], dtype=float)
            p_val, improvement = paired_permutation_test(base, dbs, random_state=seed)
            scenario_metrics["significance_vs_dbs"][baseline] = {
                "p_value": float(p_val),
                "mean_improvement": float(improvement),
            }

        scenario_report[scenario_name] = scenario_metrics

    raw_data_path = RAW_DATA_DIR / "regret_summary.csv"
    with raw_data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "algorithm", "step", "mean_regret", "std_regret"],
        )
        writer.writeheader()
        writer.writerows(rows)

    robustness_data_path = RAW_DATA_DIR / "robustness_table.csv"
    with robustness_data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "algorithm", "relative_final_regret"],
        )
        writer.writeheader()
        writer.writerows(robustness_rows)

    robustness_fig_path = FIGURES_DIR / "robustness_comparison.png"
    plot_robustness_bars(robustness_rows, robustness_fig_path, dpi=PLOT_CONFIG["dpi"])

    report_path = RAW_DATA_DIR / "statistical_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(scenario_report, f, indent=2)

    build_markdown_summary(report_path, RAW_DATA_DIR / "statistical_summary.md")
    build_latex_table_from_robustness(robustness_data_path, RAW_DATA_DIR / "robustness_table.tex")

    print("Experiment finished.")
    print(f"Saved scenario figures in: {FIGURES_DIR}")
    print(f"Saved robustness figure: {robustness_fig_path}")
    print(f"Saved data: {raw_data_path}")
    print(f"Saved robustness table: {robustness_data_path}")
    print(f"Saved statistical report: {report_path}")


if __name__ == "__main__":
    main()
