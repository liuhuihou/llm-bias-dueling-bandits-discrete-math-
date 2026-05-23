from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from config.config import DEFAULT_CONFIG, FIGURES_DIR, PLOT_CONFIG, RAW_DATA_DIR, ensure_runtime_dirs
from src.algorithms.baselines import RREAlgorithm, UCBRAlgorithm
from src.algorithms.sbcr import SBCRAlgorithm
from src.data.synthetic_generator import (
    BiasedVotingEnvironment,
    MovieRatingSimulator,
    generate_movie_names,
)
from src.utils.metrics import (
    aggregate_regret,
    auc_regret,
    cumulative_regret,
    final_regret,
    instantaneous_regret,
    paired_permutation_test,
    preference_mae,
    rating_mae,
    spearman_rho,
)
from src.utils.plotter import (
    plot_bias_diagnostics,
    plot_bias_effect_bars,
    plot_movie_ratings,
    plot_regret_curves,
    plot_robustness_bars,
)
from src.utils.reporting import build_latex_table_from_robustness, build_markdown_summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_star_ratings(preference_matrix: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Convert a pairwise preference matrix to star ratings on [1, scale].

    R_i = 1 + (scale − 1) * mean_j P[i,j]
    """
    win_rates = preference_matrix.mean(axis=1)
    return 1.0 + (scale - 1.0) * win_rates


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


def ranking_diagnostics(
    estimated_preference_matrix: np.ndarray,
    true_preference_matrix: np.ndarray,
    best_arm: int,
) -> dict[str, float]:
    """Accuracy metrics for a single run.

    Returns:
        preference_mae  – MAE on upper-triangle of preference matrix.
        rating_mae      – MAE on 1-10 star ratings (most interpretable).
        spearman_rho    – Rank correlation of estimated vs true quality order.
        mean_rank_error – Mean absolute rank deviation across all movies.
        best_arm_rank   – Estimated rank of the true best movie (1 = correct).
        top1_correct    – 1 if the estimated best movie equals the true best.
    """
    true_scores = true_preference_matrix.mean(axis=1)
    estimated_scores = estimated_preference_matrix.mean(axis=1)
    true_ranks = _rank_positions(true_scores)
    estimated_ranks = _rank_positions(estimated_scores)
    estimated_best = int(np.argmax(estimated_scores))
    return {
        "preference_mae": preference_mae(estimated_preference_matrix, true_preference_matrix),
        "rating_mae": rating_mae(estimated_preference_matrix, true_preference_matrix),
        "spearman_rho": spearman_rho(estimated_preference_matrix, true_preference_matrix),
        "mean_rank_error": float(np.mean(np.abs(estimated_ranks - true_ranks))),
        "best_arm_rank": float(estimated_ranks[best_arm] + 1),
        "top1_correct": float(estimated_best == best_arm),
    }


# ── Single experiment ─────────────────────────────────────────────────────────

def run_single_experiment(
    algorithm,
    environment: BiasedVotingEnvironment,
    true_preference_matrix: np.ndarray,
    best_arm: int,
    horizon: int,
):
    regret_trace: list[float] = []
    bias_diagnostics: dict[str, list[float]] = {
        "vote_share_mae": [],
        "probability_mae": [],
        "noise_mae": [],
        "structural_bias_abs": [],
        "position_component_abs": [],
        "conformity_component_abs": [],
        "selective_component_abs": [],
        "decision_flip": [],
    }

    for _ in range(horizon):
        selected_i, selected_j = algorithm.select_pair()
        i, j = selected_i, selected_j
        # Randomise display order to allow symmetric averaging.
        if environment.rng.random() < 0.5:
            i, j = selected_j, selected_i

        winner = environment.duel(i, j)
        event = environment.last_event or {}
        vote_share_i = float(event.get("vote_share_i", 1.0 if winner == i else 0.0))
        audience_size = int(event.get("audience_size", 1))
        algorithm.update(i, j, winner, vote_share_i=vote_share_i, audience_size=audience_size)
        regret_trace.append(instantaneous_regret(best_arm, i, j, true_preference_matrix))

        true_p = float(event.get("true_probability", true_preference_matrix[i, j]))
        noisy_p = float(event.get("noisy_probability", true_p))
        observed_p = float(event.get("observed_probability", true_p))
        bias_diagnostics["vote_share_mae"].append(abs(vote_share_i - true_p))
        bias_diagnostics["probability_mae"].append(abs(observed_p - true_p))
        bias_diagnostics["noise_mae"].append(abs(noisy_p - true_p))
        bias_diagnostics["structural_bias_abs"].append(abs(float(event.get("bias_delta", 0.0))))
        bias_diagnostics["position_component_abs"].append(abs(float(event.get("position_component", 0.0))))
        bias_diagnostics["conformity_component_abs"].append(abs(float(event.get("conformity_component", 0.0))))
        bias_diagnostics["selective_component_abs"].append(abs(float(event.get("selective_component", 0.0))))
        bias_diagnostics["decision_flip"].append(float((true_p >= 0.5) != (observed_p >= 0.5)))

    if hasattr(algorithm, "debiased_preferences"):
        estimated_pref = algorithm.debiased_preferences()
    else:
        estimated_pref = algorithm.estimated_preferences()

    run_metrics: dict[str, float] = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in bias_diagnostics.items()
    }
    run_metrics.update(
        ranking_diagnostics(
            estimated_preference_matrix=estimated_pref,
            true_preference_matrix=true_preference_matrix,
            best_arm=best_arm,
        )
    )
    return cumulative_regret(regret_trace), run_metrics, estimated_pref


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Movie rating bias correction benchmark."
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 000 rounds, 8 runs.")
    args = parser.parse_args()
    if args.horizon is not None and args.horizon <= 0:
        parser.error("--horizon must be a positive integer.")
    if args.runs is not None and args.runs <= 0:
        parser.error("--runs must be a positive integer.")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

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
        "RRE":   (RREAlgorithm,  cfg["rre_params"]),
        "UCB-R": (UCBRAlgorithm, cfg["ucbr_params"]),
        "SBCR":  (SBCRAlgorithm, cfg["sbcr_params"]),
    }

    n_runs  = int(cfg["n_runs"])
    horizon = int(cfg["horizon"])
    n_arms  = int(cfg["n_arms"])
    seed    = int(cfg["random_seed"])

    simulator   = MovieRatingSimulator(random_state=seed)
    movie_names = generate_movie_names(n_arms, rng=np.random.default_rng(seed))

    rows: list[dict]             = []
    scenario_report: dict        = {}
    robustness_rows: list[dict]  = []
    diagnostic_rows: list[dict]  = []
    bias_effect_rows: list[dict] = []
    movie_rating_rows: list[dict]= []
    # For final summary table: scenario → algo → metric → mean value
    summary_table: dict[str, dict[str, dict[str, float]]] = {}

    for scenario_idx, scenario in enumerate(cfg["scenarios"]):
        scenario_name = str(scenario["name"])

        all_curves:          dict[str, list] = {n: [] for n in cfg["algorithms"]}
        final_regret_algo:   dict[str, list[float]] = {n: [] for n in cfg["algorithms"]}
        auc_algo:            dict[str, list[float]] = {n: [] for n in cfg["algorithms"]}
        diagnostics_algo:    dict[str, list[dict[str, float]]] = {n: [] for n in cfg["algorithms"]}
        estimated_prefs_algo:dict[str, list[np.ndarray]] = {n: [] for n in cfg["algorithms"]}
        true_prefs_runs: list[np.ndarray] = []

        for run_id in range(n_runs):
            true_pref = simulator.generate_preference_matrix(
                n_arms=n_arms,
                temperature=float(cfg["preference_temperature"]),
                base_noise=float(cfg["base_noise"]),
            )
            true_prefs_runs.append(true_pref)
            best_arm = simulator.find_condorcet_winner(true_pref)
            if best_arm is None:
                best_arm = int(true_pref.mean(axis=1).argmax())

            for algo_idx, name in enumerate(cfg["algorithms"]):
                env_seed  = seed + scenario_idx * 10000 + run_id * 100 + algo_idx
                env = BiasedVotingEnvironment(
                    preference_matrix=true_pref,
                    position_bias=float(scenario["position_bias"]),
                    conformity_bias=float(scenario["conformity_bias"]),
                    selective_bias=float(scenario["selective_bias"]),
                    selective_threshold=float(scenario["selective_threshold"]),
                    judgement_noise=float(scenario.get("judgement_noise",
                                          cfg.get("judgement_noise", 0.035))),
                    audience_size=int(scenario.get("audience_size",
                                      cfg.get("audience_size", 7))),
                    random_state=env_seed,
                )
                algo_cls, algo_kw = algo_registry[name]
                algo = algo_cls(n_arms=n_arms, random_state=seed + run_id, **algo_kw)
                curve, run_metrics, est_pref = run_single_experiment(
                    algorithm=algo,
                    environment=env,
                    true_preference_matrix=true_pref,
                    best_arm=best_arm,
                    horizon=horizon,
                )
                all_curves[name].append(curve)
                final_regret_algo[name].append(final_regret(curve))
                auc_algo[name].append(auc_regret(curve))
                diagnostics_algo[name].append(run_metrics)
                estimated_prefs_algo[name].append(est_pref)

        # ── Regret curves ──────────────────────────────────────────────
        summary = {}
        for algo_name, run_curves in all_curves.items():
            mean, std = aggregate_regret(run_curves)
            summary[algo_name] = {"mean": mean, "std": std}
            for step_idx, (m, s) in enumerate(zip(mean, std), start=1):
                rows.append({
                    "scenario": scenario_name, "algorithm": algo_name,
                    "step": step_idx, "mean_regret": m, "std_regret": s,
                })

        plot_regret_curves(
            summary=summary,
            output_path=FIGURES_DIR / f"regret_{scenario_name}.png",
            title=f"{PLOT_CONFIG['title']} — {scenario_name}",
            dpi=PLOT_CONFIG["dpi"],
            figsize=PLOT_CONFIG["figsize"],
        )

        # ── Robustness ─────────────────────────────────────────────────
        best_final = min(float(np.mean(v)) for v in final_regret_algo.values())
        for algo_name, vals in final_regret_algo.items():
            robustness_rows.append({
                "scenario": scenario_name,
                "algorithm": algo_name,
                "relative_final_regret": float(np.mean(vals) / max(best_final, 1e-12)),
            })

        # ── Bias diagnostics ───────────────────────────────────────────
        diagnostic_summary: dict[str, dict[str, float]] = {}
        for algo_name, metric_rows in diagnostics_algo.items():
            metric_names = sorted(metric_rows[0].keys()) if metric_rows else []
            diagnostic_summary[algo_name] = {
                mn: float(np.mean([r[mn] for r in metric_rows]))
                for mn in metric_names
            }
            for mn, mv in diagnostic_summary[algo_name].items():
                diagnostic_rows.append({
                    "scenario": scenario_name, "algorithm": algo_name,
                    "metric": mn, "value": mv,
                })

            bias_effect_rows.append({
                "scenario":          scenario_name,
                "algorithm":         algo_name,
                "vote_share_mae":    diagnostic_summary[algo_name]["vote_share_mae"],
                "probability_mae":   diagnostic_summary[algo_name]["probability_mae"],
                "preference_mae":    diagnostic_summary[algo_name]["preference_mae"],
                "rating_mae":        diagnostic_summary[algo_name]["rating_mae"],
                "spearman_rho":      diagnostic_summary[algo_name]["spearman_rho"],
                "mean_rank_error":   diagnostic_summary[algo_name]["mean_rank_error"],
                "top1_accuracy":     diagnostic_summary[algo_name]["top1_correct"],
                "decision_flip_rate":diagnostic_summary[algo_name]["decision_flip"],
            })

        plot_bias_diagnostics(
            table=[
                r for r in diagnostic_rows
                if r["scenario"] == scenario_name
                and r["metric"] in {
                    "vote_share_mae", "probability_mae", "preference_mae",
                    "mean_rank_error", "decision_flip",
                }
            ],
            output_path=FIGURES_DIR / f"bias_diagnostics_{scenario_name}.png",
            title=f"Bias Diagnostics — {scenario_name}",
            dpi=PLOT_CONFIG["dpi"],
        )

        # ── Movie star ratings ─────────────────────────────────────────
        avg_true_pref = np.mean(true_prefs_runs, axis=0)
        true_ratings  = compute_star_ratings(avg_true_pref)

        rating_data: dict[str, np.ndarray] = {"True Quality": true_ratings}
        for algo_name in cfg["algorithms"]:
            avg_pref = np.mean(estimated_prefs_algo[algo_name], axis=0)
            rating_data[algo_name] = compute_star_ratings(avg_pref)

        plot_movie_ratings(
            movie_names=movie_names,
            rating_data=rating_data,
            output_path=FIGURES_DIR / f"movie_ratings_{scenario_name}.png",
            title=f"Estimated Movie Ratings — {scenario_name}",
            dpi=PLOT_CONFIG["dpi"],
        )

        for movie_idx, movie_name in enumerate(movie_names):
            row_data: dict = {
                "scenario": scenario_name,
                "movie":    movie_name,
                "true_rating": float(true_ratings[movie_idx]),
            }
            for algo_name in cfg["algorithms"]:
                row_data[f"rating_{algo_name}"] = float(rating_data[algo_name][movie_idx])
            movie_rating_rows.append(row_data)

        # ── Statistical significance ───────────────────────────────────
        scenario_metrics: dict = {
            "mean_final_regret": {k: float(np.mean(v)) for k, v in final_regret_algo.items()},
            "std_final_regret":  {k: float(np.std(v))  for k, v in final_regret_algo.items()},
            "mean_auc_regret":   {k: float(np.mean(v)) for k, v in auc_algo.items()},
            "bias_diagnostics":  diagnostic_summary,
            "significance_vs_sbcr": {},
        }
        sbcr_arr = np.asarray(final_regret_algo["SBCR"], dtype=float)
        for baseline in ["RRE", "UCB-R"]:
            base_arr = np.asarray(final_regret_algo[baseline], dtype=float)
            p_val, improvement = paired_permutation_test(base_arr, sbcr_arr, random_state=seed)
            scenario_metrics["significance_vs_sbcr"][baseline] = {
                "p_value": float(p_val),
                "mean_improvement": float(improvement),
            }
        scenario_report[scenario_name] = scenario_metrics

        # ── Build per-scenario summary row ────────────────────────────
        summary_table[scenario_name] = {}
        for algo_name in cfg["algorithms"]:
            summary_table[scenario_name][algo_name] = {
                "rating_mae":        diagnostic_summary[algo_name]["rating_mae"],
                "spearman_rho":      diagnostic_summary[algo_name]["spearman_rho"],
                "top1_accuracy":     diagnostic_summary[algo_name]["top1_correct"],
                "mean_rank_error":   diagnostic_summary[algo_name]["mean_rank_error"],
                "final_regret":      float(np.mean(final_regret_algo[algo_name])),
                "decision_flip_rate":diagnostic_summary[algo_name]["decision_flip"],
            }

    # ── Save CSVs ──────────────────────────────────────────────────────────
    _write_csv(RAW_DATA_DIR / "regret_summary.csv",
               ["scenario", "algorithm", "step", "mean_regret", "std_regret"], rows)

    _write_csv(RAW_DATA_DIR / "robustness_table.csv",
               ["scenario", "algorithm", "relative_final_regret"], robustness_rows)

    _write_csv(RAW_DATA_DIR / "bias_diagnostics.csv",
               ["scenario", "algorithm", "metric", "value"], diagnostic_rows)

    bias_effect_fields = [
        "scenario", "algorithm",
        "vote_share_mae", "probability_mae", "preference_mae",
        "rating_mae", "spearman_rho",
        "mean_rank_error", "top1_accuracy", "decision_flip_rate",
    ]
    _write_csv(RAW_DATA_DIR / "bias_effect_summary.csv", bias_effect_fields, bias_effect_rows)

    rating_fields = ["scenario", "movie", "true_rating"] + [
        f"rating_{n}" for n in cfg["algorithms"]
    ]
    movie_ratings_path = RAW_DATA_DIR / "movie_ratings.csv"
    _write_csv(movie_ratings_path, rating_fields, movie_rating_rows)

    # ── Figures ────────────────────────────────────────────────────────────
    robustness_fig = FIGURES_DIR / "robustness_comparison.png"
    plot_robustness_bars(robustness_rows, robustness_fig, dpi=PLOT_CONFIG["dpi"])

    bias_effect_fig = FIGURES_DIR / "bias_effect_summary.png"
    plot_bias_effect_bars(bias_effect_rows, bias_effect_fig, dpi=PLOT_CONFIG["dpi"])

    # ── Statistical report ─────────────────────────────────────────────────
    report_path = RAW_DATA_DIR / "statistical_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(scenario_report, f, indent=2)

    build_markdown_summary(report_path, RAW_DATA_DIR / "statistical_summary.md")
    build_latex_table_from_robustness(
        RAW_DATA_DIR / "robustness_table.csv",
        RAW_DATA_DIR / "robustness_table.tex",
    )

    # ── Console summary ────────────────────────────────────────────────────
    _print_summary(cfg, n_runs, horizon, movie_names, summary_table,
                   scenario_report, movie_ratings_path, report_path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(cfg, n_runs, horizon, movie_names, summary_table,
                   scenario_report, movie_ratings_path, report_path) -> None:
    algos    = cfg["algorithms"]
    scenarios= [s["name"] for s in cfg["scenarios"]]
    sep      = "─" * 90

    print()
    print(sep)
    print("  MOVIE RATING BIAS CORRECTION — EXPERIMENT SUMMARY")
    print(sep)
    print(f"  Pool: {cfg['n_arms']} movies  |  "
          f"{horizon} voting sessions/run  |  {n_runs} Monte Carlo runs")
    print(f"  Sample: {', '.join(movie_names[:4])}, …")
    print()

    # ── Accuracy & robustness table ──
    metric_labels = {
        "rating_mae":         "Rating MAE (↓)",
        "spearman_rho":       "Rank Corr ρ (↑)",
        "top1_accuracy":      "Top-1 Acc   (↑)",
        "mean_rank_error":    "Mean Rank Δ (↓)",
        "final_regret":       "Final Regret(↓)",
        "decision_flip_rate": "Flip Rate   (↓)",
    }
    col_w = 12
    header = f"{'Scenario':<20}{'Metric':<20}" + "".join(f"{a:>{col_w}}" for a in algos)
    print(header)
    print("─" * len(header))

    for scenario in scenarios:
        first_row = True
        for metric_key, metric_label in metric_labels.items():
            scen_label = scenario if first_row else ""
            first_row  = False
            vals = [
                summary_table[scenario][a][metric_key]
                for a in algos
            ]
            row = f"{scen_label:<20}{metric_label:<20}"
            for v in vals:
                row += f"{v:>{col_w}.4f}"
            print(row)
        print()

    # ── Significance ──
    print("  Paired permutation test  (SBCR vs baselines, p-value / improvement)")
    print(f"  {'Scenario':<20}{'Baseline':<10}{'p-value':>10}{'Improvement':>14}")
    for scenario in scenarios:
        sig = scenario_report[scenario].get("significance_vs_sbcr", {})
        for baseline, v in sig.items():
            print(f"  {scenario:<20}{baseline:<10}"
                  f"{v['p_value']:>10.4f}{v['mean_improvement']:>14.4f}")
    print()
    print(f"  Movie ratings CSV : {movie_ratings_path}")
    print(f"  Statistical report: {report_path}")
    print(sep)


if __name__ == "__main__":
    main()
