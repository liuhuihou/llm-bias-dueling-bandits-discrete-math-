from __future__ import annotations

import numpy as np


# ── Regret metrics ────────────────────────────────────────────────────────────

def instantaneous_regret(
    best_arm: int,
    arm_i: int,
    arm_j: int,
    preference_matrix: np.ndarray,
) -> float:
    """Strong regret: how much quality is lost by not comparing the best movie."""
    regret = preference_matrix[best_arm, arm_i] + preference_matrix[best_arm, arm_j] - 1.0
    return float(max(regret, 0.0))


def cumulative_regret(regret_trace: list[float]) -> np.ndarray:
    return np.cumsum(np.asarray(regret_trace, dtype=float))


def aggregate_regret(run_curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(run_curves)
    return stacked.mean(axis=0), stacked.std(axis=0)


def final_regret(curve: np.ndarray) -> float:
    return float(curve[-1])


def auc_regret(curve: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(curve))
    if hasattr(np, "trapz"):
        return float(np.trapz(curve))
    return float(np.sum((curve[:-1] + curve[1:]) * 0.5))


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def preference_mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error over the upper triangle of the preference matrix."""
    idx = np.triu_indices_from(estimated, k=1)
    return float(np.mean(np.abs(estimated[idx] - true[idx])))


def rating_mae(estimated: np.ndarray, true: np.ndarray, scale: float = 10.0) -> float:
    """Star-rating MAE on [1, scale].

    Converts both matrices to star ratings via the formula
        R_i = 1 + (scale - 1) * mean_j P[i,j]
    and returns mean_i |R_hat_i - R_true_i|.
    """
    r_hat = 1.0 + (scale - 1.0) * estimated.mean(axis=1)
    r_true = 1.0 + (scale - 1.0) * true.mean(axis=1)
    return float(np.mean(np.abs(r_hat - r_true)))


def spearman_rho(estimated: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation between estimated and true movie quality ranks.

    Both 1-D score vectors (averaged win rates) are ranked; the Pearson
    correlation of the rank vectors gives the Spearman ρ ∈ [−1, 1].
    A value near 1 means the estimated ordering matches the true ordering.
    """
    est_scores = estimated.mean(axis=1)
    true_scores = true.mean(axis=1)
    n = len(est_scores)
    if n < 2:
        return 1.0

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks

    r_est = _rank(est_scores)
    r_true = _rank(true_scores)
    r_est_c = r_est - r_est.mean()
    r_true_c = r_true - r_true.mean()
    denom = np.sqrt(np.sum(r_est_c ** 2) * np.sum(r_true_c ** 2))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(r_est_c, r_true_c) / denom)


# ── Statistical test ──────────────────────────────────────────────────────────

def paired_permutation_test(
    baseline: np.ndarray,
    contender: np.ndarray,
    n_permutations: int = 4000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Two-sided paired permutation test on final regret differences.

    Returns (p_value, mean_improvement) where improvement = baseline − contender.
    """
    if baseline.shape != contender.shape:
        raise ValueError("Arrays must have the same shape for paired testing.")

    diff = baseline - contender
    observed = float(np.mean(diff))
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diff.size))
    permuted_means = (signs * diff[None, :]).mean(axis=1)
    p_value = float(np.mean(np.abs(permuted_means) >= abs(observed)))
    return p_value, observed
