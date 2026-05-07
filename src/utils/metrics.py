from __future__ import annotations

import numpy as np


def instantaneous_regret(
    best_arm: int,
    arm_i: int,
    arm_j: int,
    preference_matrix: np.ndarray,
) -> float:
    """Strong regret for a duel against the best arm benchmark."""
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
    return float(np.trapezoid(curve))


def paired_permutation_test(
    baseline: np.ndarray,
    contender: np.ndarray,
    n_permutations: int = 4000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Two-sided paired permutation test on final regret differences.

    Returns (p_value, mean_improvement), where improvement is baseline - contender.
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
