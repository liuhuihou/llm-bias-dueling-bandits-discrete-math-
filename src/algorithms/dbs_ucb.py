from __future__ import annotations

import numpy as np

from src.algorithms.base_algorithm import BaseDuelingBanditAlgorithm


class DBSUCBAlgorithm(BaseDuelingBanditAlgorithm):
    """Debiasing-aware UCB strategy for dueling bandits."""

    def __init__(
        self,
        n_arms: int,
        alpha: float = 1.2,
        bias_penalty: float = 0.7,
        order_correction: float = 0.9,
        popularity_correction: float = 0.7,
        min_correction_samples: int = 12,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, name="DBS-UCB", random_state=random_state)
        self.alpha = alpha
        self.bias_penalty = bias_penalty
        self.order_correction = float(order_correction)
        self.popularity_correction = float(popularity_correction)
        self.min_correction_samples = int(min_correction_samples)
        self.display_counts = np.zeros((n_arms, n_arms), dtype=float)
        self.display_wins = np.zeros((n_arms, n_arms), dtype=float)
        self.arm_exposure = np.zeros(n_arms, dtype=float)
        self.arm_vote_mass = np.zeros(n_arms, dtype=float)

    def _random_argmax(self, scores: np.ndarray) -> int:
        max_score = float(np.max(scores))
        candidates = np.flatnonzero(np.isclose(scores, max_score))
        return int(self.rng.choice(candidates))

    def update(
        self,
        i: int,
        j: int,
        winner: int,
        vote_share_i: float | None = None,
        audience_size: int | None = None,
    ) -> None:
        weight = 1.0 if audience_size is None else max(1.0, float(audience_size))
        if vote_share_i is None:
            share_i = 1.0 if winner == i else 0.0
        else:
            share_i = float(np.clip(vote_share_i, 0.0, 1.0))

        self.display_counts[i, j] += weight
        self.display_wins[i, j] += weight * share_i
        self.arm_exposure[i] += weight
        self.arm_exposure[j] += weight
        self.arm_vote_mass[i] += weight * share_i
        self.arm_vote_mass[j] += weight * (1.0 - share_i)
        super().update(i, j, winner, vote_share_i=share_i, audience_size=int(weight))

    def _direction_balanced_preferences(self) -> np.ndarray:
        raw = self.estimated_preferences()
        displayed = self.display_counts > 0
        reverse_displayed = self.display_counts.T > 0
        paired = displayed & reverse_displayed

        first_share = np.divide(
            self.display_wins,
            self.display_counts,
            out=np.full_like(self.display_wins, 0.5),
            where=displayed,
        )
        reverse_second_share = 1.0 - np.divide(
            self.display_wins.T,
            self.display_counts.T,
            out=np.full_like(self.display_wins, 0.5),
            where=self.display_counts.T > 0,
        )
        balanced = np.where(paired, 0.5 * (first_share + reverse_second_share), raw)
        balanced = (
            self.order_correction * balanced
            + (1.0 - self.order_correction) * raw
        )
        balanced = 0.5 * (balanced + (1.0 - balanced.T))
        np.fill_diagonal(balanced, 0.5)
        return np.clip(balanced, 0.01, 0.99)

    def _popularity_bias_estimate(self) -> np.ndarray:
        exposure = np.maximum(self.arm_exposure, 1.0)
        arm_share = self.arm_vote_mass / exposure
        centered_share = arm_share - float(np.mean(arm_share))
        popularity_gap = centered_share[:, None] - centered_share[None, :]
        exposure_ratio = exposure / np.maximum(float(np.max(exposure)), 1.0)
        exposure_gap = exposure_ratio[:, None] - exposure_ratio[None, :]
        return np.clip(0.15 * popularity_gap * np.abs(exposure_gap), -0.05, 0.05)

    def debiased_preferences(self) -> np.ndarray:
        p_hat = self._direction_balanced_preferences()
        popularity_bias = self._popularity_bias_estimate()

        corrected = p_hat - self.popularity_correction * popularity_bias
        observed = self.comparisons > 0
        support = self.comparisons / (self.comparisons + self.min_correction_samples)
        corrected = np.where(observed, support * corrected + (1.0 - support) * 0.5, 0.5)
        corrected = 0.5 * (corrected + (1.0 - corrected.T))
        np.fill_diagonal(corrected, 0.5)
        return np.clip(corrected, 0.01, 0.99)

    def select_pair(self) -> tuple[int, int]:
        p_hat = self.debiased_preferences()
        radius = np.minimum(self.confidence_radius(alpha=self.alpha), 0.45)
        raw_hat = self.estimated_preferences()
        pair_bias = np.abs(raw_hat - p_hat)

        mean_strength = p_hat.mean(axis=1)
        raw_strength = raw_hat.mean(axis=1)
        estimated_bias = np.abs(raw_strength - mean_strength)
        adaptive_penalty = self.bias_penalty / np.sqrt(max(self.t, 1))

        debiased_ucb = p_hat + radius - adaptive_penalty * pair_bias
        candidates = np.all(debiased_ucb >= 0.5, axis=1)
        np.fill_diagonal(debiased_ucb, 1.0)

        arm_usage = self.comparisons.sum(axis=1)
        coverage_bonus = 1.0 / np.sqrt(np.maximum(arm_usage, 1.0))
        lower_margin = np.min(p_hat - radius, axis=1)
        lower_margin = np.where(np.isfinite(lower_margin), lower_margin, -0.5)

        primary_score = (
            mean_strength
            + 0.18 * lower_margin
            + 0.10 * coverage_bonus
            - adaptive_penalty * estimated_bias
        )
        if np.any(candidates):
            masked_score = np.where(candidates, primary_score, -np.inf)
            first = self._random_argmax(masked_score)
        else:
            first = self._random_argmax(primary_score)

        uncertainty = radius[:, first]
        near_boundary = 0.5 - np.abs(p_hat[:, first] - 0.5)
        challenge_score = (
            p_hat[:, first]
            + 0.35 * uncertainty
            + 0.15 * near_boundary
            - adaptive_penalty * pair_bias[:, first]
        )
        challenge_score[first] = -np.inf
        second = self._random_argmax(challenge_score)

        if second == first:
            pool = [a for a in range(self.n_arms) if a != first]
            second = int(self.rng.choice(pool))
        return first, second
