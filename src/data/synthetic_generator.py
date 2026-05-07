from __future__ import annotations

import numpy as np


class BiasedDuelingEnvironment:
    """Observed duel outcomes with configurable human bias mechanisms."""

    def __init__(
        self,
        preference_matrix: np.ndarray,
        position_bias: float = 0.0,
        conformity_bias: float = 0.0,
        selective_bias: float = 0.0,
        selective_threshold: float = 0.12,
        random_state: int | None = None,
    ) -> None:
        self.preference_matrix = preference_matrix
        self.n_arms = preference_matrix.shape[0]
        self.position_bias = float(position_bias)
        self.conformity_bias = float(conformity_bias)
        self.selective_bias = float(selective_bias)
        self.selective_threshold = float(selective_threshold)
        self.rng = np.random.default_rng(random_state)

        self.public_wins = np.zeros(self.n_arms, dtype=float)
        self.public_duels = np.zeros(self.n_arms, dtype=float)

    def _popularity(self) -> np.ndarray:
        return (self.public_wins + 1.0) / (self.public_duels + 2.0)

    def observed_probability(self, i: int, j: int) -> float:
        p = float(self.preference_matrix[i, j])

        # Position bias: the first shown item is more likely to be selected.
        p += self.position_bias

        # Conformity bias: users follow the currently more popular arm.
        popularity = self._popularity()
        p += self.conformity_bias * (popularity[i] - popularity[j])

        # Selective feedback: users overreact when two arms are close.
        margin = abs(self.preference_matrix[i, j] - 0.5)
        if margin <= self.selective_threshold and self.rng.random() < self.selective_bias:
            social_push = np.sign(popularity[i] - popularity[j])
            p += 0.12 * social_push

        return float(np.clip(p, 0.02, 0.98))

    def duel(self, i: int, j: int) -> int:
        p = self.observed_probability(i, j)
        winner = i if self.rng.random() < p else j

        self.public_duels[i] += 1.0
        self.public_duels[j] += 1.0
        self.public_wins[winner] += 1.0
        return winner


class SyntheticDuelingGenerator:
    def __init__(self, random_state: int | None = None) -> None:
        self.rng = np.random.default_rng(random_state)

    def generate_preference_matrix(
        self,
        n_arms: int,
        temperature: float = 0.35,
        base_noise: float = 0.08,
    ) -> np.ndarray:
        """Create latent pairwise preference matrix before observed bias effects."""
        utility = self.rng.normal(0.0, 1.0, size=n_arms)
        diff = utility[:, None] - utility[None, :]
        logits = diff / max(temperature, 1e-3)
        matrix = 1.0 / (1.0 + np.exp(-logits))

        bias = self.rng.normal(0.0, base_noise, size=n_arms)
        for i in range(n_arms):
            for j in range(i + 1, n_arms):
                shifted = float(np.clip(matrix[i, j] + bias[i] - bias[j], 0.05, 0.95))
                matrix[i, j] = shifted
                matrix[j, i] = 1.0 - shifted

        np.fill_diagonal(matrix, 0.5)
        return matrix

    @staticmethod
    def find_condorcet_winner(preference_matrix: np.ndarray) -> int | None:
        n_arms = preference_matrix.shape[0]
        for arm in range(n_arms):
            wins_all = np.all(preference_matrix[arm, np.arange(n_arms) != arm] > 0.5)
            if wins_all:
                return arm
        return None
