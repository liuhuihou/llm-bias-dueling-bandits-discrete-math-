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
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, name="DBS-UCB", random_state=random_state)
        self.alpha = alpha
        self.bias_penalty = bias_penalty

    def select_pair(self) -> tuple[int, int]:
        p_hat = self.estimated_preferences()
        radius = self.confidence_radius(alpha=self.alpha)

        mean_strength = p_hat.mean(axis=1)
        estimated_bias = np.abs(mean_strength - 0.5)
        explore_bonus = radius.mean(axis=1)

        adaptive_penalty = self.bias_penalty / np.sqrt(max(self.t, 1))
        primary_score = mean_strength - adaptive_penalty * estimated_bias + explore_bonus
        first = int(np.argmax(primary_score))

        uncertainty = radius[:, first]
        near_boundary = 0.5 - np.abs(p_hat[:, first] - 0.5)
        challenge_score = uncertainty + near_boundary
        challenge_score[first] = -np.inf
        second = int(np.argmax(challenge_score))

        if second == first:
            pool = [a for a in range(self.n_arms) if a != first]
            second = int(self.rng.choice(pool))
        return first, second
