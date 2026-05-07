from __future__ import annotations

import numpy as np

from src.algorithms.base_algorithm import BaseDuelingBanditAlgorithm


class RUCBAlgorithm(BaseDuelingBanditAlgorithm):
    def __init__(self, n_arms: int, alpha: float = 2.0, random_state: int | None = None) -> None:
        super().__init__(n_arms=n_arms, name="RUCB", random_state=random_state)
        self.alpha = alpha

    def select_pair(self) -> tuple[int, int]:
        p_hat = self.estimated_preferences()
        ucb = np.clip(p_hat + self.confidence_radius(alpha=self.alpha), 0.0, 1.0)

        candidates = [i for i in range(self.n_arms) if np.all(ucb[i, :] >= 0.5)]
        if not candidates:
            candidates = list(range(self.n_arms))

        champion = int(self.rng.choice(candidates))
        opponent_scores = ucb[:, champion].copy()
        opponent_scores[champion] = -np.inf
        opponent = int(np.argmax(opponent_scores))

        if opponent == champion:
            pool = [a for a in range(self.n_arms) if a != champion]
            opponent = int(self.rng.choice(pool))
        return champion, opponent


class BSUCBAlgorithm(BaseDuelingBanditAlgorithm):
    def __init__(self, n_arms: int, beta: float = 1.5, random_state: int | None = None) -> None:
        super().__init__(n_arms=n_arms, name="BS-UCB", random_state=random_state)
        self.beta = beta

    def select_pair(self) -> tuple[int, int]:
        arm_usage = self.comparisons.sum(axis=1)
        perturb = self.rng.random(self.n_arms) * 1e-6
        first = int(np.argmin(arm_usage + perturb))

        p_hat = self.estimated_preferences()
        radius = self.confidence_radius(alpha=1.0)
        scores = p_hat[:, first] + self.beta * radius[:, first]
        scores[first] = -np.inf
        second = int(np.argmax(scores))

        if second == first:
            pool = [a for a in range(self.n_arms) if a != first]
            second = int(self.rng.choice(pool))
        return first, second
