from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseDuelingBanditAlgorithm(ABC):
    def __init__(self, n_arms: int, name: str, random_state: int | None = None) -> None:
        self.n_arms = n_arms
        self.name = name
        self.rng = np.random.default_rng(random_state)
        self.wins = np.zeros((n_arms, n_arms), dtype=float)
        self.comparisons = np.zeros((n_arms, n_arms), dtype=float)
        self.t = 1

    @abstractmethod
    def select_pair(self) -> Tuple[int, int]:
        """Choose a pair of arms for a duel."""

    def update(
        self,
        i: int,
        j: int,
        winner: int,
        vote_share_i: float | None = None,
        audience_size: int | None = None,
    ) -> None:
        loser = j if winner == i else i
        weight = 1.0 if audience_size is None else max(1.0, float(audience_size))

        if vote_share_i is None:
            self.wins[winner, loser] += weight
        else:
            share_i = float(np.clip(vote_share_i, 0.0, 1.0))
            self.wins[i, j] += weight * share_i
            self.wins[j, i] += weight * (1.0 - share_i)

        self.comparisons[i, j] += weight
        self.comparisons[j, i] += weight
        self.t += 1

    def estimated_preferences(self) -> np.ndarray:
        pref = np.full((self.n_arms, self.n_arms), 0.5, dtype=float)
        observed = self.comparisons > 0
        pref[observed] = self.wins[observed] / self.comparisons[observed]
        np.fill_diagonal(pref, 0.5)
        return pref

    def confidence_radius(self, alpha: float = 1.0) -> np.ndarray:
        denom = np.maximum(self.comparisons, 1.0)
        return np.sqrt(alpha * np.log(max(self.t, 2)) / denom)

    def run_step(self, preference_matrix: np.ndarray) -> Tuple[int, int, int]:
        i, j = self.select_pair()
        if i == j:
            j = int((j + 1) % self.n_arms)

        winner = i if self.rng.random() < preference_matrix[i, j] else j
        self.update(i, j, winner)
        return i, j, winner
