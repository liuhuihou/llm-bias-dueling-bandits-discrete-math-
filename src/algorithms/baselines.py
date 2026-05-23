from __future__ import annotations

import numpy as np

from src.algorithms.base_algorithm import BaseDuelingBanditAlgorithm


class RREAlgorithm(BaseDuelingBanditAlgorithm):
    """Round-Robin Empirical (RRE).

    Cycles through every movie pair in a fixed round-robin schedule so that
    each pair receives the same number of comparisons.  Vote shares are
    averaged directly with no bias correction.

    This baseline represents the implicit behaviour of most platforms: collect
    votes uniformly and report the raw average.  It serves as the lower-bound
    reference showing the full impact of uncorrected structural bias.

    Pair selection  : deterministic round-robin over all K*(K-1)/2 pairs.
    Preference estimate: P_hat[i,j] = sum(vote_shares_ij) / N_ij  (raw mean).
    Star rating     : 1 + 9 * mean_j P_hat[i,j].
    """

    def __init__(self, n_arms: int, random_state: int | None = None) -> None:
        super().__init__(n_arms=n_arms, name="RRE", random_state=random_state)
        # Pre-generate all unique pairs once; cycle through them.
        self._pairs: list[tuple[int, int]] = [
            (i, j) for i in range(n_arms) for j in range(i + 1, n_arms)
        ]
        self._step: int = 0

    def select_pair(self) -> tuple[int, int]:
        pair = self._pairs[self._step % len(self._pairs)]
        self._step += 1
        return pair


class UCBRAlgorithm(BaseDuelingBanditAlgorithm):
    """UCB-based Rating (UCB-R).

    Adaptively allocates comparisons to the movie pair that is expected to
    reduce total rating uncertainty most.  For a movie i, the star rating is
    determined by its average pairwise win rate; thus the rating variance is:

        Var(R_i) = (9 / (K-1))^2 * sum_{j≠i} Var(P_hat[i,j])

    Minimising the sum of Var(R_i) over all movies is equivalent to minimising
    sum_{i<j} Var(P_hat[i,j]).  Using the Bernoulli variance upper bound
    P(1-P) <= 1/4 and a UCB exploration bonus, the per-pair priority becomes:

        priority(i,j) = P_hat[i,j] * (1 - P_hat[i,j]) / N_ij
                        + alpha * log(t) / N_ij

    Higher priority → the pair's comparison would most reduce rating error.
    Vote shares are used without bias correction to isolate the effect of
    smart exploration alone.

    Pair selection  : argmax priority(i,j).
    Preference estimate: P_hat[i,j] = sum(vote_shares_ij) / N_ij  (raw mean).
    Star rating     : 1 + 9 * mean_j P_hat[i,j].
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, name="UCB-R", random_state=random_state)
        self.alpha = float(alpha)

    def select_pair(self) -> tuple[int, int]:
        P_hat = self.estimated_preferences()
        N = np.maximum(self.comparisons, 1.0)
        log_t = np.log(max(self.t, 2))

        var_term = P_hat * (1.0 - P_hat) / N
        ucb_term = self.alpha * log_t / N
        priority = var_term + ucb_term

        # Small random tie-breaking to avoid always favouring low indices.
        priority += self.rng.random(priority.shape) * 1e-10
        np.fill_diagonal(priority, -np.inf)

        flat_idx = int(np.argmax(priority))
        i, j = flat_idx // self.n_arms, flat_idx % self.n_arms
        return i, j
