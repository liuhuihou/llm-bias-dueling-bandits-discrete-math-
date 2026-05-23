from __future__ import annotations

import numpy as np

from src.algorithms.base_algorithm import BaseDuelingBanditAlgorithm


class SBCRAlgorithm(BaseDuelingBanditAlgorithm):
    """Symmetric Bias-Corrected Rating (SBCR).

    Designed from scratch for the movie pairwise-rating problem where three
    structural biases contaminate observed vote shares.

    == Bias model ==

    When movie i is shown first vs movie j, the observed vote share is:

        V_ij(t) = P_ij + d_pos + d_peer(i,j,t) + d_sel(i,j,t) + eps(t)

    where
        d_pos           = b_pos    (constant; tied to display position 1)
        d_peer(i,j,t)   = b_conf * tanh(3*(pop_i(t) - pop_j(t)))
        d_sel(i,j,t)    = selective amplification near P_ij = 0.5
        eps(t)          = Gaussian noise

    == Step 1: Position correction ==

    Track display direction separately:
        fwd_wins[i,j]   -> weighted vote share for i when i shown FIRST vs j
        fwd_counts[i,j] -> total weight of (i-first) comparisons

    Symmetrised estimate (both directions available):
        P_sym[i,j] = 0.5 * (P_fwd[i,j] + (1 - P_rev[i,j]))
                   = 0.5 * ((P_ij + d_pos + d_peer) + (P_ij - d_pos + d_peer))
                   = P_ij + d_peer(i,j,t)   <- position bias CANCELS exactly

    When only one direction is available P_sym falls back to that single
    observation; the symmetrisation bonus in pair selection encourages
    collecting the missing direction quickly.

    == Step 2: Peer-effect correction (excess formulation) ==

    Key insight: without peer-effect bias the cumulative arm vote share should
    equal the quality score predicted by the current preference estimate:

        arm_share_i  ~  mean_j P_sym[i,j]   (no peer effect)

    Any EXCESS beyond this prediction is attributed to herding:

        residual_i   = arm_share_i - mean_j P_sym[i,j]
                     ~ b_conf * mean_j tanh(3*(pop_i - pop_j))

    Pairwise peer bias estimate:
        B_peer[i,j]  = peer_gamma * (residual_i - residual_j)

    When b_conf = 0: arm_share ~ P_sym.mean(axis=1), so residual ~ 0 and
    B_peer ~ 0 -- no spurious correction even when qualities differ.

    Corrected:
        P_corr[i,j]  = P_sym[i,j] - B_peer[i,j]

    == Step 3: Correction shrinkage ==

    For sparse pairs the correction B_peer itself is noisy.  Shrink only
    the CORRECTION toward zero -- not the base estimate toward 0.5 -- so
    that the quality spread is not artificially compressed:

        support[i,j] = N_ij / (N_ij + min_count)
        P_corr[i,j]  = P_sym[i,j] - support * B_peer[i,j]

    When support -> 0 this gives P_corr = P_sym (no correction, no compression).
    When support -> 1 it gives P_corr = P_sym - B_peer (full correction).

    Anti-symmetry is enforced after every correction step.

    == Step 4: Pair selection ==

    Three additive priority terms:

    (a) Variance-UCB (uses P_corr):
            var_ucb[i,j] = P_corr*(1-P_corr)/N + alpha*log(t)/N

    (b) Symmetrisation bonus for pairs seen in only one display direction:
            asym[i,j] = sym_bonus / N   if exactly one direction observed

    (c) Peer-confidence term -- large residual -> estimate still uncertain:
            peer_conf[i,j] = 0.5 * |B_peer[i,j]| / (N + 1)

    == Output ==

    debiased_preferences() returns P_corr.
    Star rating: R_i = 1 + 9 * mean_j P_corr[i,j].
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 1.0,
        peer_gamma: float = 0.70,
        sym_bonus: float = 0.06,
        peer_clip: float = 0.20,
        min_count: int = 10,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_arms=n_arms, name="SBCR", random_state=random_state)
        self.alpha      = float(alpha)
        self.peer_gamma = float(peer_gamma)
        self.sym_bonus  = float(sym_bonus)
        self.peer_clip  = float(peer_clip)
        self.min_count  = int(min_count)

        # Direction-specific statistics: fwd = movie i displayed first vs j.
        self.fwd_counts = np.zeros((n_arms, n_arms), dtype=float)
        self.fwd_wins   = np.zeros((n_arms, n_arms), dtype=float)

        # Arm-level aggregates for peer-effect estimation.
        self.arm_exposure  = np.zeros(n_arms, dtype=float)
        self.arm_vote_mass = np.zeros(n_arms, dtype=float)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        i: int,
        j: int,
        winner: int,
        vote_share_i: float | None = None,
        audience_size: int | None = None,
    ) -> None:
        weight  = 1.0 if audience_size is None else max(1.0, float(audience_size))
        share_i = (
            (1.0 if winner == i else 0.0)
            if vote_share_i is None
            else float(np.clip(vote_share_i, 0.0, 1.0))
        )

        # Movie i was shown first in this round.
        self.fwd_counts[i, j] += weight
        self.fwd_wins[i, j]   += weight * share_i

        self.arm_exposure[i]  += weight
        self.arm_exposure[j]  += weight
        self.arm_vote_mass[i] += weight * share_i
        self.arm_vote_mass[j] += weight * (1.0 - share_i)

        super().update(i, j, winner, vote_share_i=share_i, audience_size=int(weight))

    # ------------------------------------------------------------------
    # Step 1: Position correction
    # ------------------------------------------------------------------

    def _position_corrected(self) -> np.ndarray:
        """Symmetrise over display order to cancel constant position bias."""
        has_fwd = self.fwd_counts > 0     # i shown first vs j
        has_rev = self.fwd_counts.T > 0   # j shown first vs i

        fwd_rate = np.where(
            has_fwd,
            self.fwd_wins / np.where(has_fwd, self.fwd_counts, 1.0),
            np.nan,
        )
        # rev_rate[i,j] = vote share for j when (j,i) shown
        rev_rate = np.where(
            has_rev,
            self.fwd_wins.T / np.where(has_rev, self.fwd_counts.T, 1.0),
            np.nan,
        )

        both     = has_fwd & has_rev
        only_fwd = has_fwd & ~has_rev
        only_rev = ~has_fwd & has_rev

        P_sym = np.full((self.n_arms, self.n_arms), 0.5)
        # Both directions: position bias cancels in the average.
        P_sym[both]     = 0.5 * (fwd_rate[both] + (1.0 - rev_rate[both]))
        P_sym[only_fwd] = fwd_rate[only_fwd]
        P_sym[only_rev] = 1.0 - rev_rate[only_rev]

        P_sym = 0.5 * (P_sym + (1.0 - P_sym.T))
        np.fill_diagonal(P_sym, 0.5)
        return np.clip(P_sym, 0.01, 0.99)

    # ------------------------------------------------------------------
    # Step 2: Peer-effect correction (excess formulation)
    # ------------------------------------------------------------------

    def _peer_bias_estimate(self, P_ref: np.ndarray) -> np.ndarray:
        """Excess popularity gap unexplained by the current preference estimate.

        arm_share_i  - mean_j P_ref[i,j]  ~  0          when b_conf = 0
                                           ~  peer_i     when b_conf > 0
        """
        exposure  = np.maximum(self.arm_exposure, 1.0)
        arm_share = self.arm_vote_mass / exposure   # observed cumulative win rate
        expected  = P_ref.mean(axis=1)              # quality predicted by P_ref
        residual  = arm_share - expected            # unexplained excess ~ peer inflation
        peer_gap  = residual[:, None] - residual[None, :]
        return np.clip(self.peer_gamma * peer_gap, -self.peer_clip, self.peer_clip)

    # ------------------------------------------------------------------
    # Full debiased estimate
    # ------------------------------------------------------------------

    def debiased_preferences(self) -> np.ndarray:
        """Return the position- and peer-corrected preference matrix P_corr."""
        P_sym  = self._position_corrected()
        B_peer = self._peer_bias_estimate(P_sym)
        N      = self.comparisons

        # Shrink only the correction term, preserving the quality spread in P_sym.
        support = N / (N + self.min_count)
        P_corr  = P_sym - support * B_peer

        P_corr = 0.5 * (P_corr + (1.0 - P_corr.T))
        np.fill_diagonal(P_corr, 0.5)
        return np.clip(P_corr, 0.01, 0.99)

    # ------------------------------------------------------------------
    # Step 4: Pair selection
    # ------------------------------------------------------------------

    def select_pair(self) -> tuple[int, int]:
        P_corr = self.debiased_preferences()
        N      = np.maximum(self.comparisons, 1.0)
        log_t  = np.log(max(self.t, 2))

        # (a) Variance-UCB on corrected preferences.
        var_ucb = P_corr * (1.0 - P_corr) / N + self.alpha * log_t / N

        # (b) Symmetrisation bonus: one direction only -> position bias lingers.
        one_direction = (self.fwd_counts > 0) ^ (self.fwd_counts.T > 0)
        asym_bonus    = self.sym_bonus * np.where(one_direction, 1.0 / N, 0.0)

        # (c) Peer-confidence: large residual -> estimate still uncertain.
        B_peer    = self._peer_bias_estimate(P_corr)
        peer_conf = 0.5 * np.abs(B_peer) / (N + 1.0)

        priority = var_ucb + asym_bonus + peer_conf
        priority += self.rng.random(priority.shape) * 1e-10
        np.fill_diagonal(priority, -np.inf)

        flat_idx = int(np.argmax(priority))
        i, j = flat_idx // self.n_arms, flat_idx % self.n_arms
        return i, j
