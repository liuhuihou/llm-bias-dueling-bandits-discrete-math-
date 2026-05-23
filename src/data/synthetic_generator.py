from __future__ import annotations

import numpy as np

# Pool of fictional movie titles used to label synthetic movies in experiments.
MOVIE_NAMES = [
    "Stellar Drift", "The Neon Labyrinth", "Crimson Horizons", "Whispers of the Deep",
    "Iron Requiem", "The Last Cartographer", "Shattered Moons", "The Ember Coast",
    "Phantom Meridian", "Echoes of Tomorrow", "The Obsidian Gate", "Silver Lining Storm",
    "The Forgotten Archive", "Desert Mirage", "Nocturne Rising", "The Glass Empire",
    "Beneath Still Waters", "The Iron Phoenix", "Starfall Chronicles", "The Hollow King",
    "Tide of Shadows", "Crystal Labyrinth", "The Vanishing Hour", "Aurora's End",
    "The Broken Compass", "Midnight Cascade", "The Amber Threshold", "Sovereign Dust",
    "The Last Migration", "Fractured Light", "The Sable Crown", "Zenith Protocol",
    "The Wandering Flame", "Cold Harbor", "The Gilded Cage", "Temporal Rift",
    "The Silver Accord", "Dusk Before Dawn", "The Infinite Garden", "Storm Covenant",
    "The Ivory Tower", "Rust and Embers", "The Sapphire Current", "Ghost Longitude",
    "The Crimson Pact", "Horizon Protocol", "The Verdant Siege", "Neon Requiem",
    "The Marble Archive", "Shadow Cartography",
]


def generate_movie_names(n: int, rng: np.random.Generator | None = None) -> list[str]:
    """Return n distinct movie names sampled from MOVIE_NAMES without replacement."""
    if n > len(MOVIE_NAMES):
        raise ValueError(f"Requested {n} names but only {len(MOVIE_NAMES)} are available.")
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(len(MOVIE_NAMES), size=n, replace=False)
    return [MOVIE_NAMES[i] for i in indices]


class BiasedVotingEnvironment:
    """Simulates biased pairwise voting between two movies.

    When a platform shows users movie A vs movie B, their votes are not purely
    determined by true quality.  Three structured biases distort the observed
    vote share:

    1. Display-order bias  – the movie shown first receives a systematic vote
       boost regardless of quality (attention / anchoring effect).
    2. Peer-effect bias    – movies that already have a high cumulative vote
       share attract even more votes due to herding behaviour.
    3. Extreme-opinion bias – when two movies are very similar in quality,
       users with polarised opinions dominate, amplifying the observed signal.

    A small simulated audience (audience_size voters) casts ballots each round.
    The winner is determined by majority vote; ties are broken at random.
    """

    def __init__(
        self,
        preference_matrix: np.ndarray,
        position_bias: float = 0.0,
        conformity_bias: float = 0.0,
        selective_bias: float = 0.0,
        selective_threshold: float = 0.12,
        judgement_noise: float = 0.035,
        audience_size: int = 7,
        min_probability: float = 0.02,
        max_probability: float = 0.98,
        random_state: int | None = None,
    ) -> None:
        self.preference_matrix = preference_matrix
        self.n_arms = preference_matrix.shape[0]
        self.position_bias = float(position_bias)
        self.conformity_bias = float(conformity_bias)
        self.selective_bias = float(selective_bias)
        self.selective_threshold = float(selective_threshold)
        self.judgement_noise = float(judgement_noise)
        self.audience_size = max(1, int(audience_size))
        self.min_probability = float(min_probability)
        self.max_probability = float(max_probability)
        self.rng = np.random.default_rng(random_state)

        # Cumulative public vote statistics drive the peer-effect bias.
        self.public_wins = np.zeros(self.n_arms, dtype=float)
        self.public_duels = np.zeros(self.n_arms, dtype=float)
        self.last_event: dict[str, float | int] | None = None

    def _popularity(self) -> np.ndarray:
        """Laplace-smoothed cumulative win rate used as the peer-effect signal."""
        return (self.public_wins + 1.0) / (self.public_duels + 2.0)

    def _clip_probability(self, p: float) -> float:
        return float(np.clip(p, self.min_probability, self.max_probability))

    def _ambiguity_weight(self, true_probability: float) -> float:
        """Weight that peaks when two movies are near-equal in true quality."""
        if self.selective_threshold <= 0.0:
            return 0.0
        margin = abs(true_probability - 0.5)
        return float(np.clip(1.0 - margin / self.selective_threshold, 0.0, 1.0))

    def bias_function(self, i: int, j: int, true_probability: float) -> dict[str, float]:
        """Compute structured vote bias for showing movie i before movie j."""
        popularity = self._popularity()
        popularity_i = float(popularity[i])
        popularity_j = float(popularity[j])
        popularity_diff = popularity_i - popularity_j
        ambiguity = self._ambiguity_weight(true_probability)

        # Display-order bias: constant boost for the first-shown movie.
        position_component = self.position_bias

        # Peer-effect bias: proportional to popularity gap, smooth via tanh.
        conformity_component = self.conformity_bias * float(np.tanh(3.0 * popularity_diff))

        # Extreme-opinion bias: active only when movies are close in quality.
        social_cue = float(np.tanh(3.0 * popularity_diff))
        display_cue = 1.0
        selective_component = 0.5 * self.selective_bias * ambiguity * (
            0.65 * social_cue + 0.35 * display_cue
        )

        bias_delta = position_component + conformity_component + selective_component
        return {
            "position_component": float(position_component),
            "conformity_component": float(conformity_component),
            "selective_component": float(selective_component),
            "bias_delta": float(bias_delta),
            "popularity_i": popularity_i,
            "popularity_j": popularity_j,
            "popularity_diff": float(popularity_diff),
            "ambiguity_weight": ambiguity,
        }

    def _sample_probability_components(self, i: int, j: int) -> dict[str, float | int]:
        true_probability = float(self.preference_matrix[i, j])
        ambiguity = self._ambiguity_weight(true_probability)
        noise_sigma = self.judgement_noise * (1.0 + ambiguity)
        gaussian_noise = float(self.rng.normal(0.0, noise_sigma))
        noisy_probability = self._clip_probability(true_probability + gaussian_noise)

        bias_parts = self.bias_function(i, j, true_probability)
        observed_probability = self._clip_probability(
            noisy_probability + float(bias_parts["bias_delta"])
        )

        event: dict[str, float | int] = {
            "display_first": int(i),
            "display_second": int(j),
            "true_probability": true_probability,
            "gaussian_noise": gaussian_noise,
            "noise_sigma": float(noise_sigma),
            "noisy_probability": noisy_probability,
            "observed_probability": observed_probability,
        }
        event.update(bias_parts)
        return event

    def observed_probability(self, i: int, j: int) -> float:
        return float(self._sample_probability_components(i, j)["observed_probability"])

    def duel(self, i: int, j: int) -> int:
        """Run one pairwise voting session between movie i and movie j."""
        event = self._sample_probability_components(i, j)
        p = float(event["observed_probability"])
        votes_i = int(self.rng.binomial(self.audience_size, p))
        votes_j = self.audience_size - votes_i

        if votes_i == votes_j:
            winner = i if self.rng.random() < 0.5 else j
        else:
            winner = i if votes_i > votes_j else j

        self.public_duels[i] += float(self.audience_size)
        self.public_duels[j] += float(self.audience_size)
        self.public_wins[i] += float(votes_i)
        self.public_wins[j] += float(votes_j)

        event.update(
            {
                "audience_size": int(self.audience_size),
                "votes_i": int(votes_i),
                "votes_j": int(votes_j),
                "vote_share_i": float(votes_i / self.audience_size),
                "winner": int(winner),
                "first_won": int(winner == i),
            }
        )
        self.last_event = event
        return winner


# Backward-compatible alias so existing imports still work.
BiasedDuelingEnvironment = BiasedVotingEnvironment


class MovieRatingSimulator:
    """Generates synthetic movie quality matrices for simulation experiments.

    Each movie is assigned a latent quality score.  Pairwise comparison
    probabilities are derived from quality differences via a logistic function,
    then perturbed by a small movie-level noise term so that the true ranking
    is not purely one-dimensional.
    """

    def __init__(self, random_state: int | None = None) -> None:
        self.rng = np.random.default_rng(random_state)

    def generate_preference_matrix(
        self,
        n_arms: int,
        temperature: float = 0.35,
        base_noise: float = 0.08,
    ) -> np.ndarray:
        """Create a pairwise preference matrix from latent movie quality scores.

        P[i, j] = probability that movie i is preferred over movie j.
        """
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

    # Alias for semantic clarity in movie rating context.
    generate_quality_matrix = generate_preference_matrix

    @staticmethod
    def find_condorcet_winner(preference_matrix: np.ndarray) -> int | None:
        """Return the index of the movie that beats all others, or None."""
        n_arms = preference_matrix.shape[0]
        for arm in range(n_arms):
            wins_all = np.all(preference_matrix[arm, np.arange(n_arms) != arm] > 0.5)
            if wins_all:
                return arm
        return None

    find_best_movie = find_condorcet_winner


# Backward-compatible alias.
SyntheticDuelingGenerator = MovieRatingSimulator
