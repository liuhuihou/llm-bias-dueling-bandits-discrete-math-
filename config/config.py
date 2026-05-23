from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
RAW_DATA_DIR = RESULTS_DIR / "raw_data"

DEFAULT_CONFIG = {
    "random_seed": 42,
    "n_arms": 12,           # movies in the candidate pool
    "horizon": 5000,        # pairwise voting sessions per Monte Carlo run
    "n_runs": 24,
    "preference_temperature": 0.35,
    "base_noise": 0.08,
    "algorithms": ["RRE", "UCB-R", "SBCR"],
    # Algorithm hyper-parameters.
    "rre_params": {},
    "ucbr_params": {"alpha": 1.0},
    "sbcr_params": {
        "alpha": 1.0,
        "peer_gamma": 0.85,
        "sym_bonus": 0.06,
        "peer_clip": 0.12,
        "min_count": 25,
    },
    # ── Bias scenarios ────────────────────────────────────────────────────
    # Four scenarios isolate and combine the three bias mechanisms that
    # arise in real movie rating platforms.
    #
    # attention_bias   : Anchoring effect – first-listed movie gets more votes
    #                    regardless of quality (position_bias only).
    # echo_chamber     : Herding behaviour – movies already popular keep getting
    #                    higher votes (conformity_bias only, strong amplitude).
    # polarisation     : Near-tie amplification – when two movies are similar
    #                    in quality, polarised users dominate (selective_bias only).
    # realistic        : Mild combination of all three, representing a real
    #                    platform with moderate but simultaneous biases.
    "scenarios": [
        {
            "name": "attention_bias",
            "position_bias": 0.10,
            "conformity_bias": 0.00,
            "selective_bias": 0.00,
            "selective_threshold": 0.12,
        },
        {
            "name": "echo_chamber",
            "position_bias": 0.00,
            "conformity_bias": 0.20,
            "selective_bias": 0.00,
            "selective_threshold": 0.12,
        },
        {
            "name": "realistic",
            "position_bias": 0.06,
            "conformity_bias": 0.14,
            "selective_bias": 0.15,
            "selective_threshold": 0.12,
        },
    ],
}

PLOT_CONFIG = {
    "title": "Movie Rating Bias Correction",
    "dpi": 160,
    "figsize": (9, 5),
}


def ensure_runtime_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
