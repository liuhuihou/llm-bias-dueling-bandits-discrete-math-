from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
RAW_DATA_DIR = RESULTS_DIR / "raw_data"

DEFAULT_CONFIG = {
    "random_seed": 42,
    "n_arms": 12,
    "horizon": 4000,
    "n_runs": 24,
    "preference_temperature": 0.35,
    "base_noise": 0.08,
    "algorithms": ["RUCB", "BS-UCB", "DBS-UCB"],
    "dbs_params": {"alpha": 1.25, "bias_penalty": 0.85},
    "rucb_params": {"alpha": 2.0},
    "bsucb_params": {"beta": 1.5},
    "scenarios": [
        {
            "name": "position_bias",
            "position_bias": 0.06,
            "conformity_bias": 0.00,
            "selective_bias": 0.00,
            "selective_threshold": 0.12,
        },
        {
            "name": "conformity_bias",
            "position_bias": 0.00,
            "conformity_bias": 0.14,
            "selective_bias": 0.00,
            "selective_threshold": 0.12,
        },
        {
            "name": "selective_feedback",
            "position_bias": 0.00,
            "conformity_bias": 0.00,
            "selective_bias": 0.24,
            "selective_threshold": 0.10,
        },
        {
            "name": "mixed_bias",
            "position_bias": 0.04,
            "conformity_bias": 0.10,
            "selective_bias": 0.15,
            "selective_threshold": 0.12,
        },
    ],
}

PLOT_CONFIG = {
    "title": "Bias-Robust Dueling Bandits",
    "dpi": 160,
    "figsize": (9, 5),
}


def ensure_runtime_dirs() -> None:
    """Create runtime output folders if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
