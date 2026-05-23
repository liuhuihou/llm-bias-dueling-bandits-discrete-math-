# movie-rating-bias-correction

A pairwise-comparison framework for correcting human bias in movie rating systems,
evaluated through adaptive dueling-bandit algorithms.

## Background

Online movie platforms collect ratings from millions of users, but those ratings are
systematically biased by three well-documented mechanisms:

1. **Attention bias** — the movie displayed first receives more votes due to anchoring.
2. **Echo-chamber bias** — movies that already score high attract inflated future votes
   (herding / peer effect).
3. **Polarisation bias** — when two movies are similar in quality, users with extreme
   opinions dominate the vote, amplifying the observed signal.

This project models these biases mathematically, simulates pairwise voting sessions,
and evaluates three algorithms with increasing debiasing sophistication.  The output
is a corrected 1–10 star rating for every movie in the pool.

## Algorithms

### RRE — Round-Robin Empirical (naive baseline)

Cycles through all movie pairs in a fixed schedule.  Records raw vote shares without
any correction.  Represents the implicit behaviour of most platforms.

### UCB-R — UCB-based Rating (smart exploration baseline)

Selects the pair that most reduces total rating variance:

```
priority(i,j) = P_hat[i,j]*(1−P_hat[i,j])/N_ij  +  α·log(t)/N_ij
```

Uses raw vote shares without bias correction.  Isolates the value of smart exploration.

### SBCR — Symmetric Bias-Corrected Rating (proposed method)

Designed from scratch for the movie rating objective.  Three correction steps:

**Step 1 — Position correction.**  Tracks each pair in both display orders.  The
symmetrised estimate cancels the constant attention-bias offset exactly:

```
P_sym[i,j] = (vote_share_i_when_first + vote_share_i_when_second) / 2
           = P_ij + peer_effect          (position bias cancelled)
```

**Step 2 — Peer-effect correction (excess formulation).**  Computes each movie's
cumulative vote share minus what the current preference estimate predicts.  Any
excess is attributed to herding and subtracted:

```
residual_i   = arm_share_i − mean_j P_sym[i,j]   ≈ peer_effect_i
B_peer[i,j]  = γ · (residual_i − residual_j)
P_corr[i,j]  = P_sym[i,j] − support · B_peer[i,j]
```

When no peer effect exists (b_conf = 0), arm_share ≈ P_sym.mean(axis=1), so
residual ≈ 0 and B_peer ≈ 0.  No spurious correction is applied.

**Step 3 — Correction shrinkage.**  The correction is scaled by
`support = N/(N+min_count)` so sparse pairs receive only partial correction,
while the quality spread of P_sym is preserved.

Pair selection uses variance-UCB on P_corr, with a symmetrisation bonus that
adds extra priority to pairs seen in only one display direction.

## Evaluation

Four bias scenarios isolate and combine the three mechanisms:

| Scenario | Bias active |
|---|---|
| `attention_bias` | Position / attention only |
| `echo_chamber` | Peer-effect / herding only |
| `polarisation` | Extreme-opinion only |
| `realistic` | Mild combination of all three |

Three comparison dimensions:

| Dimension | Metrics |
|---|---|
| **Accuracy** | Rating MAE (1–10 scale), Rank Spearman ρ, Top-1 accuracy, Mean rank error |
| **Efficiency** | Cumulative regret, AUC regret, Final regret |
| **Robustness** | Performance across all four scenarios, relative final regret |

## Quick start

```bash
pip install -r requirements.txt
python -m src.main --quick          # 1 000 rounds, 8 runs
python -m src.main                  # 5 000 rounds, 24 runs
python -m src.main --horizon 8000 --runs 30
```

## Outputs

| Path | Content |
|---|---|
| `results/figures/regret_*.png` | Cumulative regret per scenario |
| `results/figures/movie_ratings_*.png` | Estimated star ratings vs true quality |
| `results/figures/robustness_comparison.png` | Relative final regret across scenarios |
| `results/figures/bias_diagnostics_*.png` | Per-scenario bias diagnostic metrics |
| `results/raw_data/movie_ratings.csv` | Corrected 1–10 ratings for every movie |
| `results/raw_data/bias_effect_summary.csv` | Rating MAE, Spearman ρ, flip rate per algo |
| `results/raw_data/statistical_report.json` | Significance tests (SBCR vs baselines) |

## Reproducibility

Fixed seed in `config/config.py`.  Deterministic seeds per scenario, run, and
algorithm.  Statistical significance via paired permutation test (4 000 permutations).
