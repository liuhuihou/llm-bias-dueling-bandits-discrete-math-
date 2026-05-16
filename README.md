# llm-bias-dueling-bandits

Research-oriented framework for bias-robust dueling bandits under human evaluation bias.

## Quick start

Run from the project root:

1. Install dependencies once:
   - `pip install -r requirements.txt`
2. Run a quick debug benchmark:
   - `python -m src.main --quick`
3. Run the full benchmark:
   - `python -m src.main`
4. Optionally choose a custom scale:
   - `python -m src.main --horizon 6000 --runs 30`

For a fuller Chinese project guide, see `docs/README_zh.md`.

## Testing

Run these checks from the project root:

1. Run the quick benchmark smoke test:
   - `python -m src.main --quick`
2. Verify CLI argument validation:
   - `python -m src.main --horizon 0`
   - Expected result: the command exits with an error saying `--horizon must be a positive integer.`

## Outputs

Benchmark runs write results under `results/`:

1. Scenario regret plots: `results/figures/regret_*.png`
2. Robustness figure: `results/figures/robustness_comparison.png`
3. Regret table: `results/raw_data/regret_summary.csv`
4. Robustness table: `results/raw_data/robustness_table.csv`
5. Significance report: `results/raw_data/statistical_report.json`
6. Report source: `docs/project_report.tex`
7. Compiled report: `docs/project_report.pdf`

## Bias scenarios

The benchmark includes four synthetic human-bias mechanisms:

1. Position bias: users favor the first shown option.
2. Conformity bias: users are influenced by public popularity.
3. Selective feedback bias: users overreact in near-tie comparisons.
4. Mixed bias: combined perturbation of all above.

## Algorithms

1. RUCB (baseline)
2. BS-UCB (baseline)
3. DBS-UCB (debiasing-aware method)

## Reproducibility checklist

1. Fixed random seed in `config/config.py`
2. Explicit scenario parameters in `config/config.py`
3. Statistical significance by paired permutation test
