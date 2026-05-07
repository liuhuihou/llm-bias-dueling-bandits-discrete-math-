# llm-bias-dueling-bandits

Research-oriented framework for bias-robust dueling bandits under human evaluation bias.

## Quick start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run full benchmark:
   - `python -m src.main`
3. Run quick debug benchmark:
   - `python -m src.main --quick`
4. Optional custom scale:
   - `python -m src.main --horizon 6000 --runs 30`
5. Check outputs:
   - Per-scenario regret plots: `results/figures/regret_*.png`
   - Robustness figure: `results/figures/robustness_comparison.png`
   - Regret table: `results/raw_data/regret_summary.csv`
   - Robustness table: `results/raw_data/robustness_table.csv`
   - Significance report: `results/raw_data/statistical_report.json`
   - Report source: `docs/project_report.tex`
   - Compiled PDF: `docs/project_report.pdf`

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
