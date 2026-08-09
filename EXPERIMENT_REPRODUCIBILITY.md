# Experiment reproducibility record

## Authoritative protocol

Every predictive result in the qualification manuscript uses five-fold stratified cross-validation. For each outer fold:

1. preprocessing parameters are learned from the four training folds only;
2. CTGAN is fitted only to the fold's real training records;
3. synthetic candidates and SepAware selections are generated inside that fold;
4. CatBoost and XGBoost are fitted to the designated real, synthetic, or augmented training data;
5. performance is evaluated once on the untouched real validation fold.

The older dataset-specific fixed-lambda notebooks use one 75/25 split and are superseded. They are not evidence for the manuscript.

## Executed notebooks

- `experiments/SepAware_Fixed_Lambda_5Fold_CV_Study.executed.ipynb`
- `experiments/sepaware_replicated_2k_factorial_anova.executed.ipynb`
- `experiments/Covid_Causal_Filter_Local.executed.ipynb`

Compact authoritative CSV exports are included under `results/`.

## Local software stack

- Python 3.12
- NumPy 1.26.4
- pandas 2.2.2
- scikit-learn 1.5.1
- SDV 1.16.2
- CatBoost 1.2.5
- XGBoost 2.1.4
- statsmodels 0.14.2

The local runner forces CTGAN onto CPU. XGBoost receives the same imputed numerical design matrix as a NumPy array because its macOS pandas adapter crashed on the mixed clinical frame. This changes representation only, not observations, folds, preprocessing, features, labels, or model parameters.

TabPFN was not available in the reproducible local environment. No stale TabPFN value is retained as a rerun result.
