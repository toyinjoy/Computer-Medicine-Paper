# Computers in Biology and Medicine readiness note

## Proposed contribution

A transparent post-generation selection framework showing that class-boundary preservation is a distinct and controllable property of synthetic tabular health data. The paper should lead with the predeclared replicated factorial experiment, cross-dataset heterogeneity, and the clinically important finding that separability can improve synthetic-to-real utility while anchor effects and fidelity trade-offs are dataset dependent.

## Possible title

**Beyond Fidelity: Factorial Evaluation of Separability-Aware Synthetic Tabular Data for Imbalanced Health Prediction**

## Completed support

- Two health datasets: Vigitel obesity (5,000 rows) and COVID-19 mortality (334 rows).
- Five-fold train-on-synthetic/test-on-real factorial experiment with four predeclared selection regimes.
- Fold-level Macro-F1 for 40 observations.
- Raw/log blocked ANOVA, variation allocation, confidence intervals, and residual diagnostics.
- Fidelity, correlation, SI, HM, and lambda-grid ablations.
- A completed COVID causal-anchor augmentation experiment as supporting evidence.

## Experiments required before submission

1. Repeat the factorial design over independent generator and CV seeds; use nested selection.
2. Add TVAE, Gaussian copula, TabDDPM, SMOTE, and random-oversampling baselines.
3. Report minority precision, recall, F1, AUPRC, AUROC, calibration, and decision-curve analysis.
4. Add class-conditional fidelity, minority diversity/coverage, and real-vs-synthetic distinguishability.
5. Run membership- and attribute-inference attacks and rarity-stratified nearest-record analysis.
6. Validate temporally or externally on an independent hospital/survey cohort.
7. Include robust hierarchical inference across fold, seed, classifier, and dataset.

## Likely reviewer concerns

- Five overlapping CV folds are not independent replications.
- CTGAN is the only generator in the confirmatory experiment.
- The COVID sample is small and effects are non-significant.
- Composite fidelity and anchor scores need stronger validation.
- Strong Vigitel separation may reflect shortcut amplification or minority mode collapse.
- No current formal privacy guarantee or attack-based privacy evaluation.
- “Causal” anchors are associational unless supported by a validated graph and identification assumptions.

## Recommended analyses

Prioritise multi-seed nested CV, AUPRC/minority recall, class-conditional coverage, privacy attacks, and external validation. Frame the current COVID null result as evidence of dataset-dependent applicability, not as a weakness to hide.
