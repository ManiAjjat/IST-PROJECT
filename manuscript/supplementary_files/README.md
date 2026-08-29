# Supplementary files

This folder contains the essential, nonredundant supporting tables and publication-ready supplementary figures. Original files in `results/` and the existing manuscript folders were preserved.

## Supplementary tables

- **Table S1. Physicochemical descriptor statistics.** `tables/Table_S1_Physicochemical_Descriptor_Statistics.csv`
- **Table S2. Highly correlated physicochemical descriptor pairs.** `tables/Table_S2_Highly_Correlated_Descriptor_Pairs.csv`
- **Table S3. Cross-validated performance of traditional classifiers.** `tables/Table_S3_Traditional_Model_Performance.csv`
- **Table S4. Permutation importance of traditional XGBoost descriptors.** `tables/Table_S4_Traditional_XGBoost_Permutation_Importance.csv`
- **Table S5. Cross-validated performance of ESM-2 classifiers.** `tables/Table_S5_ESM2_Model_Performance.csv`
- **Table S6. Paired-bootstrap comparison of traditional and ESM-2 representations.** `tables/Table_S6_Traditional_vs_ESM2_Paired_Bootstrap.csv`
- **Table S7. Consensus hard-case peptides and prediction evidence.** `tables/Table_S7_Consensus_Hard_Case_Peptides.csv`
- **Table S8. Sequence families and redundancy summary.** `tables/Table_S8_Sequence_Family_Summary.csv`
- **Table S9. Model performance stratified by sequence novelty.** `tables/Table_S9_Sequence_Novelty_Performance.csv`
- **Table S10. Probability calibration metrics.** `tables/Table_S10_Probability_Calibration_Metrics.csv`
- **Table S11. Paired-bootstrap comparison of model calibration.** `tables/Table_S11_Paired_Calibration_Comparison.csv`
- **Table S12. Decision-curve analysis summary.** `tables/Table_S12_Decision_Curve_Summary.csv`
- **Table S13. Paired-bootstrap comparison of leading ESM-2 classifiers.** `tables/Table_S13_Leading_ESM2_Classifier_Comparison.csv`
- **Table S14. Cross-validated canonical-correlation analysis of representation complementarity.** `tables/Table_S14_Cross_Validated_Representation_Complementarity.csv`
- **Table S15. Feature-fusion versus single-representation model performance.** `tables/Table_S15_Feature_Fusion_Model_Comparison.csv`
- **Table S16. Stable ESM-2 latent-feature importance across cross-validation folds.** `tables/Table_S16_ESM2_Feature_Importance_Stability.csv`
- **Table S17. Peptide-level residue perturbation summary.** `tables/Table_S17_Peptide_Residue_Perturbation_Summary.csv`
- **Table S18. Motif-context sensitivity in consensus hard cases.** `tables/Table_S18_Motif_Context_Sensitivity.csv`

## Supplementary figures

- **Figure S1. Additional descriptor and model diagnostics.** Supplied as PDF and TIFF in `figures/`.
- **Figure S2. Probability shifts and hard-case proximity.** Supplied as PDF and TIFF in `figures/`.
- **Figure S3. Sequence families and development-neighborhood context.** Supplied as PDF and TIFF in `figures/`.

## Selection notes

Model binaries, raw bootstrap replicates, test-prediction dumps, plotting intermediates, environment checks, and QC-only outputs were intentionally excluded. Main-text Tables 1-4 remain in `manuscript/tables/` and are not duplicated here.

The CSV column headers are preserved exactly from the validated result files. Table labels and publication headings are provided by the filenames, this index, and `supplementary_file_manifest.csv`.
