$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$resultsRoot = Join-Path $projectRoot 'results'
$manuscriptRoot = Join-Path $projectRoot 'manuscript'
$suppRoot = Join-Path $manuscriptRoot 'supplementary_files'
$tablesRoot = Join-Path $suppRoot 'tables'
$figuresRoot = Join-Path $suppRoot 'figures'

New-Item -ItemType Directory -Path $tablesRoot -Force | Out-Null
New-Item -ItemType Directory -Path $figuresRoot -Force | Out-Null

$tables = @(
    @{ Number = 'S1';  Title = 'Physicochemical descriptor statistics'; Source = 'step24_physicochemical_statistics_manuscript.csv'; File = 'Table_S1_Physicochemical_Descriptor_Statistics.csv' },
    @{ Number = 'S2';  Title = 'Highly correlated physicochemical descriptor pairs'; Source = 'step26_high_correlation_pairs.csv'; File = 'Table_S2_Highly_Correlated_Descriptor_Pairs.csv' },
    @{ Number = 'S3';  Title = 'Cross-validated performance of traditional classifiers'; Source = 'step35_traditional_model_comparison.csv'; File = 'Table_S3_Traditional_Model_Performance.csv' },
    @{ Number = 'S4';  Title = 'Permutation importance of traditional XGBoost descriptors'; Source = 'step38_xgboost_permutation_importance.csv'; File = 'Table_S4_Traditional_XGBoost_Permutation_Importance.csv' },
    @{ Number = 'S5';  Title = 'Cross-validated performance of ESM-2 classifiers'; Source = 'step52_esm2_model_comparison.csv'; File = 'Table_S5_ESM2_Model_Performance.csv' },
    @{ Number = 'S6';  Title = 'Paired-bootstrap comparison of traditional and ESM-2 representations'; Source = 'step54_paired_bootstrap_summary.csv'; File = 'Table_S6_Traditional_vs_ESM2_Paired_Bootstrap.csv' },
    @{ Number = 'S7';  Title = 'Consensus hard-case peptides and prediction evidence'; Source = 'step59_consensus_hard_cases_manuscript.csv'; File = 'Table_S7_Consensus_Hard_Case_Peptides.csv' },
    @{ Number = 'S8';  Title = 'Sequence families and redundancy summary'; Source = 'step66_sequence_family_summary.csv'; File = 'Table_S8_Sequence_Family_Summary.csv' },
    @{ Number = 'S9';  Title = 'Model performance stratified by sequence novelty'; Source = 'step67_sequence_novelty_performance.csv'; File = 'Table_S9_Sequence_Novelty_Performance.csv' },
    @{ Number = 'S10'; Title = 'Probability calibration metrics'; Source = 'step71_calibration_metrics.csv'; File = 'Table_S10_Probability_Calibration_Metrics.csv' },
    @{ Number = 'S11'; Title = 'Paired-bootstrap comparison of model calibration'; Source = 'step72_paired_calibration_comparison.csv'; File = 'Table_S11_Paired_Calibration_Comparison.csv' },
    @{ Number = 'S12'; Title = 'Decision-curve analysis summary'; Source = 'step73_decision_curve_model_summary.csv'; File = 'Table_S12_Decision_Curve_Summary.csv' },
    @{ Number = 'S13'; Title = 'Paired-bootstrap comparison of leading ESM-2 classifiers'; Source = 'step75_esm2_rf_vs_xgboost_paired_comparison.csv'; File = 'Table_S13_Leading_ESM2_Classifier_Comparison.csv' },
    @{ Number = 'S14'; Title = 'Cross-validated canonical-correlation analysis of representation complementarity'; Source = 'step78_cv_cca_dimension_summary.csv'; File = 'Table_S14_Cross_Validated_Representation_Complementarity.csv' },
    @{ Number = 'S15'; Title = 'Feature-fusion versus single-representation model performance'; Source = 'step79_fusion_model_comparison.csv'; File = 'Table_S15_Feature_Fusion_Model_Comparison.csv' },
    @{ Number = 'S16'; Title = 'Stable ESM-2 latent-feature importance across cross-validation folds'; Source = 'step81_esm2_cv_feature_importance_summary.csv'; File = 'Table_S16_ESM2_Feature_Importance_Stability.csv' },
    @{ Number = 'S17'; Title = 'Peptide-level residue perturbation summary'; Source = 'step82_peptide_perturbation_summary.csv'; File = 'Table_S17_Peptide_Residue_Perturbation_Summary.csv' },
    @{ Number = 'S18'; Title = 'Motif-context sensitivity in consensus hard cases'; Source = 'step83_motif_sensitivity_summary.csv'; File = 'Table_S18_Motif_Context_Sensitivity.csv' }
)

foreach ($table in $tables) {
    $sourcePath = Join-Path $resultsRoot $table.Source
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Missing source table: $sourcePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $tablesRoot $table.File) -Force
}

$figureSourceRoot = Join-Path $manuscriptRoot 'supplementary_figures_high_impact'
$figures = @(
    @{ Number = 'S1'; Title = 'Additional descriptor and model diagnostics'; Stem = 'Supplementary_Figure_S1_Additional_Descriptor_and_Model_Diagnostics' },
    @{ Number = 'S2'; Title = 'Probability shifts and hard-case proximity'; Stem = 'Supplementary_Figure_S2_Probability_Shifts_and_Hard_Case_Proximity' },
    @{ Number = 'S3'; Title = 'Sequence families and development-neighborhood context'; Stem = 'Supplementary_Figure_S3_Sequence_Families_and_Development_Neighborhoods' }
)

foreach ($figure in $figures) {
    foreach ($extension in @('.pdf', '.tiff')) {
        $sourcePath = Join-Path $figureSourceRoot ($figure.Stem + $extension)
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Missing source figure: $sourcePath"
        }
        Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $figuresRoot ($figure.Stem + $extension)) -Force
    }
}

$manifest = foreach ($table in $tables) {
    $destination = Join-Path $tablesRoot $table.File
    [pscustomobject]@{
        Type = 'Table'
        Label = "Table $($table.Number)"
        Title = $table.Title
        File = "tables/$($table.File)"
        Source = "results/$($table.Source)"
        SHA256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    }
}
foreach ($figure in $figures) {
    foreach ($extension in @('.pdf', '.tiff')) {
        $file = $figure.Stem + $extension
        $destination = Join-Path $figuresRoot $file
        $manifest += [pscustomobject]@{
            Type = 'Figure'
            Label = "Figure $($figure.Number)"
            Title = $figure.Title
            File = "figures/$file"
            Source = "manuscript/supplementary_figures_high_impact/$file"
            SHA256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        }
    }
}
$manifest | Export-Csv -LiteralPath (Join-Path $suppRoot 'supplementary_file_manifest.csv') -NoTypeInformation -Encoding UTF8

$readme = @(
    '# Supplementary files',
    '',
    'This folder contains the essential, nonredundant supporting tables and publication-ready supplementary figures. Original files in `results/` and the existing manuscript folders were preserved.',
    '',
    '## Supplementary tables',
    ''
)
foreach ($table in $tables) {
    $readme += "- **Table $($table.Number). $($table.Title).** ``tables/$($table.File)``"
}
$readme += @('', '## Supplementary figures', '')
foreach ($figure in $figures) {
    $readme += "- **Figure $($figure.Number). $($figure.Title).** Supplied as PDF and TIFF in ``figures/``."
}
$readme += @(
    '',
    '## Selection notes',
    '',
    'Model binaries, raw bootstrap replicates, test-prediction dumps, plotting intermediates, environment checks, and QC-only outputs were intentionally excluded. Main-text Tables 1-4 remain in `manuscript/tables/` and are not duplicated here.',
    '',
    'The CSV column headers are preserved exactly from the validated result files. Table labels and publication headings are provided by the filenames, this index, and `supplementary_file_manifest.csv`.'
)
Set-Content -LiteralPath (Join-Path $suppRoot 'README.md') -Value $readme -Encoding UTF8

Write-Output "Created $suppRoot"
Write-Output "Tables: $($tables.Count)"
Write-Output "Figures: $($figures.Count) in PDF and TIFF"
