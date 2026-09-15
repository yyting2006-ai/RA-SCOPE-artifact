# E13 — Support states and sensitivity

## Question

Do explicit support states stratify error, and are point predictions stable across reasonable support and margin settings?

## Grid

- Source-distance percentile cutoffs: .50, .60, .70, .80, .90, .95
- Margin relaxations: .25, .50, .75
- Bootstrap unit for paired method comparisons: original-work lineage
- Resamples: 10,000

## Artifacts

- State metrics: `results/global/support_state_metrics.csv`
- Sensitivity grid: `results/global/sensitivity.csv`
- Evidence figure: `paper/figures/support.png`
- Figure provenance: `evidence/figure_set_manifest.json`

## Results

Main exact accuracy is .711 for 38 support-backed consensuses, .565 for 46 supported conflicts, and .500 across 36 no-support books. Main QWK remains between .817 and .825 over the support grid. Robustness QWK is .786 throughout. Point predictions are invariant over the tested margin-relaxation values.
