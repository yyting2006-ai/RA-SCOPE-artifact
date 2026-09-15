# E14 — Expert diversity and probability diagnostics

## Question

Are the four pooled experts complementary, and does projection improve probability quality or only point labels?

## Artifacts

- Diversity analysis: `scripts/reproduce_paper_tables.py`
- Diversity results: `results/diagnostics/expert_diversity.json`
- Pairwise table: `results/diagnostics/pairwise_disagreement.csv`
- Probability metrics: `results/global/ablation_metrics.csv`

## Results

On the main set, 104/120 books have expert disagreement, mean pairwise disagreement is .576, and the experts form 42 prediction patterns. Each expert is uniquely exact on at least four books. At least one expert is exact on .858 of books; RA-SCOPE is exact on 60 disagreement cases versus 56 for uniform pooling.

RA-SCOPE improves uniform pooling on ordinal Brier (.083 vs .085), NLL (1.411 vs 1.443), and MAE risk--coverage AUC (.418 vs .437), while ECE changes from .187 to .203. Length has lower Brier/NLL but worse ECE and risk--coverage behavior.

