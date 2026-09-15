# E12 — Support-backed full-consensus projection

## Question

Can a zero-parameter projection improve a heterogeneous ordinal pool while guaranteeing preservation only when both anchors agree and at least one anchor is source-supported?

## Method contract

- Experts: length ordinal, character TF-IDF ordinal, Standard measurements, fine-tuned MacBERT
- Pool: uniform cumulative-probability mean, weight .25 per expert
- Anchors: length and character
- Activation: all anchors have the same point grade and at least one anchor has source-distance percentile at most .70
- Projection: unique Euclidean projection onto the monotone retained-margin box
- Retained anchor margin: .50
- Fusion/projection trainable parameters: 0

## Inputs and outputs

- Module: `src/scope_coling/ra_scope.py`
- Analysis: `scripts/reproduce_paper_tables.py`
- Results: `results/global/summary.json`
- Predictions: `results/global/predictions.csv`
- Paired intervals: `results/global/paired_intervals.csv`

## Results

- Source out-of-fold (96): QWK .858, MAE .406
- Main (120): QWK .823, MAE .417, exact .592, error above one level .008
- Robustness (28): QWK .786, MAE .393, exact .607, error above one level 0
- Preservation: 32/32 source, 38/38 main, and 7/7 robustness activated decisions

On the main set, paired QWK intervals exclude zero against uniform pooling, length, fine-tuned MacBERT, and Qwen2.5-3B. The support-disabled full-agreement ablation has identical point predictions on these samples; support determines when the projection guarantee is licensed.

