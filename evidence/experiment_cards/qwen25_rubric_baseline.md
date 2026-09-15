# E11 — Qwen2.5-3B rubric baseline

## Question

How does a reproducible open-weight instruction model perform when the six-level Chinese grading rubric is supplied directly, without demonstrations or target-label tuning?

## Configuration

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model commit: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Precision: bfloat16
- Decision: restricted next-token softmax over the six single-token level candidates
- Prompt SHA-256: `342ddb11dd5975bffae53a487a700025a73f7738f18ef1959f8465a85827816e`
- Trainable parameters and demonstrations: 0

## Inputs and outputs

- Script: `scripts/run_qwen25_rubric_baseline.py`
- Prompt: `results/qwen/prompt.txt`
- Per-book probabilities: `results/qwen/predictions.csv`
- Metrics: `results/qwen/summary.json`

## Results

- Source development (96): QWK .626, MAE .958
- Main (120): QWK .538, MAE .825, exact .425, error above one level .183
- Robustness (28): QWK .603, MAE .714

The baseline is a deterministic, text-only 3B reference. It does not represent larger or proprietary LLMs.

