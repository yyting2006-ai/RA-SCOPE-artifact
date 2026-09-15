# Responsible NLP research checklist

## Scientific claims

- The abstract and conclusion report the main and separate robustness collections.
- Every quantitative statement maps to the claim registry and a packaged result artifact.
- Paired comparisons use 10,000 bootstrap samples over original-work lineages.
- Support ranks are described as diagnostics under exchangeability, not correctness probabilities.
- The paper states the upper-level, learner-validity, source-balance, OCR, and model-scope boundaries.

## Data and annotation

- P01--P03 rated books independently; P04 adjudicated and did not enter inter-rater reliability.
- Annotators had International Chinese Education training and are represented by role identifiers in the anonymous materials.
- Original-work lineages do not cross source-development and evaluation collections.
- Restricted story text and images are excluded; first-party source and licence locations are documented.
- No learner records or children's personal data are used.

## Models and computation

- The projection and pooling layer has zero trainable parameters.
- The manuscript specifies expert families, source-only fold grouping, support cutoff, retained margin, bootstrap count, MacBERT seeds and epochs, and the exact Qwen model commit and prompt hash.
- GPU model, elapsed compute, precision, and peak Qwen memory are reported.
- Unit tests cover activation semantics, monotonicity, preservation, invalid inputs, gradients, and parameter count.

## Artifact and anonymity

- The package includes an anonymous manuscript and excludes author names, emails, private paths, credentials, and raw provider content.
- SHA-256 hashes cover every packaged file.
- The method figure is identified as an ImageGen conceptual schematic and is not used as empirical evidence.
- The three quantitative figures are ImageGen PNG rasters anchored to deterministic result artifacts; every visible value, sign, interval, denominator, axis, and route is checked against the packaged data, with prompts and hashes recorded in the figure-set manifest.
