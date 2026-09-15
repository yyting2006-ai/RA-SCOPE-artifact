# RA-SCOPE research artifact

Reproducibility artifact for the paper

> **RA-SCOPE: An Auditable Zero-Parameter Ensemble for Ordinal Difficulty
> Assessment of Second-Language Chinese Graded Readers**

submitted to the **IEEE AI for Science Congress 2026** (IEEE SocialCom 2026),
Kuala Lumpur, 27--30 December 2026.

**Authors.** Tingrui You (Hainan International College), Na Zhang, Ling Xiong,
and Jimei Li (corresponding author; School of Information Science, Beijing
Language and Culture University).

**Funding.** National College Students' Innovation and Entrepreneurship Training
Program, No. 202610032033.

---

## What the artifact contains

The artifact packages everything needed to re-derive the numbers in the paper:

* the executable, parameter-free support-backed consensus projection module and
  its invariant tests;
* per-book predictions for all 148 evaluated books (120-book main collection and
  28-book robustness collection) for every reported method;
* the reported metric tables, the paired lineage-bootstrap intervals, the
  support-state breakdown, and the support-cutoff sensitivity grid;
* the length-residual, counter-length, source-slice, construct-alignment and
  observable-modality-routing diagnostics;
* the deterministic Qwen2.5-3B rubric prompt, token probabilities, metrics and
  compute record;
* the post-hoc **learned-fusion baseline** added for Table IV, with its
  cross-validation protocol, per-book out-of-fold predictions and bootstrap
  intervals;
* the IEEE manuscript source, its vector figures, and the figure-generation
  script;
* provenance material: claim registry, citation ledger, annotation-reliability
  summary, experiment cards, figure contract, and a SHA-256 manifest.

## Quick verification

All commands run from the artifact root. No network access is required.

```bash
# 1. recompute every reported table from results/global/predictions.csv
python scripts/reproduce_paper_tables.py --artifact-root . \
    --output metadata/reproduced_results.json

# 2. reproduce the learned-fusion baseline of Table IV
python scripts/analyze_learned_fusion.py --artifact-root . \
    --output results/learned_fusion/learned_fusion_results.json

# 3. run the projection-module invariant tests
python -m pytest -q

# 4. check every packaged file against the hash manifest
python scripts/verify_package_manifest.py
```

Step 1 recomputes QWK, MAE, exact accuracy, within-one accuracy and the
error-above-one rate directly from the 148 packaged predictions, verifies that
all 38 (main) and 7 (robustness) activated consensus grades are preserved, checks
that the four headline paired QWK intervals exclude zero, and checks the
support-cutoff sensitivity range.

Step 2 fits a multinomial logistic regression over the four experts' continuous
ordinal scores under stratified $k$-fold cross-validation ($k\in\{4,5,6,10\}$,
ten seeds each) and reports the paired book-level bootstrap against RA-SCOPE and
the uniform pool.

Step 3 runs 19 invariant tests covering monotonicity, activation, preservation,
invalid input handling and the zero-parameter property. Step 4 confirms that
every packaged file matches `metadata/MANIFEST.sha256`; interpreter and
test-runner caches (`__pycache__`, `.pytest_cache`) are ignored, so the check
still succeeds after step 3 has been run.

To rebuild the manuscript:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

To regenerate the vector performance and validity figures from the numbers
reported in the paper:

```bash
python paper/figures/make_figures.py
```

## Artifact map

| Path | Contents |
|---|---|
| `paper/` | IEEEtran manuscript source (`main.tex`), `refs.bib`, compiled PDF, figure sources |
| `paper/figures/fig1_dataflow.tex` | Fig. 1 (method diagram) as TikZ vector source |
| `paper/figures/fig2_performance.pdf`, `fig3_validity.pdf` | Figs. 2--3 as vector PDFs |
| `paper/figures/raster_originals/` | the original raster figures, kept for provenance |
| `src/scope_coling/ra_scope.py` | the parameter-free support-backed consensus projection |
| `tests/test_ra_scope.py` | monotonicity, activation, preservation, input and parameter-count tests |
| `results/global/` | per-book predictions, ablation metrics, paired intervals, support states, sensitivity grid |
| `results/qwen/` | deterministic rubric prompt, model record, probabilities, metrics |
| `results/diagnostics/` | length-residual, counter-length, source-slice, construct and routing evidence |
| `results/learned_fusion/` | Table IV: cross-validated stacked-fusion results and out-of-fold predictions |
| `evidence/` | claim registry, citation ledger, reliability summary, figure contract, research checklist |
| `scripts/` | the reproduction, learned-fusion and manifest-verification scripts |
| `metadata/MANIFEST.sha256` | hash inventory for every packaged file except the manifest itself |
| `.gitattributes`, `.gitignore` | the package is designed to be version-controlled; `.gitattributes` disables end-of-line conversion because the manifest hashes raw bytes |

## A note on the figures and on the added baseline

Two things changed relative to the earlier version of this study that produced
the packaged result tables, and both are recorded here for transparency.

1. **Figures.** The original figures were raster images. For the IEEE
   submission Fig. 1 was redrawn in TikZ and Figs. 2--3 were redrawn as vector
   PDFs from the numbers reported in the paper; the raster originals are kept
   under `paper/figures/raster_originals/`. No numerical value changed.
2. **Learned-fusion baseline.** The paper now reports a post-hoc comparison
   against a fitted fusion over the same four experts (Table IV and
   `results/learned_fusion/`). It is labelled post-hoc in the manuscript because
   it was computed after the evaluation protocol was frozen. It reaches higher
   QWK and materially lower MAE, but it requires labelled books from the target
   collection, has no per-item preservation guarantee, and its decision rule is
   not inspectable; the paper presents it as the measured cost of auditability
   rather than as a defect of the contract.

All other reported numbers are unchanged from the frozen artifacts.

## Data boundary

The artifact contains identifiers, source families, derived annotations,
measurements, probabilities and predictions. It excludes copyrighted story text
and illustrations, account credentials, private paths, and author identities in
the sense of personal data. Original materials remain available from their
respective providers subject to item-level licence and attribution requirements.
Qwen inference is represented by its exact prompt, model commit, token
probabilities and runtime record; model weights and restricted book text are not
redistributed. See `DATA_AND_LICENSES.md` for the licence position on each
source collection.

## Environment

The projection module requires Python 3.11+ and PyTorch; the analysis and
reproduction scripts require NumPy, pandas and scikit-learn; the tests use
pytest; manuscript compilation requires a LaTeX installation providing
`IEEEtran.cls`, `IEEEtran.bst`, `tikz` and `helvet`. The packaged results were
produced with Python 3.11.9, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0 and
MiKTeX 25.12.
