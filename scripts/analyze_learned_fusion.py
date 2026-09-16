"""Post-hoc learned-fusion baseline for RA-SCOPE (Section VI-C / Table IV).

Question answered: if a fitted fusion layer is placed over the same four expert
scores, how much better does it get, and at what cost?

Protocol
--------
* features  : the four experts' continuous ordinal scores shipped in
              results/global/predictions.csv
              (score_length_ordinal, score_char_tfidf,
               score_standard_measurements, score_finetuned_macbert), i.e. four
              features per book
* model     : multinomial logistic regression (scikit-learn)
* evaluation: stratified k-fold cross-validation, repeated over ten seeds.
              Folds are grouped by original-work lineage (split_group), matching
              the main protocol; on this collection every lineage contains a
              single evaluated book, so grouped and stratified folds coincide.
              The per-book consensus of the ten out-of-fold predictions is the
              reported stacked prediction, and it is what the paired book-level
              bootstrap is computed on; per-seed means are reported alongside as
              a stability check.
* comparators: RA-SCOPE and the uniform pool as shipped; these require no
              fitting, so their full-sample values are used as reported

The comparison is deliberately conservative for the stacked model: the
comparators use the whole collection, whereas the stack is always scored
out-of-fold.

Usage
-----
    python scripts/analyze_learned_fusion.py --artifact-root . \
        --output results/learned_fusion/learned_fusion_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

EXPERTS = [
    ("length_ordinal", "score_length_ordinal"),
    ("char_tfidf", "score_char_tfidf"),
    ("standard_measurements", "score_standard_measurements"),
    ("finetuned_macbert", "score_finetuned_macbert"),
]
COMPARATORS = {
    "ra_scope": "prediction_ra_scope",
    "uniform_pool": "prediction_uniform_pool",
}
FOLDS = (4, 5, 6, 10)
SEEDS = range(10)
N_BOOT = 10_000
BOOT_SEED = 20260915


def read_rows(path: Path, dataset: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    if not rows:
        raise SystemExit(f"No rows for dataset={dataset!r} in {path}")
    return rows


def qwk(gold: np.ndarray, pred: np.ndarray, levels: int = 6) -> float:
    n = gold.size
    observed = np.zeros((levels, levels))
    for g, p in zip(gold.astype(int), pred.astype(int)):
        observed[g - 1, p - 1] += 1.0
    gold_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    denom = float((levels - 1) ** 2)
    w_obs = w_exp = 0.0
    for i in range(levels):
        for j in range(levels):
            w = ((i - j) ** 2) / denom
            w_obs += w * observed[i, j] / n
            w_exp += w * (gold_hist[i] * pred_hist[j]) / (n * n)
    return 1.0 - w_obs / w_exp


def metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(gold - pred)
    return {
        "qwk": qwk(gold, pred),
        "mae": float(err.mean()),
        "exact_accuracy": float((err == 0).mean()),
        "error_above_one_rate": float((err > 1).mean()),
    }


def cross_val_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions with folds grouped by original-work lineage."""
    oof = np.zeros(y.size)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(X, y, groups):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X[train]), y[train])
        oof[test] = clf.predict(scaler.transform(X[test]))
    return oof


def paired_bootstrap_seeds(gold, seed_preds, other, draws=N_BOOT, seed=BOOT_SEED):
    """Paired book bootstrap for a ten-seed mean.

    Each draw resamples books, recomputes the metric for every seed's
    out-of-fold prediction and for the comparator, and differences the two
    ten-seed means. The resulting interval therefore belongs to exactly the
    quantity reported in the table: the mean over ten seeds.
    """
    rng = np.random.default_rng(seed)
    n = gold.size
    d_qwk = np.empty(draws)
    d_mae = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        g = gold[idx]
        qa = np.mean([qwk(g, p[idx]) for p in seed_preds])
        ma = np.mean([np.abs(g - p[idx]).mean() for p in seed_preds])
        qb = qwk(g, other[idx])
        mb = np.abs(g - other[idx]).mean()
        d_qwk[i] = qa - qb
        d_mae[i] = ma - mb
    return {
        "d_qwk": float(np.mean([qwk(gold, p) for p in seed_preds]) - qwk(gold, other)),
        "d_qwk_ci95": [float(np.percentile(d_qwk, 2.5)), float(np.percentile(d_qwk, 97.5))],
        "d_mae": float(np.mean([np.abs(gold - p).mean() for p in seed_preds])
                       - np.abs(gold - other).mean()),
        "d_mae_ci95": [float(np.percentile(d_mae, 2.5)), float(np.percentile(d_mae, 97.5))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the learned-fusion baseline.")
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="core120")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    rows = read_rows(root / "results" / "global" / "predictions.csv", args.dataset)
    gold = np.array([int(r["gold_level"]) for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    X = np.array([[float(r[col]) for _, col in EXPERTS] for r in rows])
    sizes = np.array([int(v) for v in
                      np.unique(groups, return_counts=True)[1]])

    report: dict[str, object] = {
        "dataset": args.dataset,
        "books": len(rows),
        "features": {
            "names": [name for name, _ in EXPERTS],
            "columns": [col for _, col in EXPERTS],
            "dimensions": len(EXPERTS),
        },
        "protocol": {
            "model": "multinomial logistic regression",
            "cv": f"lineage-grouped stratified k-fold, k in {list(FOLDS)}, "
                  f"{len(list(SEEDS))} seeds",
            "grouping": {
                "key": "split_group (original-work lineage)",
                "lineages": int(np.unique(groups).size),
                "books_per_lineage": {"min": int(sizes.min()), "max": int(sizes.max())},
                "note": "every lineage holds a single evaluated book, so grouped "
                        "and stratified folds coincide on this collection",
            },
            "reported_prediction": "per-book consensus of the ten out-of-fold "
                                   "predictions, which is what the paired "
                                   "book-level bootstrap is computed on",
            "notes": "stack is scored out-of-fold; comparators need no fitting",
        },
        "comparators": {name: metrics(gold, np.array([int(r[col]) for r in rows]))
                        for name, col in COMPARATORS.items()},
        "stability_across_folds": {},
    }

    for folds in FOLDS:
        runs = [metrics(gold, cross_val_scores(X, gold, groups, folds, s)) for s in SEEDS]
        report["stability_across_folds"][f"{folds}-fold"] = {
            "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
            "qwk_sd": float(np.std([r["qwk"] for r in runs])),
            "mae_mean": float(np.mean([r["mae"] for r in runs])),
            "mae_sd": float(np.std([r["mae"] for r in runs])),
        }

    oof_all = np.vstack([cross_val_scores(X, gold, groups, 5, s) for s in SEEDS])
    consensus = np.round(oof_all.mean(axis=0))
    per_seed = [metrics(gold, cross_val_scores(X, gold, groups, 5, s)) for s in SEEDS]
    report["stacked_logistic"] = {
        "reported": "mean over ten seeds at k=5",
        "metrics_mean": {
            k: float(np.mean([r[k] for r in per_seed]))
            for k in ("qwk", "mae", "exact_accuracy", "error_above_one_rate")
        },
        "metrics_sd": {
            k: float(np.std([r[k] for r in per_seed]))
            for k in ("qwk", "mae", "exact_accuracy", "error_above_one_rate")
        },
        "seed_consensus_metrics": metrics(gold, consensus),
        "seed_consensus_note": "aggregating the ten fold assignments into a single "
                               "per-book consensus raises QWK but is not the quantity "
                               "the table reports; shown for transparency",
    }
    report["paired_bootstrap_vs"] = {
        name: paired_bootstrap_seeds(gold, oof_all, np.array([int(r[col]) for r in rows]))
        for name, col in COMPARATORS.items()
    }
    # differences of the point estimates, so the table's delta row is exactly the
    # difference of the two rows above it
    ref = report["stacked_logistic"]["metrics_mean"]
    report["delta_point_estimates_vs"] = {}
    for name, col in COMPARATORS.items():
        b = metrics(gold, np.array([int(r[col]) for r in rows]))
        report["delta_point_estimates_vs"][name] = {
            k: float(ref[k] - b[k]) for k in
            ("qwk", "mae", "exact_accuracy", "error_above_one_rate")
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        sidecar = args.output.with_name("learned_fusion_predictions.csv")
        with sidecar.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["book_id", "split_group", "source_dataset", "gold_level",
                             "prediction_ra_scope", "prediction_stacked_logistic"]) 
            for row, pred in zip(rows, consensus):
                writer.writerow([row["book_id"], row["split_group"], row["source_dataset"],
                                 row["gold_level"], row["prediction_ra_scope"], int(pred)])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
