"""Post-hoc learned-fusion baseline for RA-SCOPE (Section VI-C / Table IV).

Question answered: if a fitted fusion layer is placed over the same four expert
scores, how much better does it get, and at what cost?

Protocol
--------
* features  : the four experts' continuous ordinal scores shipped in
              results/global/predictions.csv
              (score_length_ordinal, score_char_tfidf,
               score_standard_measurements, score_finetuned_macbert)
* model     : multinomial logistic regression (scikit-learn)
* evaluation: stratified k-fold cross-validation, repeated over ten seeds; the
              per-book consensus of the out-of-fold predictions is used for the
              paired book-level bootstrap
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
from sklearn.model_selection import StratifiedKFold
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


def cross_val_scores(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    oof = np.zeros(y.size)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in skf.split(X, y):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X[train]), y[train])
        oof[test] = clf.predict(scaler.transform(X[test]))
    return oof


def paired_bootstrap(gold, a, b, draws=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = gold.size
    d_qwk = np.empty(draws)
    d_mae = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        ma, mb = metrics(gold[idx], a[idx]), metrics(gold[idx], b[idx])
        d_qwk[i] = ma["qwk"] - mb["qwk"]
        d_mae[i] = ma["mae"] - mb["mae"]
    return {
        "d_qwk": float(metrics(gold, a)["qwk"] - metrics(gold, b)["qwk"]),
        "d_qwk_ci95": [float(np.percentile(d_qwk, 2.5)), float(np.percentile(d_qwk, 97.5))],
        "d_mae": float(metrics(gold, a)["mae"] - metrics(gold, b)["mae"]),
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
    X = np.array([[float(r[col]) for _, col in EXPERTS] for r in rows])

    report: dict[str, object] = {
        "dataset": args.dataset,
        "books": len(rows),
        "features": [name for name, _ in EXPERTS],
        "protocol": {
            "model": "multinomial logistic regression",
            "cv": f"stratified k-fold, k in {list(FOLDS)}, {len(list(SEEDS))} seeds",
            "notes": "stack is scored out-of-fold; comparators need no fitting",
        },
        "comparators": {name: metrics(gold, np.array([int(r[col]) for r in rows]))
                        for name, col in COMPARATORS.items()},
        "stability_across_folds": {},
    }

    for folds in FOLDS:
        runs = [metrics(gold, cross_val_scores(X, gold, folds, s)) for s in SEEDS]
        report["stability_across_folds"][f"{folds}-fold"] = {
            "qwk_mean": float(np.mean([r["qwk"] for r in runs])),
            "qwk_sd": float(np.std([r["qwk"] for r in runs])),
            "mae_mean": float(np.mean([r["mae"] for r in runs])),
            "mae_sd": float(np.std([r["mae"] for r in runs])),
        }

    oof_all = np.vstack([cross_val_scores(X, gold, 5, s) for s in SEEDS])
    consensus = np.round(oof_all.mean(axis=0))
    per_seed = [metrics(gold, cross_val_scores(X, gold, 5, s)) for s in SEEDS]
    report["stacked_logistic"] = {
        "consensus_metrics": metrics(gold, consensus),
        "per_seed_qwk_mean": float(np.mean([r["qwk"] for r in per_seed])),
        "per_seed_qwk_sd": float(np.std([r["qwk"] for r in per_seed])),
        "per_seed_mae_mean": float(np.mean([r["mae"] for r in per_seed])),
        "per_seed_mae_sd": float(np.std([r["mae"] for r in per_seed])),
    }
    report["paired_bootstrap_vs"] = {
        name: paired_bootstrap(gold, consensus, np.array([int(r[col]) for r in rows]))
        for name, col in COMPARATORS.items()
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
