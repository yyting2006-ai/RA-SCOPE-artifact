from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


EXPECTED = {
    "core120": {
        "ra_scope": {"qwk": 0.8227171998409001, "mae": 0.4166666666666667},
        "uniform_pool": {"qwk": 0.7903837044054951, "mae": 0.4583333333333333},
        "length_ordinal": {"qwk": 0.7667668705296984, "mae": 0.5416666666666666},
        "finetuned_macbert": {"qwk": 0.763140817650876, "mae": 0.5583333333333333},
        "qwen25_3b_rubric": {"qwk": 0.5380247763248451, "mae": 0.825},
    },
    "waveD28": {
        "ra_scope": {"qwk": 0.7861111111111111, "mae": 0.39285714285714285},
        "uniform_pool": {"qwk": 0.7861111111111111, "mae": 0.39285714285714285},
        "length_ordinal": {"qwk": 0.7720488466757124, "mae": 0.42857142857142855},
        "finetuned_macbert": {"qwk": 0.6720351390922401, "mae": 0.5714285714285714},
        "qwen25_3b_rubric": {"qwk": 0.6031886625332152, "mae": 0.7142857142857143},
    },
}

COLUMNS = {
    "ra_scope": "prediction_ra_scope",
    "uniform_pool": "prediction_uniform_pool",
    "length_ordinal": "prediction_length_ordinal",
    "finetuned_macbert": "prediction_finetuned_macbert",
    "qwen25_3b_rubric": "prediction_qwen25_3b_rubric",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def qwk(gold: Iterable[int], prediction: Iterable[int], levels: int = 6) -> float:
    gold_values = list(gold)
    pred_values = list(prediction)
    n = len(gold_values)
    observed = [[0.0 for _ in range(levels)] for _ in range(levels)]
    gold_hist = [0.0] * levels
    pred_hist = [0.0] * levels
    for g, p in zip(gold_values, pred_values, strict=True):
        observed[g - 1][p - 1] += 1.0
        gold_hist[g - 1] += 1.0
        pred_hist[p - 1] += 1.0
    weighted_observed = 0.0
    weighted_expected = 0.0
    denominator = float((levels - 1) ** 2)
    for i in range(levels):
        for j in range(levels):
            weight = ((i - j) ** 2) / denominator
            weighted_observed += weight * observed[i][j] / n
            weighted_expected += weight * (gold_hist[i] * pred_hist[j]) / (n * n)
    return 1.0 - weighted_observed / weighted_expected


def metrics(rows: list[dict[str, str]], prediction_column: str) -> dict[str, float]:
    gold = [int(row["gold_level"]) for row in rows]
    pred = [int(row[prediction_column]) for row in rows]
    errors = [abs(g - p) for g, p in zip(gold, pred, strict=True)]
    return {
        "qwk": qwk(gold, pred),
        "mae": sum(errors) / len(errors),
        "exact_accuracy": sum(error == 0 for error in errors) / len(errors),
        "within_one_accuracy": sum(error <= 1 for error in errors) / len(errors),
        "error_above_one_rate": sum(error > 1 for error in errors) / len(errors),
    }


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected:.15f}, obtained {actual:.15f}")


def resolve_results_root(artifact_root: Path) -> Path:
    packaged = artifact_root / "results" / "global"
    if packaged.exists():
        return artifact_root / "results"
    project = artifact_root / "outputs" / "support_backed_consensus_projection_v2.0"
    if project.exists():
        return artifact_root / "outputs"
    raise FileNotFoundError("Could not locate packaged results/ or project outputs/ directories")


def path_for(results_root: Path, packaged: str, project: str) -> Path:
    candidate = results_root / packaged
    return candidate if candidate.exists() else results_root / project


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and verify the paper's reported result tables.")
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    results_root = resolve_results_root(artifact_root)

    predictions_path = path_for(
        results_root,
        "global/predictions.csv",
        "support_backed_consensus_projection_v2.0/predictions.csv",
    )
    rows = read_csv(predictions_path)
    reproduced: dict[str, dict[str, dict[str, float]]] = {}
    for dataset, expected_methods in EXPECTED.items():
        subset = [row for row in rows if row["dataset"] == dataset]
        if len(subset) != (120 if dataset == "core120" else 28):
            raise AssertionError(f"Unexpected row count for {dataset}: {len(subset)}")
        reproduced[dataset] = {}
        for method, expected_metrics in expected_methods.items():
            observed = metrics(subset, COLUMNS[method])
            close(observed["qwk"], expected_metrics["qwk"])
            close(observed["mae"], expected_metrics["mae"])
            reproduced[dataset][method] = observed

    main_rows = [row for row in rows if row["dataset"] == "core120"]
    robust_rows = [row for row in rows if row["dataset"] == "waveD28"]
    main_active = [row for row in main_rows if row["support_backed_full_consensus"] == "True"]
    robust_active = [row for row in robust_rows if row["support_backed_full_consensus"] == "True"]
    if len(main_active) != 38 or len(robust_active) != 7:
        raise AssertionError("Activated-consensus counts do not match the manuscript")
    if any(int(row["prediction_ra_scope"]) != int(row["protected_grade"]) for row in main_active + robust_active):
        raise AssertionError("An activated consensus grade was not preserved")

    paired_path = path_for(
        results_root,
        "global/paired_intervals.csv",
        "support_backed_consensus_projection_v2.0/paired_intervals.csv",
    )
    paired = read_csv(paired_path)
    comparators = {"uniform_pool", "length_ordinal", "finetuned_macbert", "qwen25_3b_rubric"}
    headline_intervals = {
        row["comparator"]: {
            "low": float(row["qwk_delta_low"]),
            "median": float(row["qwk_delta_median"]),
            "high": float(row["qwk_delta_high"]),
        }
        for row in paired
        if row["dataset"] == "core120" and row["comparator"] in comparators
    }
    if set(headline_intervals) != comparators or any(interval["low"] <= 0 for interval in headline_intervals.values()):
        raise AssertionError("A headline paired QWK interval does not exclude zero")

    sensitivity_path = path_for(
        results_root,
        "global/sensitivity.csv",
        "support_backed_consensus_projection_v2.0/sensitivity.csv",
    )
    sensitivity = read_csv(sensitivity_path)
    main_qwk = [float(row["qwk"]) for row in sensitivity if row["dataset"] == "core120"]
    robust_qwk = [float(row["qwk"]) for row in sensitivity if row["dataset"] == "waveD28"]
    sensitivity_summary = {
        "main_qwk_min": min(main_qwk),
        "main_qwk_max": max(main_qwk),
        "robustness_qwk_min": min(robust_qwk),
        "robustness_qwk_max": max(robust_qwk),
    }
    close(sensitivity_summary["robustness_qwk_min"], sensitivity_summary["robustness_qwk_max"])

    report = {
        "status": "pass",
        "prediction_rows": len(rows),
        "recomputed_metrics": reproduced,
        "activated_consensus_preservation": {
            "main": f"{len(main_active)}/{len(main_active)}",
            "robustness": f"{len(robust_active)}/{len(robust_active)}",
        },
        "headline_qwk_intervals": headline_intervals,
        "sensitivity": sensitivity_summary,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
