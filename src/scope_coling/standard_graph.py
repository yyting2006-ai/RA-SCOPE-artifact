from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StandardNode:
    """One cumulative criterion threshold in the executable standard."""

    id: str
    criterion_id: str
    dimension: str
    level: int
    threshold_prompt_zh: str
    modalities: tuple[str, ...]
    source_locator: str
    source_text_status: str
    operationalization_status: str
    observable_candidates: tuple[str, ...]


class StandardGraph:
    """Validated product of criterion-specific ordinal chains.

    Edge ``(u, v)`` means cumulative support at ``u`` must be greater
    than or equal to cumulative support at ``v``.
    """

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        level_count: int,
        criteria_order: Sequence[str],
        nodes: Sequence[StandardNode],
        chains: dict[str, Sequence[int]],
    ) -> None:
        self.metadata = dict(metadata)
        self.level_count = int(level_count)
        self.criteria_order = tuple(criteria_order)
        self.nodes = tuple(nodes)
        self.chains = {key: tuple(value) for key, value in chains.items()}
        self.node_index = {node.id: index for index, node in enumerate(self.nodes)}
        self.edges = tuple(
            (chain[index], chain[index + 1])
            for criterion_id in self.criteria_order
            for chain in [self.chains[criterion_id]]
            for index in range(len(chain) - 1)
        )
        self.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "StandardGraph":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        criteria_order = payload["criteria_order"]
        criteria_by_id = {criterion["id"]: criterion for criterion in payload["criteria"]}
        nodes: list[StandardNode] = []
        chains: dict[str, list[int]] = {}

        for criterion_id in criteria_order:
            if criterion_id not in criteria_by_id:
                raise ValueError(f"Criterion {criterion_id!r} is missing from criteria.")
            criterion = criteria_by_id[criterion_id]
            chain: list[int] = []
            for raw_node in sorted(criterion["nodes"], key=lambda item: item["level"]):
                chain.append(len(nodes))
                nodes.append(
                    StandardNode(
                        id=raw_node["id"],
                        criterion_id=criterion_id,
                        dimension=criterion["dimension"],
                        level=int(raw_node["level"]),
                        threshold_prompt_zh=raw_node["threshold_prompt_zh"],
                        modalities=tuple(raw_node["modalities"]),
                        source_locator=raw_node["source_locator"],
                        source_text_status=raw_node["source_text_status"],
                        operationalization_status=raw_node["operationalization_status"],
                        observable_candidates=tuple(raw_node.get("observable_candidates", [])),
                    )
                )
            chains[criterion_id] = chain

        return cls(
            metadata=payload["metadata"],
            level_count=int(payload["level_count"]),
            criteria_order=criteria_order,
            nodes=nodes,
            chains=chains,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_criteria(self) -> int:
        return len(self.criteria_order)

    def validate(self) -> None:
        if self.level_count < 2:
            raise ValueError("level_count must be at least 2.")
        if not self.criteria_order:
            raise ValueError("At least one criterion is required.")
        if len(set(self.criteria_order)) != len(self.criteria_order):
            raise ValueError("criteria_order contains duplicates.")
        if len(self.node_index) != len(self.nodes):
            raise ValueError("Node IDs must be unique.")
        if set(self.criteria_order) != set(self.chains):
            raise ValueError("criteria_order and chain keys differ.")

        expected_levels = list(range(1, self.level_count + 1))
        occupied: set[int] = set()
        for criterion_id in self.criteria_order:
            chain = self.chains[criterion_id]
            if len(chain) != self.level_count:
                raise ValueError(
                    f"Criterion {criterion_id!r} has {len(chain)} nodes; "
                    f"expected {self.level_count}."
                )
            levels = [self.nodes[index].level for index in chain]
            if levels != expected_levels:
                raise ValueError(
                    f"Criterion {criterion_id!r} levels are {levels}; "
                    f"expected {expected_levels}."
                )
            for index in chain:
                if index < 0 or index >= len(self.nodes):
                    raise ValueError(f"Node index {index} is out of range.")
                if index in occupied:
                    raise ValueError(f"Node index {index} occurs in multiple chains.")
                occupied.add(index)
                if self.nodes[index].criterion_id != criterion_id:
                    raise ValueError("Node criterion does not match its chain.")
        if occupied != set(range(len(self.nodes))):
            raise ValueError("Every node must occur in exactly one chain.")

    def chain_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.chains[criterion_id] for criterion_id in self.criteria_order)

    def adjacency_matrix(self, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=dtype)
        for source, target in self.edges:
            matrix[source, target] = 1
        return matrix

    @staticmethod
    def pava_nonincreasing(
        values: Sequence[float] | np.ndarray,
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact weighted L2 projection onto ``x[0] >= ... >= x[n-1]``."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("PAVA expects a one-dimensional sequence.")
        if array.size == 0:
            return array.copy()
        weight_array = (
            np.ones_like(array)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if weight_array.shape != array.shape:
            raise ValueError("weights must have the same shape as values.")
        if np.any(weight_array <= 0):
            raise ValueError("PAVA weights must be positive.")

        # A block stores [start, end_exclusive, total_weight, weighted_mean].
        blocks: list[list[float]] = []
        for index, (value, weight) in enumerate(zip(array, weight_array, strict=True)):
            blocks.append([float(index), float(index + 1), float(weight), float(value)])
            while len(blocks) >= 2 and blocks[-2][3] < blocks[-1][3]:
                right = blocks.pop()
                left = blocks.pop()
                merged_weight = left[2] + right[2]
                merged_mean = (left[2] * left[3] + right[2] * right[3]) / merged_weight
                blocks.append([left[0], right[1], merged_weight, merged_mean])

        projected = np.empty_like(array)
        for start, end, _, mean in blocks:
            projected[int(start) : int(end)] = mean
        return projected

    def project_numpy(self, values: np.ndarray) -> np.ndarray:
        """Project arrays whose final axis follows this graph's node order."""

        array = np.asarray(values)
        if array.shape[-1] != self.num_nodes:
            raise ValueError(
                f"Final dimension is {array.shape[-1]}; expected {self.num_nodes}."
            )
        original_shape = array.shape
        flat = array.astype(np.float64, copy=True).reshape(-1, self.num_nodes)
        for row in flat:
            for chain in self.chain_indices():
                indices = np.asarray(chain, dtype=np.int64)
                row[indices] = self.pava_nonincreasing(row[indices])
        projected = flat.reshape(original_shape)
        if np.issubdtype(array.dtype, np.floating):
            projected = projected.astype(array.dtype, copy=False)
        return projected

    def violation_count(self, values: np.ndarray, tolerance: float = 1e-7) -> int:
        array = np.asarray(values)
        if array.shape[-1] != self.num_nodes:
            raise ValueError(
                f"Final dimension is {array.shape[-1]}; expected {self.num_nodes}."
            )
        flat = array.reshape(-1, self.num_nodes)
        return int(
            sum(
                np.count_nonzero(flat[:, source] + tolerance < flat[:, target])
                for source, target in self.edges
            )
        )

    def violation_rate(self, values: np.ndarray, tolerance: float = 1e-7) -> float:
        array = np.asarray(values)
        rows = int(np.prod(array.shape[:-1])) if array.ndim > 1 else 1
        denominator = rows * len(self.edges)
        return 0.0 if denominator == 0 else self.violation_count(array, tolerance) / denominator

    def criterion_scores(self, cumulative_probabilities: np.ndarray) -> np.ndarray:
        """Average cumulative support into one [0, 1] severity per criterion."""

        array = np.asarray(cumulative_probabilities)
        if array.shape[-1] != self.num_nodes:
            raise ValueError(
                f"Final dimension is {array.shape[-1]}; expected {self.num_nodes}."
            )
        scores = [array[..., list(chain)].mean(axis=-1) for chain in self.chain_indices()]
        return np.stack(scores, axis=-1)

    def iter_node_prompts(self) -> Iterable[str]:
        for node in self.nodes:
            yield node.threshold_prompt_zh

