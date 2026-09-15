from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import GraphIsotonicProjection


class EmpiricalSupportCalibrator:
    """Map label-free source distances to empirical out-of-support percentiles.

    The reference array is collected from source-group holdout episodes. A value
    near zero denotes strong source support and a value near one denotes a
    distance at or beyond the source episode tail. No labels are consumed here.
    """

    def __init__(self) -> None:
        self._reference: np.ndarray | None = None

    def fit(self, source_episode_distances: np.ndarray) -> "EmpiricalSupportCalibrator":
        distances = np.asarray(source_episode_distances, dtype=np.float64)
        if distances.ndim != 3:
            raise ValueError("source_episode_distances must have shape [episodes, criteria, experts].")
        if distances.shape[0] < 2:
            raise ValueError("At least two source episodes are required.")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("Support distances must be finite and nonnegative.")
        self._reference = np.sort(distances, axis=0)
        return self

    def transform(self, distances: np.ndarray) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("The support calibrator has not been fitted.")
        values = np.asarray(distances, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != self._reference.shape[1:]:
            raise ValueError("distances must have shape [items, criteria, experts].")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Support distances must be finite and nonnegative.")
        output = np.empty_like(values)
        denominator = self._reference.shape[0] + 1.0
        for criterion in range(values.shape[1]):
            for expert in range(values.shape[2]):
                output[:, criterion, expert] = np.searchsorted(
                    self._reference[:, criterion, expert],
                    values[:, criterion, expert],
                    side="right",
                ) / denominator
        return output

    def fit_transform(self, source_episode_distances: np.ndarray) -> np.ndarray:
        return self.fit(source_episode_distances).transform(source_episode_distances)


@dataclass(frozen=True)
class ShiftGateOutput:
    weights: Tensor
    support_score: Tensor
    out_of_support: Tensor
    disagreement: Tensor


class ShiftAwareExpertGate(nn.Module):
    """Criterion-specific expert routing from label-free support diagnostics."""

    def __init__(
        self,
        num_criteria: int,
        num_experts: int,
        *,
        abstention_threshold: float = 0.25,
        support_penalty_init: float = 2.0,
        disagreement_penalty_init: float = 0.25,
    ) -> None:
        super().__init__()
        if num_criteria < 1 or num_experts < 2:
            raise ValueError("The gate needs at least one criterion and two experts.")
        if not 0.0 < abstention_threshold < 1.0:
            raise ValueError("abstention_threshold must be between zero and one.")
        self.num_criteria = int(num_criteria)
        self.num_experts = int(num_experts)
        self.abstention_threshold = float(abstention_threshold)
        self.expert_prior = nn.Parameter(torch.zeros(num_criteria, num_experts))
        self.raw_support_penalty = nn.Parameter(
            torch.full(
                (num_criteria, num_experts),
                self._inverse_softplus(float(support_penalty_init)),
            )
        )
        self.raw_disagreement_penalty = nn.Parameter(
            torch.full(
                (num_criteria, num_experts),
                self._inverse_softplus(float(disagreement_penalty_init)),
            )
        )

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0:
            raise ValueError("Penalty initializers must be positive.")
        return float(np.log(np.expm1(value)))

    def forward(
        self,
        expert_scores: Tensor,
        support_percentiles: Tensor,
        expert_quality: Tensor | None = None,
    ) -> ShiftGateOutput:
        expected = (*expert_scores.shape[:-1], self.num_experts)
        if expert_scores.ndim != 3 or expert_scores.shape[1:] != (
            self.num_criteria,
            self.num_experts,
        ):
            raise ValueError(
                "expert_scores must have shape [batch, num_criteria, num_experts]."
            )
        if support_percentiles.shape != expected:
            raise ValueError("support_percentiles must align with expert_scores.")
        if torch.any(~torch.isfinite(expert_scores)):
            raise ValueError("expert_scores must be finite.")
        if torch.any(~torch.isfinite(support_percentiles)) or torch.any(
            support_percentiles < 0
        ):
            raise ValueError("support_percentiles must be finite and nonnegative.")
        consensus = expert_scores.mean(dim=-1, keepdim=True)
        disagreement = torch.abs(expert_scores - consensus)
        support_penalty = F.softplus(self.raw_support_penalty)
        disagreement_penalty = F.softplus(self.raw_disagreement_penalty)
        routing_logits = (
            self.expert_prior.unsqueeze(0)
            - support_penalty.unsqueeze(0) * support_percentiles
            - disagreement_penalty.unsqueeze(0) * disagreement
        )
        if expert_quality is not None:
            if expert_quality.shape != expected:
                raise ValueError("expert_quality must align with expert_scores.")
            if torch.any(~torch.isfinite(expert_quality)) or torch.any(expert_quality <= 0):
                raise ValueError("expert_quality must be finite and strictly positive.")
            routing_logits = routing_logits + torch.log(expert_quality.clamp_max(1.0))
        weights = torch.softmax(routing_logits, dim=-1)
        support_score = torch.sum(weights * torch.exp(-support_percentiles), dim=-1)
        out_of_support = support_score < self.abstention_threshold
        return ShiftGateOutput(
            weights=weights,
            support_score=support_score,
            out_of_support=out_of_support,
            disagreement=disagreement,
        )


@dataclass(frozen=True)
class SparseAnchorGateOutput:
    weights: Tensor
    correction_strengths: Tensor
    support_score: Tensor
    out_of_support: Tensor
    residuals: Tensor


class SparseAnchorCorrectionGate(nn.Module):
    """Conservative expert routing with a mandatory sparse safety anchor.

    Expert zero is the sparse orthographic–lexical anchor. Semantic and
    measurement experts contribute bounded residual corrections only when both
    their representation support and their anchor-relative correction magnitude
    resemble source holdout episodes. This prevents two correlated dense experts
    from outvoting the sparse path merely because they fail in the same direction.
    """

    def __init__(
        self,
        num_criteria: int,
        num_corrections: int,
        *,
        abstention_threshold: float = 0.25,
        safety_decay: float = 2.0,
        correction_prior_init: float = -0.5,
        support_penalty_init: float = 1.0,
        residual_penalty_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_criteria < 1 or num_corrections < 1:
            raise ValueError("The anchor gate needs criteria and at least one correction expert.")
        if not 0.0 < abstention_threshold < 1.0:
            raise ValueError("abstention_threshold must be between zero and one.")
        if safety_decay <= 0:
            raise ValueError("safety_decay must be positive.")
        self.num_criteria = int(num_criteria)
        self.num_corrections = int(num_corrections)
        self.num_experts = self.num_corrections + 1
        self.abstention_threshold = float(abstention_threshold)
        self.safety_decay = float(safety_decay)
        self.correction_prior = nn.Parameter(
            torch.full((num_criteria, num_corrections), float(correction_prior_init))
        )
        self.raw_support_penalty = nn.Parameter(
            torch.full(
                (num_criteria, num_corrections),
                ShiftAwareExpertGate._inverse_softplus(float(support_penalty_init)),
            )
        )
        self.raw_residual_penalty = nn.Parameter(
            torch.full(
                (num_criteria, num_corrections),
                ShiftAwareExpertGate._inverse_softplus(float(residual_penalty_init)),
            )
        )

    def forward(
        self,
        expert_scores: Tensor,
        support_percentiles: Tensor,
        residual_percentiles: Tensor,
        correction_quality: Tensor | None = None,
    ) -> SparseAnchorGateOutput:
        if expert_scores.ndim != 3 or expert_scores.shape[1:] != (
            self.num_criteria,
            self.num_experts,
        ):
            raise ValueError(
                "expert_scores must have shape [batch, num_criteria, 1 + num_corrections]."
            )
        if support_percentiles.shape != expert_scores.shape:
            raise ValueError("support_percentiles must align with expert_scores.")
        expected_residual = (
            expert_scores.shape[0],
            self.num_criteria,
            self.num_corrections,
        )
        if residual_percentiles.shape != expected_residual:
            raise ValueError(
                "residual_percentiles must have shape [batch, num_criteria, num_corrections]."
            )
        for name, value in [
            ("expert_scores", expert_scores),
            ("support_percentiles", support_percentiles),
            ("residual_percentiles", residual_percentiles),
        ]:
            if torch.any(~torch.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
        if torch.any(support_percentiles < 0) or torch.any(residual_percentiles < 0):
            raise ValueError("Support and residual percentiles must be nonnegative.")

        anchor = expert_scores[..., :1]
        residuals = expert_scores[..., 1:] - anchor
        correction_support = support_percentiles[..., 1:]
        learned_logit = (
            self.correction_prior.unsqueeze(0)
            - F.softplus(self.raw_support_penalty).unsqueeze(0) * correction_support
            - F.softplus(self.raw_residual_penalty).unsqueeze(0) * residual_percentiles
        )
        learned_strength = torch.sigmoid(learned_logit)
        safety_envelope = torch.exp(
            -self.safety_decay * (correction_support + residual_percentiles)
        )
        correction_strengths = learned_strength * safety_envelope
        if correction_quality is not None:
            if correction_quality.shape != expected_residual:
                raise ValueError("correction_quality must align with correction experts.")
            if torch.any(~torch.isfinite(correction_quality)) or torch.any(correction_quality <= 0):
                raise ValueError("correction_quality must be finite and strictly positive.")
            correction_strengths = correction_strengths * correction_quality.clamp_max(1.0)

        unnormalized = torch.cat(
            [torch.ones_like(anchor), correction_strengths],
            dim=-1,
        )
        weights = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
        support_score = torch.sum(weights * torch.exp(-support_percentiles), dim=-1)
        out_of_support = support_score < self.abstention_threshold
        return SparseAnchorGateOutput(
            weights=weights,
            correction_strengths=correction_strengths,
            support_score=support_score,
            out_of_support=out_of_support,
            residuals=residuals,
        )


@dataclass(frozen=True)
class TrustRegionFusionOutput:
    raw_logits: Tensor
    projected_logits: Tensor
    bounded_residuals: Tensor
    applied_corrections: Tensor


class SparseAnchorTrustRegionFusion(nn.Module):
    """Apply bounded semantic/measurement corrections to sparse ordinal logits."""

    def __init__(self, trust_radii: Tensor) -> None:
        super().__init__()
        radii = torch.as_tensor(trust_radii, dtype=torch.float32)
        if radii.ndim != 3:
            raise ValueError("trust_radii must have shape [criteria, corrections, thresholds].")
        if torch.any(~torch.isfinite(radii)) or torch.any(radii <= 0):
            raise ValueError("trust_radii must be finite and strictly positive.")
        self.register_buffer("trust_radii", radii, persistent=True)
        self.num_criteria, self.num_corrections, self.num_thresholds = radii.shape
        chains = [
            list(range(criterion * self.num_thresholds, (criterion + 1) * self.num_thresholds))
            for criterion in range(self.num_criteria)
        ]
        self.projection = GraphIsotonicProjection(chains)

    def forward(
        self,
        anchor_logits: Tensor,
        correction_logits: Tensor,
        correction_strengths: Tensor,
    ) -> TrustRegionFusionOutput:
        if anchor_logits.ndim != 3 or anchor_logits.shape[1:] != (
            self.num_criteria,
            self.num_thresholds,
        ):
            raise ValueError("anchor_logits must have shape [batch, criteria, thresholds].")
        expected_corrections = (
            anchor_logits.shape[0],
            self.num_criteria,
            self.num_corrections,
            self.num_thresholds,
        )
        if correction_logits.shape != expected_corrections:
            raise ValueError(
                "correction_logits must have shape [batch, criteria, corrections, thresholds]."
            )
        if correction_strengths.shape != expected_corrections[:-1]:
            raise ValueError(
                "correction_strengths must have shape [batch, criteria, corrections]."
            )
        for name, value in [
            ("anchor_logits", anchor_logits),
            ("correction_logits", correction_logits),
            ("correction_strengths", correction_strengths),
        ]:
            if torch.any(~torch.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
        if torch.any(correction_strengths < 0) or torch.any(correction_strengths > 1):
            raise ValueError("correction_strengths must lie in [0, 1].")
        residuals = correction_logits - anchor_logits.unsqueeze(2)
        radii = self.trust_radii.unsqueeze(0).to(
            device=residuals.device, dtype=residuals.dtype
        )
        bounded = radii * torch.tanh(residuals / radii)
        applied = correction_strengths.unsqueeze(-1) * bounded
        raw = anchor_logits + applied.sum(dim=2)
        projected = self.projection(raw.reshape(raw.shape[0], -1)).reshape_as(raw)
        return TrustRegionFusionOutput(
            raw_logits=raw,
            projected_logits=projected,
            bounded_residuals=bounded,
            applied_corrections=applied,
        )


@dataclass(frozen=True)
class DecisionSafeFusionOutput:
    raw_logits: Tensor
    projected_logits: Tensor
    proposed_correction: Tensor
    applied_correction: Tensor
    anchor_supported: Tensor


class DecisionSafeSparseAnchorFusion(nn.Module):
    """Preserve supported sparse decisions while allowing bounded calibration."""

    def __init__(self, trust_radii: Tensor, *, supported_margin_fraction: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= supported_margin_fraction < 1.0:
            raise ValueError("supported_margin_fraction must lie in [0, 1).")
        self.trust_fusion = SparseAnchorTrustRegionFusion(trust_radii)
        self.supported_margin_fraction = float(supported_margin_fraction)

    def forward(
        self,
        anchor_logits: Tensor,
        correction_logits: Tensor,
        correction_strengths: Tensor,
        anchor_supported: Tensor,
    ) -> DecisionSafeFusionOutput:
        if anchor_supported.shape != anchor_logits.shape[:-1]:
            raise ValueError("anchor_supported must have shape [batch, criteria].")
        if anchor_supported.dtype != torch.bool:
            raise ValueError("anchor_supported must be Boolean.")
        trust = self.trust_fusion(anchor_logits, correction_logits, correction_strengths)
        proposed = trust.applied_corrections.sum(dim=2)
        budget = self.supported_margin_fraction * torch.abs(anchor_logits)
        safe = torch.maximum(torch.minimum(proposed, budget), -budget)
        applied = torch.where(anchor_supported.unsqueeze(-1), safe, proposed)
        raw = anchor_logits + applied
        projected = self.trust_fusion.projection(raw.reshape(raw.shape[0], -1)).reshape_as(raw)
        return DecisionSafeFusionOutput(
            raw_logits=raw,
            projected_logits=projected,
            proposed_correction=proposed,
            applied_correction=applied,
            anchor_supported=anchor_supported,
        )


@dataclass(frozen=True)
class MultiAnchorDecisionContractOutput:
    """Result of projecting a proposal through a consensus decision contract."""

    proposed_logits: Tensor
    contracted_logits: Tensor
    qualified_anchor_count: Tensor
    qualified_anchor_consensus: Tensor
    qualified_anchor_conflict: Tensor
    contract_changed_logits: Tensor
    protected_grade: Tensor


class MultiAnchorDecisionContract(nn.Module):
    """Project an ordinal proposal onto an explicit anchor-consensus region.

    ``activation="qualified_subset"`` retains the original behavior: any
    non-empty set of qualified anchors can activate the contract when that
    subset agrees.  ``activation="support_backed_full_consensus"`` is the
    stricter RA-SCOPE rule: every anchor must predict the same point grade and
    at least one anchor must be support-qualified.  The stricter rule never
    turns a singleton supported decision into a multi-anchor consensus.

    For an active item, coordinate-wise clipping is the unique Euclidean
    projection of the pooled logits onto the monotone box induced by retained
    anchor margins.  Labels are neither consumed nor required at inference.
    """

    def __init__(
        self,
        *,
        supported_margin_fraction: float = 0.5,
        activation: str = "qualified_subset",
    ) -> None:
        super().__init__()
        if not 0.0 <= supported_margin_fraction < 1.0:
            raise ValueError("supported_margin_fraction must lie in [0, 1).")
        if activation not in {
            "qualified_subset",
            "support_backed_full_consensus",
        }:
            raise ValueError(
                "activation must be 'qualified_subset' or "
                "'support_backed_full_consensus'."
            )
        self.supported_margin_fraction = float(supported_margin_fraction)
        self.activation = activation

    def forward(
        self,
        proposed_logits: Tensor,
        anchor_logits: Tensor,
        anchor_qualified: Tensor,
    ) -> MultiAnchorDecisionContractOutput:
        if proposed_logits.ndim != 2:
            raise ValueError("proposed_logits must have shape [batch, thresholds].")
        if anchor_logits.ndim != 3 or anchor_logits.shape[0] != proposed_logits.shape[0]:
            raise ValueError(
                "anchor_logits must have shape [batch, anchors, thresholds]."
            )
        if anchor_logits.shape[2] != proposed_logits.shape[1]:
            raise ValueError("Anchor and proposal threshold counts must match.")
        if anchor_qualified.shape != anchor_logits.shape[:2]:
            raise ValueError("anchor_qualified must have shape [batch, anchors].")
        if anchor_qualified.dtype != torch.bool:
            raise ValueError("anchor_qualified must be Boolean.")
        if torch.any(~torch.isfinite(proposed_logits)) or torch.any(
            ~torch.isfinite(anchor_logits)
        ):
            raise ValueError("Proposal and anchor logits must be finite.")
        if torch.any(proposed_logits[:, 1:] > proposed_logits[:, :-1] + 1e-6):
            raise ValueError("Proposed cumulative logits must be nonincreasing.")
        if torch.any(anchor_logits[:, :, 1:] > anchor_logits[:, :, :-1] + 1e-6):
            raise ValueError("Anchor cumulative logits must be nonincreasing.")

        anchor_grades = 1 + (anchor_logits >= 0).sum(dim=-1)
        qualified_count = anchor_qualified.sum(dim=-1)
        sentinel_low = torch.iinfo(anchor_grades.dtype).max
        sentinel_high = torch.iinfo(anchor_grades.dtype).min
        qualified_min = torch.where(
            anchor_qualified,
            anchor_grades,
            torch.full_like(anchor_grades, sentinel_low),
        ).amin(dim=-1)
        qualified_max = torch.where(
            anchor_qualified,
            anchor_grades,
            torch.full_like(anchor_grades, sentinel_high),
        ).amax(dim=-1)
        if self.activation == "support_backed_full_consensus":
            all_min = anchor_grades.amin(dim=-1)
            all_max = anchor_grades.amax(dim=-1)
            consensus = (qualified_count > 0) & (all_min == all_max)
            conflict = (qualified_count > 0) & (all_min != all_max)
            consensus_grade = all_min
            contract_anchor_mask = torch.ones_like(anchor_qualified)
        else:
            consensus = (qualified_count > 0) & (qualified_min == qualified_max)
            conflict = (qualified_count > 1) & (qualified_min != qualified_max)
            consensus_grade = qualified_min
            contract_anchor_mask = anchor_qualified
        protected_grade = torch.where(
            consensus,
            consensus_grade,
            torch.zeros_like(consensus_grade),
        )

        retained_fraction = 1.0 - self.supported_margin_fraction
        retained_anchor = retained_fraction * anchor_logits
        positive = anchor_logits >= 0
        active = contract_anchor_mask.unsqueeze(-1) & consensus[:, None, None]
        lower_candidates = torch.where(
            active & positive,
            retained_anchor,
            torch.full_like(retained_anchor, -torch.inf),
        )
        upper_candidates = torch.where(
            active & ~positive,
            retained_anchor,
            torch.full_like(retained_anchor, torch.inf),
        )
        lower = lower_candidates.amax(dim=1)
        upper = upper_candidates.amin(dim=1)
        contracted = torch.maximum(torch.minimum(proposed_logits, upper), lower)

        # Each proposal and every finite bound is nonincreasing, so clipping is
        # order preserving.  Cummin removes only floating-point edge violations
        # while retaining the feasible sign bounds.
        contracted = torch.cummin(contracted, dim=-1).values
        if torch.any(contracted[:, 1:] > contracted[:, :-1] + 1e-6):
            raise RuntimeError("Contracted cumulative logits are not monotone.")
        if torch.any(consensus & ((1 + (contracted >= 0).sum(dim=-1)) != protected_grade)):
            raise RuntimeError("A qualified consensus decision was not preserved.")

        changed = torch.any(torch.abs(contracted - proposed_logits) > 1e-7, dim=-1)
        return MultiAnchorDecisionContractOutput(
            proposed_logits=proposed_logits,
            contracted_logits=contracted,
            qualified_anchor_count=qualified_count,
            qualified_anchor_consensus=consensus,
            qualified_anchor_conflict=conflict,
            contract_changed_logits=changed,
            protected_grade=protected_grade,
        )


@dataclass(frozen=True)
class ConsensusContractedOrdinalPoolOutput:
    """Predictions and diagnostics from a consensus-contracted ordinal pool."""

    proposed_logits: Tensor
    proposed_cumulative: Tensor
    contracted_logits: Tensor
    contracted_cumulative: Tensor
    point_grades: Tensor
    expected_scores: Tensor
    qualified_anchor_count: Tensor
    qualified_anchor_consensus: Tensor
    qualified_anchor_conflict: Tensor
    contract_changed_logits: Tensor
    protected_grade: Tensor


class ConsensusContractedOrdinalPool(nn.Module):
    """Uniformly pool ordinal experts and preserve supported anchor consensus.

    Each expert supplies probabilities for the cumulative events ``P(Y >= k)``.
    A convex, uniform pool is monotone whenever every expert is monotone.  The
    resulting proposal is then passed through a support-qualified multi-anchor
    decision contract.  Finite logits are constructed only at the contract
    boundary, so exact expert probabilities at zero or one remain available to
    the pool.  The module has no trainable parameters: expert identities, anchor
    identities, and support qualification are explicit in the architecture.
    """

    def __init__(
        self,
        num_experts: int,
        anchor_indices: tuple[int, ...],
        *,
        supported_margin_fraction: float = 0.5,
        activation: str = "qualified_subset",
        probability_epsilon: float = 1e-6,
        expert_logit_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("The ordinal pool needs at least two experts.")
        if not anchor_indices:
            raise ValueError("At least one anchor index is required.")
        normalized_indices = tuple(int(index) for index in anchor_indices)
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("anchor_indices must be unique.")
        if any(index < 0 or index >= num_experts for index in normalized_indices):
            raise ValueError("Every anchor index must identify a pooled expert.")
        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie in (0, 0.5).")
        if not 0.0 < expert_logit_epsilon < 0.5:
            raise ValueError("expert_logit_epsilon must lie in (0, 0.5).")

        self.num_experts = int(num_experts)
        self.probability_epsilon = float(probability_epsilon)
        self.expert_logit_epsilon = float(expert_logit_epsilon)
        self.register_buffer(
            "pool_weights",
            torch.full((self.num_experts,), 1.0 / self.num_experts),
            persistent=True,
        )
        self.register_buffer(
            "anchor_indices",
            torch.tensor(normalized_indices, dtype=torch.long),
            persistent=True,
        )
        self.contract = MultiAnchorDecisionContract(
            supported_margin_fraction=supported_margin_fraction,
            activation=activation,
        )

    def forward(
        self,
        expert_cumulative: Tensor,
        anchor_qualified: Tensor,
    ) -> ConsensusContractedOrdinalPoolOutput:
        if expert_cumulative.ndim != 3 or expert_cumulative.shape[1] != self.num_experts:
            raise ValueError(
                "expert_cumulative must have shape [batch, num_experts, thresholds]."
            )
        if expert_cumulative.shape[2] < 1:
            raise ValueError("At least one ordinal threshold is required.")
        expected_qualification_shape = (
            expert_cumulative.shape[0],
            int(self.anchor_indices.numel()),
        )
        if anchor_qualified.shape != expected_qualification_shape:
            raise ValueError(
                "anchor_qualified must have shape [batch, number_of_anchors]."
            )
        if anchor_qualified.dtype != torch.bool:
            raise ValueError("anchor_qualified must be Boolean.")
        if torch.any(~torch.isfinite(expert_cumulative)):
            raise ValueError("expert_cumulative must be finite.")
        if torch.any(expert_cumulative < 0) or torch.any(expert_cumulative > 1):
            raise ValueError("expert_cumulative must lie in [0, 1].")
        if torch.any(
            expert_cumulative[:, :, 1:] > expert_cumulative[:, :, :-1] + 1e-6
        ):
            raise ValueError("Every expert's cumulative probabilities must be nonincreasing.")

        weights = self.pool_weights.to(
            device=expert_cumulative.device,
            dtype=expert_cumulative.dtype,
        )
        proposed_cumulative = torch.sum(
            expert_cumulative * weights[None, :, None],
            dim=1,
        )
        # Convex pooling is order preserving.  Cummin removes only possible
        # machine-precision violations and matches the frozen reference path.
        proposed_cumulative = torch.cummin(proposed_cumulative, dim=-1).values
        bounded_cumulative = proposed_cumulative.clamp(
            min=self.probability_epsilon,
            max=1.0 - self.probability_epsilon,
        )
        proposed_logits = torch.logit(bounded_cumulative)
        expert_logits = torch.logit(
            expert_cumulative.clamp(
                min=self.expert_logit_epsilon,
                max=1.0 - self.expert_logit_epsilon,
            )
        )
        anchor_logits = expert_logits.index_select(1, self.anchor_indices)
        contracted = self.contract(
            proposed_logits,
            anchor_logits,
            anchor_qualified,
        )
        contracted_cumulative = torch.sigmoid(contracted.contracted_logits)
        point_grades = 1 + (contracted_cumulative >= 0.5).sum(dim=-1)
        expected_scores = 1.0 + contracted_cumulative.sum(dim=-1)
        return ConsensusContractedOrdinalPoolOutput(
            proposed_logits=proposed_logits,
            proposed_cumulative=proposed_cumulative,
            contracted_logits=contracted.contracted_logits,
            contracted_cumulative=contracted_cumulative,
            point_grades=point_grades,
            expected_scores=expected_scores,
            qualified_anchor_count=contracted.qualified_anchor_count,
            qualified_anchor_consensus=contracted.qualified_anchor_consensus,
            qualified_anchor_conflict=contracted.qualified_anchor_conflict,
            contract_changed_logits=contracted.contract_changed_logits,
            protected_grade=contracted.protected_grade,
        )


@dataclass(frozen=True)
class CriterionExpertFusionOutput:
    raw_fused_logits: Tensor
    projected_logits: Tensor
    criterion_scores: Tensor
    gate_weights: Tensor
    support_score: Tensor
    criterion_abstentions: Tensor
    book_abstentions: Tensor
    disagreement: Tensor


class CriterionExpertFusion(nn.Module):
    """Fuse criterion experts and exactly project every ordinal threshold chain."""

    def __init__(
        self,
        num_criteria: int,
        num_experts: int,
        num_levels: int = 6,
        *,
        abstention_threshold: float = 0.25,
        max_unsupported_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be at least two.")
        if not 0.0 <= max_unsupported_fraction <= 1.0:
            raise ValueError("max_unsupported_fraction must be in [0, 1].")
        self.num_criteria = int(num_criteria)
        self.num_experts = int(num_experts)
        self.num_levels = int(num_levels)
        self.num_thresholds = self.num_levels - 1
        self.max_unsupported_fraction = float(max_unsupported_fraction)
        self.gate = ShiftAwareExpertGate(
            num_criteria,
            num_experts,
            abstention_threshold=abstention_threshold,
        )
        chains = [
            list(range(criterion * self.num_thresholds, (criterion + 1) * self.num_thresholds))
            for criterion in range(self.num_criteria)
        ]
        self.projection = GraphIsotonicProjection(chains)

    def forward(
        self,
        expert_cumulative_logits: Tensor,
        support_percentiles: Tensor,
        expert_quality: Tensor | None = None,
    ) -> CriterionExpertFusionOutput:
        expected = (
            self.num_criteria,
            self.num_experts,
            self.num_thresholds,
        )
        if expert_cumulative_logits.ndim != 4 or expert_cumulative_logits.shape[1:] != expected:
            raise ValueError(
                "expert_cumulative_logits must have shape "
                "[batch, num_criteria, num_experts, num_levels-1]."
            )
        if torch.any(~torch.isfinite(expert_cumulative_logits)):
            raise ValueError("expert_cumulative_logits must be finite.")
        expert_scores = 1.0 + torch.sigmoid(expert_cumulative_logits).sum(dim=-1)
        gate_output = self.gate(expert_scores, support_percentiles, expert_quality)
        raw_fused = torch.sum(
            gate_output.weights.unsqueeze(-1) * expert_cumulative_logits,
            dim=2,
        )
        flat = raw_fused.reshape(raw_fused.shape[0], -1)
        projected = self.projection(flat).reshape_as(raw_fused)
        criterion_scores = 1.0 + torch.sigmoid(projected).sum(dim=-1)
        unsupported_fraction = gate_output.out_of_support.to(dtype=projected.dtype).mean(dim=-1)
        book_abstentions = unsupported_fraction > self.max_unsupported_fraction
        return CriterionExpertFusionOutput(
            raw_fused_logits=raw_fused,
            projected_logits=projected,
            criterion_scores=criterion_scores,
            gate_weights=gate_output.weights,
            support_score=gate_output.support_score,
            criterion_abstentions=gate_output.out_of_support,
            book_abstentions=book_abstentions,
            disagreement=gate_output.disagreement,
        )


def episodic_worst_group_loss(
    output: CriterionExpertFusionOutput,
    target_levels: Tensor,
    group_ids: Tensor,
    *,
    entropy_weight: float = 0.01,
) -> Tensor:
    """Worst-group smooth ordinal loss for source holdout gate episodes."""

    if target_levels.shape != output.criterion_scores.shape:
        raise ValueError("target_levels must align with criterion_scores.")
    if group_ids.ndim != 1 or group_ids.shape[0] != target_levels.shape[0]:
        raise ValueError("group_ids must have one value per book.")
    valid = target_levels > 0
    per_cell = F.smooth_l1_loss(
        output.criterion_scores,
        target_levels.to(dtype=output.criterion_scores.dtype),
        reduction="none",
    )
    group_losses: list[Tensor] = []
    for group in torch.unique(group_ids):
        selected = (group_ids == group).unsqueeze(-1) & valid
        if torch.any(selected):
            group_losses.append(per_cell[selected].mean())
    if not group_losses:
        raise ValueError("No valid targets were available for any group.")
    entropy = -torch.sum(
        output.gate_weights * torch.log(output.gate_weights.clamp_min(1e-8)),
        dim=-1,
    ).mean()
    return torch.stack(group_losses).amax() - float(entropy_weight) * entropy
