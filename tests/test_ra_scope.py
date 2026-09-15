from __future__ import annotations

import numpy as np
import pytest
import torch

from scope_coling.ra_scope import (
    ConsensusContractedOrdinalPool,
    CriterionExpertFusion,
    DecisionSafeSparseAnchorFusion,
    EmpiricalSupportCalibrator,
    MultiAnchorDecisionContract,
    SparseAnchorCorrectionGate,
    SparseAnchorTrustRegionFusion,
    ShiftAwareExpertGate,
    episodic_worst_group_loss,
)


def test_empirical_support_calibrator_is_label_free_and_monotone() -> None:
    source = np.asarray(
        [
            [[0.1, 0.4], [0.2, 0.8]],
            [[0.2, 0.5], [0.4, 0.9]],
            [[0.3, 0.6], [0.6, 1.0]],
        ]
    )
    calibrator = EmpiricalSupportCalibrator().fit(source)
    values = calibrator.transform(np.asarray([[[0.15, 0.55], [0.5, 1.2]]]))
    assert values.shape == (1, 2, 2)
    assert np.all((values >= 0.0) & (values <= 1.0))
    farther = calibrator.transform(np.asarray([[[0.35, 0.65], [0.7, 1.3]]]))
    assert np.all(farther >= values)


def test_shift_gate_downweights_an_unsupported_expert() -> None:
    gate = ShiftAwareExpertGate(1, 3, abstention_threshold=0.2)
    scores = torch.tensor([[[4.0, 4.0, 4.0]]])
    support = torch.tensor([[[0.05, 0.5, 1.0]]])
    output = gate(scores, support)
    assert output.weights.shape == (1, 1, 3)
    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones(1, 1))
    assert output.weights[0, 0, 0] > output.weights[0, 0, 1] > output.weights[0, 0, 2]
    assert not bool(output.out_of_support.item())


def test_gate_abstains_when_every_expert_is_out_of_support() -> None:
    gate = ShiftAwareExpertGate(2, 2, abstention_threshold=0.25)
    output = gate(torch.full((1, 2, 2), 4.0), torch.full((1, 2, 2), 3.0))
    assert torch.all(output.out_of_support)


def test_sparse_anchor_gate_blocks_correlated_unsupported_corrections() -> None:
    gate = SparseAnchorCorrectionGate(1, 2, safety_decay=2.0)
    scores = torch.tensor([[[4.0, 5.0, 5.0]]])
    support = torch.tensor([[[0.05, 0.6, 0.5]]])
    residual_percentiles = torch.tensor([[[1.0, 1.0]]])
    output = gate(scores, support, residual_percentiles)
    assert output.weights[0, 0, 0] > 0.95
    assert output.weights[0, 0, 0] > output.weights[0, 0, 1]
    assert output.weights[0, 0, 0] > output.weights[0, 0, 2]
    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones(1, 1))


def test_sparse_anchor_gate_allows_supported_small_corrections() -> None:
    gate = SparseAnchorCorrectionGate(1, 2, safety_decay=0.5, correction_prior_init=2.0)
    scores = torch.tensor([[[3.5, 3.8, 3.6]]])
    support = torch.tensor([[[0.05, 0.02, 0.03]]])
    residual_percentiles = torch.tensor([[[0.05, 0.05]]])
    output = gate(scores, support, residual_percentiles)
    assert output.weights[0, 0, 1] > 0.2
    assert output.weights[0, 0, 2] > 0.2


def test_trust_region_bounds_extreme_dense_logits_and_projects_order() -> None:
    fusion = SparseAnchorTrustRegionFusion(torch.full((1, 2, 5), 0.5))
    anchor = torch.tensor([[[1.0, 0.5, 0.0, -0.5, -1.0]]], requires_grad=True)
    corrections = torch.stack(
        [torch.full_like(anchor, 100.0), torch.full_like(anchor, -100.0)], dim=2
    )
    strengths = torch.tensor([[[0.2, 0.1]]])
    output = fusion(anchor, corrections, strengths)
    assert torch.max(torch.abs(output.applied_corrections)) <= 0.1 + 1e-6
    assert torch.all(output.projected_logits[..., 1:] <= output.projected_logits[..., :-1] + 1e-7)
    output.projected_logits.sum().backward()
    assert anchor.grad is not None
    assert torch.all(torch.isfinite(anchor.grad))


def test_decision_safe_fusion_preserves_supported_anchor_threshold_signs() -> None:
    fusion = DecisionSafeSparseAnchorFusion(
        torch.full((1, 2, 5), 2.0), supported_margin_fraction=0.5
    )
    anchor = torch.tensor([[[2.0, 1.0, 0.2, -0.3, -1.0]]])
    corrections = torch.stack(
        [torch.full_like(anchor, -100.0), torch.full_like(anchor, 100.0)], dim=2
    )
    strengths = torch.ones(1, 1, 2)
    output = fusion(
        anchor,
        corrections,
        strengths,
        torch.tensor([[True]]),
    )
    assert torch.equal(output.projected_logits >= 0, anchor >= 0)
    assert torch.all(output.projected_logits[..., 1:] <= output.projected_logits[..., :-1] + 1e-7)


def test_multi_anchor_contract_preserves_qualified_consensus_under_extreme_proposal() -> None:
    contract = MultiAnchorDecisionContract(supported_margin_fraction=0.5)
    anchors = torch.tensor(
        [
            [
                [2.0, 1.2, 0.2, -0.4, -1.0],
                [1.6, 0.8, 0.1, -0.2, -0.9],
            ]
        ]
    )
    proposal = torch.tensor([[100.0, 50.0, 10.0, 5.0, 1.0]])
    output = contract(proposal, anchors, torch.tensor([[True, True]]))
    assert bool(output.qualified_anchor_consensus.item())
    assert not bool(output.qualified_anchor_conflict.item())
    assert int(output.protected_grade.item()) == 4
    assert int(1 + (output.contracted_logits >= 0).sum().item()) == 4
    assert torch.all(
        output.contracted_logits[:, 1:] <= output.contracted_logits[:, :-1] + 1e-7
    )


def test_multi_anchor_contract_exposes_conflicts_without_forcing_a_decision() -> None:
    contract = MultiAnchorDecisionContract()
    anchors = torch.tensor(
        [
            [
                [1.0, 0.4, -0.2, -0.7, -1.1],
                [1.0, 0.6, 0.2, -0.4, -0.9],
            ]
        ]
    )
    proposal = torch.tensor([[0.9, 0.5, 0.1, -0.3, -0.8]])
    output = contract(proposal, anchors, torch.tensor([[True, True]]))
    assert bool(output.qualified_anchor_conflict.item())
    assert not bool(output.qualified_anchor_consensus.item())
    torch.testing.assert_close(output.contracted_logits, proposal)


def test_multi_anchor_contract_uses_the_only_qualified_anchor() -> None:
    contract = MultiAnchorDecisionContract(
        supported_margin_fraction=0.25,
        activation="qualified_subset",
    )
    anchors = torch.tensor(
        [
            [
                [1.0, 0.5, -0.1, -0.5, -1.0],
                [2.0, 1.0, 0.5, -0.1, -0.8],
            ]
        ]
    )
    proposal = torch.tensor([[1.5, 1.0, 0.7, 0.2, -0.2]])
    output = contract(proposal, anchors, torch.tensor([[True, False]]))
    assert int(output.qualified_anchor_count.item()) == 1
    assert int(output.protected_grade.item()) == 3
    assert int(1 + (output.contracted_logits >= 0).sum().item()) == 3


def test_support_backed_full_consensus_requires_anchor_agreement() -> None:
    contract = MultiAnchorDecisionContract(
        supported_margin_fraction=0.5,
        activation="support_backed_full_consensus",
    )
    anchors = torch.tensor(
        [
            [
                [1.2, 0.6, -0.1, -0.5, -1.0],
                [1.4, 0.8, 0.2, -0.4, -0.9],
            ]
        ]
    )
    proposal = torch.tensor([[1.0, 0.7, 0.3, -0.2, -0.8]])
    output = contract(proposal, anchors, torch.tensor([[True, False]]))
    assert not bool(output.qualified_anchor_consensus.item())
    assert bool(output.qualified_anchor_conflict.item())
    torch.testing.assert_close(output.contracted_logits, proposal)


def test_support_backed_full_consensus_uses_all_agreeing_anchors() -> None:
    contract = MultiAnchorDecisionContract(
        supported_margin_fraction=0.5,
        activation="support_backed_full_consensus",
    )
    anchors = torch.tensor(
        [
            [
                [1.2, 0.6, 0.1, -0.5, -1.0],
                [1.4, 0.8, 0.2, -0.4, -0.9],
            ]
        ]
    )
    proposal = torch.tensor([[1.0, 0.7, -0.3, -0.4, -0.8]])
    output = contract(proposal, anchors, torch.tensor([[True, False]]))
    assert int(output.qualified_anchor_count.item()) == 1
    assert bool(output.qualified_anchor_consensus.item())
    assert int(output.protected_grade.item()) == 4
    assert int(1 + (output.contracted_logits >= 0).sum().item()) == 4
    assert float(output.contracted_logits[0, 2]) >= 0.1 - 1e-7


def test_support_backed_full_consensus_is_inactive_without_support() -> None:
    contract = MultiAnchorDecisionContract(
        activation="support_backed_full_consensus"
    )
    anchors = torch.tensor(
        [
            [
                [1.2, 0.6, 0.1, -0.5, -1.0],
                [1.4, 0.8, 0.2, -0.4, -0.9],
            ]
        ]
    )
    proposal = torch.tensor([[1.0, 0.7, -0.3, -0.4, -0.8]])
    output = contract(proposal, anchors, torch.tensor([[False, False]]))
    assert not bool(output.qualified_anchor_consensus.item())
    assert not bool(output.qualified_anchor_conflict.item())
    torch.testing.assert_close(output.contracted_logits, proposal)


def test_consensus_pool_is_parameter_free_monotone_and_differentiable() -> None:
    pool = ConsensusContractedOrdinalPool(4, (0, 1))
    expert_cumulative = torch.tensor(
        [
            [
                [0.90, 0.75, 0.58, 0.35, 0.20],
                [0.88, 0.72, 0.55, 0.32, 0.18],
                [0.94, 0.81, 0.61, 0.40, 0.25],
                [0.84, 0.68, 0.53, 0.29, 0.16],
            ]
        ],
        requires_grad=True,
    )
    output = pool(expert_cumulative, torch.tensor([[True, True]]))
    assert sum(parameter.numel() for parameter in pool.parameters()) == 0
    assert output.point_grades.shape == (1,)
    assert output.expected_scores.shape == (1,)
    assert torch.all(
        output.proposed_cumulative[:, 1:] <= output.proposed_cumulative[:, :-1] + 1e-7
    )
    assert torch.all(
        output.contracted_cumulative[:, 1:] <= output.contracted_cumulative[:, :-1] + 1e-7
    )
    output.expected_scores.sum().backward()
    assert expert_cumulative.grad is not None
    assert torch.all(torch.isfinite(expert_cumulative.grad))


def test_consensus_pool_preserves_anchors_against_extreme_nonanchor_votes() -> None:
    pool = ConsensusContractedOrdinalPool(4, (0, 1), supported_margin_fraction=0.5)
    expert_cumulative = torch.tensor(
        [
            [
                [0.90, 0.65, 0.45, 0.25, 0.10],
                [0.85, 0.60, 0.48, 0.28, 0.12],
                [1.00, 1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ]
        ]
    )
    output = pool(expert_cumulative, torch.tensor([[True, True]]))
    assert bool(output.qualified_anchor_consensus.item())
    assert int(output.protected_grade.item()) == 3
    assert int(output.point_grades.item()) == 3
    assert bool(output.contract_changed_logits.item())


def test_consensus_pool_surfaces_anchor_conflict_and_keeps_proposal() -> None:
    pool = ConsensusContractedOrdinalPool(4, (0, 1))
    expert_cumulative = torch.tensor(
        [
            [
                [0.78, 0.60, 0.45, 0.30, 0.18],
                [0.80, 0.66, 0.54, 0.38, 0.22],
                [0.75, 0.62, 0.46, 0.31, 0.19],
                [0.82, 0.64, 0.43, 0.29, 0.17],
            ]
        ]
    )
    output = pool(expert_cumulative, torch.tensor([[True, True]]))
    assert bool(output.qualified_anchor_conflict.item())
    assert not bool(output.qualified_anchor_consensus.item())
    assert not bool(output.contract_changed_logits.item())
    torch.testing.assert_close(output.contracted_logits, output.proposed_logits)


def test_consensus_pool_rejects_invalid_architecture_and_inputs() -> None:
    with pytest.raises(ValueError):
        ConsensusContractedOrdinalPool(4, (0, 0))
    pool = ConsensusContractedOrdinalPool(4, (0, 1))
    with pytest.raises(ValueError):
        pool(torch.randn(2, 3, 5), torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        pool(torch.randn(2, 4, 5), torch.ones(2, 3, dtype=torch.bool))


def test_criterion_fusion_projects_chains_and_backpropagates() -> None:
    torch.manual_seed(27)
    fusion = CriterionExpertFusion(2, 3, num_levels=6, max_unsupported_fraction=0.5)
    logits = torch.randn(4, 2, 3, 5, requires_grad=True)
    support = torch.rand(4, 2, 3)
    output = fusion(logits, support)
    assert output.projected_logits.shape == (4, 2, 5)
    assert output.gate_weights.shape == (4, 2, 3)
    assert output.criterion_scores.shape == (4, 2)
    assert output.book_abstentions.shape == (4,)
    assert torch.all(output.projected_logits[..., 1:] <= output.projected_logits[..., :-1] + 1e-7)
    loss = episodic_worst_group_loss(
        output,
        torch.randint(1, 7, (4, 2)),
        torch.tensor([0, 0, 1, 1]),
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert fusion.gate.raw_support_penalty.grad is not None


def test_fusion_rejects_misaligned_inputs() -> None:
    fusion = CriterionExpertFusion(2, 3)
    with pytest.raises(ValueError):
        fusion(torch.randn(4, 2, 2, 5), torch.rand(4, 2, 3))
