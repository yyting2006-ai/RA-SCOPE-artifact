from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .standard_graph import StandardGraph


def _pava_blocks_nonincreasing(values: Sequence[float]) -> list[tuple[int, int]]:
    """Return active PAVA blocks for a detached one-dimensional sequence."""

    blocks: list[list[float]] = []
    for index, value in enumerate(values):
        blocks.append([float(index), float(index + 1), 1.0, float(value)])
        while len(blocks) >= 2 and blocks[-2][3] < blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[2] + right[2]
            mean = (left[2] * left[3] + right[2] * right[3]) / weight
            blocks.append([left[0], right[1], weight, mean])
    return [(int(block[0]), int(block[1])) for block in blocks]


class GraphIsotonicProjection(nn.Module):
    """Exact chain-wise L2 isotonic projection with active-block gradients.

    The executable standard is a product of disjoint chains. For each sample and
    chain, PAVA determines the active pooling blocks from detached logits. The
    returned tensor is a linear block-average of the original logits, so autograd
    uses the correct piecewise-linear Jacobian almost everywhere.
    """

    def __init__(self, chains: Sequence[Sequence[int]], tolerance: float = 1e-7) -> None:
        super().__init__()
        self.chains = tuple(tuple(int(index) for index in chain) for chain in chains)
        self.tolerance = float(tolerance)
        occupied = [index for chain in self.chains for index in chain]
        if len(occupied) != len(set(occupied)):
            raise ValueError("Projection chains must be disjoint.")
        if any(len(chain) < 2 for chain in self.chains):
            raise ValueError("Every projection chain needs at least two nodes.")

    def forward(self, raw_logits: Tensor) -> Tensor:
        if raw_logits.ndim < 2:
            raise ValueError("raw_logits must have a batch axis and a node axis.")
        original_shape = raw_logits.shape
        flat = raw_logits.reshape(-1, original_shape[-1])
        projected = flat

        for chain in self.chains:
            index = torch.as_tensor(chain, device=flat.device, dtype=torch.long)
            chain_values = flat.index_select(1, index)
            batch_size, chain_length = chain_values.shape
            matrices = torch.zeros(
                batch_size,
                chain_length,
                chain_length,
                device=flat.device,
                dtype=flat.dtype,
            )
            detached = chain_values.detach().to(device="cpu", dtype=torch.float64)
            for batch_index in range(batch_size):
                blocks = _pava_blocks_nonincreasing(detached[batch_index].tolist())
                for start, end in blocks:
                    matrices[batch_index, start:end, start:end] = 1.0 / (end - start)
            chain_projected = torch.bmm(matrices, chain_values.unsqueeze(-1)).squeeze(-1)
            projected = projected.scatter(
                1,
                index.unsqueeze(0).expand(batch_size, -1),
                chain_projected,
            )
        return projected.reshape(original_shape)

    def max_violation(self, values: Tensor) -> Tensor:
        violations = []
        for chain in self.chains:
            index = torch.as_tensor(chain, device=values.device, dtype=torch.long)
            chain_values = values.index_select(-1, index)
            violations.append((chain_values[..., 1:] - chain_values[..., :-1]).amax())
        if not violations:
            return values.new_zeros(())
        return torch.stack(violations).amax().clamp_min(0)


class MultimodalPageEncoder(nn.Module):
    """Encode page-level text, image, layout, and phonetic streams separately."""

    MODALITY_ORDER = ("text", "vision", "layout", "phonetic")

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        *,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_pages: int,
    ) -> None:
        super().__init__()
        missing = set(self.MODALITY_ORDER) - set(modality_dims)
        if missing:
            raise ValueError(f"Missing modality dimensions: {sorted(missing)}")
        self.modality_dims = dict(modality_dims)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.modality_dims[name], model_dim),
                    nn.LayerNorm(model_dim),
                    nn.GELU(),
                )
                for name in self.MODALITY_ORDER
            }
        )
        self.position = nn.Embedding(max_pages, model_dim)
        self.modality_embedding = nn.Embedding(len(self.MODALITY_ORDER), model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.max_pages = int(max_pages)

    def forward(
        self,
        modality_features: Mapping[str, Tensor],
        page_mask: Tensor,
        modality_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if page_mask.dtype != torch.bool or page_mask.ndim != 2:
            raise ValueError("page_mask must be a Boolean [batch, pages] tensor.")
        batch_size, pages = page_mask.shape
        if pages > self.max_pages:
            raise ValueError(f"Received {pages} pages; max_pages={self.max_pages}.")

        reference = next((value for value in modality_features.values() if value is not None), None)
        if reference is None:
            raise ValueError("At least one modality tensor is required.")
        if reference.shape[:2] != (batch_size, pages):
            raise ValueError("Modality features do not match page_mask shape.")

        projected_streams: list[Tensor] = []
        provided_presence: list[Tensor] = []
        for name in self.MODALITY_ORDER:
            value = modality_features.get(name)
            if value is None:
                value = reference.new_zeros(batch_size, pages, self.modality_dims[name])
                present = page_mask.new_zeros(batch_size, pages)
            else:
                if value.shape[:2] != (batch_size, pages):
                    raise ValueError(f"{name} features do not match page_mask shape.")
                if value.shape[-1] != self.modality_dims[name]:
                    raise ValueError(
                        f"{name} feature dim is {value.shape[-1]}; "
                        f"expected {self.modality_dims[name]}."
                    )
                present = page_mask.clone()
            projected_streams.append(self.projections[name](value))
            provided_presence.append(present)

        streams = torch.stack(projected_streams, dim=2)
        provided = torch.stack(provided_presence, dim=-1)
        if modality_mask is None:
            presence = provided
        else:
            if modality_mask.shape != (batch_size, pages, len(self.MODALITY_ORDER)):
                raise ValueError("modality_mask must have shape [batch, pages, 4].")
            presence = modality_mask.to(dtype=torch.bool) & page_mask.unsqueeze(-1) & provided
        if torch.any(page_mask & ~presence.any(dim=-1)):
            raise ValueError("Every valid page must have at least one present modality.")

        position_ids = torch.arange(pages, device=streams.device)
        modality_ids = torch.arange(len(self.MODALITY_ORDER), device=streams.device)
        streams = (
            streams
            + self.position(position_ids)[None, :, None, :]
            + self.modality_embedding(modality_ids)[None, None, :, :]
        )
        streams = torch.where(presence.unsqueeze(-1), streams, torch.zeros_like(streams))

        # The same sequence encoder is shared across modalities. Fully missing
        # modality sequences receive a temporary safe token and are zeroed later.
        flat_streams = streams.permute(0, 2, 1, 3).reshape(
            batch_size * len(self.MODALITY_ORDER), pages, -1
        )
        flat_presence = presence.permute(0, 2, 1).reshape(
            batch_size * len(self.MODALITY_ORDER), pages
        )
        safe_presence = flat_presence.clone()
        fully_missing = ~safe_presence.any(dim=-1)
        safe_presence[fully_missing, 0] = True
        encoded = self.sequence_encoder(flat_streams, src_key_padding_mask=~safe_presence)
        encoded = self.output_norm(encoded)
        encoded = torch.where(flat_presence.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        encoded = encoded.reshape(batch_size, len(self.MODALITY_ORDER), pages, -1)
        encoded = encoded.permute(0, 2, 1, 3).contiguous()
        return encoded, presence


class StandardConditionedEvidence(nn.Module):
    """Route modalities and page evidence with standard-node descriptors."""

    def __init__(
        self,
        *,
        standard_feature_dim: int,
        model_dim: int,
        num_modalities: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(standard_feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.modality_gate = nn.Linear(model_dim, num_modalities)
        self.support_head = nn.Sequential(
            nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, 1),
        )

    def forward(
        self,
        page_modality_states: Tensor,
        modality_presence: Tensor,
        page_mask: Tensor,
        standard_node_features: Tensor,
        node_modality_prior: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if standard_node_features.ndim != 2:
            raise ValueError("standard_node_features must have shape [nodes, feature_dim].")
        if page_modality_states.ndim != 4:
            raise ValueError(
                "page_modality_states must have shape [batch, pages, modalities, model_dim]."
            )
        if modality_presence.shape != page_modality_states.shape[:3]:
            raise ValueError("modality_presence shape does not match page modality states.")
        batch_size, pages, num_modalities, model_dim = page_modality_states.shape
        node_states = self.node_projection(standard_node_features)
        if node_modality_prior.shape != (node_states.shape[0], num_modalities):
            raise ValueError("node_modality_prior has the wrong shape.")

        compatibility = torch.einsum(
            "nd,bpmd->bnpm",
            node_states,
            page_modality_states,
        ) / math.sqrt(model_dim)
        routing_logits = compatibility + self.modality_gate(node_states)[None, :, None, :]
        available = modality_presence[:, None, :, :]
        allowed = node_modality_prior[None, :, None, :].to(dtype=torch.bool)
        valid_routes = available & allowed
        # A missing expected modality should not produce NaNs. The fallback is
        # explicitly detectable because it can only occur when no allowed stream
        # is present on that page.
        fallback = ~valid_routes.any(dim=-1, keepdim=True)
        valid_routes = torch.where(fallback, available, valid_routes)
        routing_logits = routing_logits.masked_fill(
            ~valid_routes,
            torch.finfo(routing_logits.dtype).min,
        )
        routing_weights = torch.softmax(routing_logits, dim=-1)
        routing_weights = torch.where(valid_routes, routing_weights, torch.zeros_like(routing_weights))
        node_page_states = torch.einsum(
            "bnpm,bpmd->bnpd",
            routing_weights,
            page_modality_states,
        )

        page_logits = torch.einsum("nd,bnpd->bnp", node_states, node_page_states)
        page_logits = page_logits / math.sqrt(model_dim)
        page_logits = page_logits.masked_fill(
            ~page_mask[:, None, :],
            torch.finfo(page_logits.dtype).min,
        )
        attention = torch.softmax(page_logits, dim=-1)
        attention = torch.where(
            page_mask[:, None, :],
            attention,
            torch.zeros_like(attention),
        )
        context = torch.einsum("bnp,bnpd->bnd", attention, node_page_states)
        queries = node_states.unsqueeze(0).expand(batch_size, -1, -1)
        joint = torch.cat(
            [queries, context, queries * context, torch.abs(queries - context)],
            dim=-1,
        )
        raw_support = self.support_head(joint).squeeze(-1)
        criterion_modality_weights = torch.einsum(
            "bnp,bnpm->bnm",
            attention,
            routing_weights,
        )
        return raw_support, attention, node_states, criterion_modality_weights


class MonotoneMinMaxAggregator(nn.Module):
    """Interaction-rich monotone map from criterion severities to one score."""

    def __init__(self, num_criteria: int, *, groups: int = 8, pieces: int = 4) -> None:
        super().__init__()
        if num_criteria < 1 or groups < 1 or pieces < 1:
            raise ValueError("num_criteria, groups, and pieces must be positive.")
        self.raw_weight = nn.Parameter(torch.empty(groups, pieces, num_criteria))
        self.bias = nn.Parameter(torch.zeros(groups, pieces))
        nn.init.normal_(self.raw_weight, mean=-1.0, std=0.2)
        self.num_criteria = int(num_criteria)

    def forward(self, criterion_scores: Tensor) -> Tensor:
        if criterion_scores.shape[-1] != self.num_criteria:
            raise ValueError("criterion_scores has the wrong final dimension.")
        positive_weight = F.softplus(self.raw_weight) / self.num_criteria
        affine = torch.einsum("...c,gpc->...gp", criterion_scores, positive_weight)
        affine = affine + self.bias
        lower_envelopes = affine.amin(dim=-1)
        return lower_envelopes.amax(dim=-1)

    def minimum_weight(self) -> Tensor:
        return F.softplus(self.raw_weight).amin()


class OrderedCutpoints(nn.Module):
    def __init__(self, num_levels: int) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2.")
        self.base = nn.Parameter(torch.tensor(-0.75))
        self.raw_gaps = nn.Parameter(torch.zeros(num_levels - 2))
        self.num_levels = int(num_levels)

    def forward(self) -> Tensor:
        if self.num_levels == 2:
            return self.base.unsqueeze(0)
        gaps = F.softplus(self.raw_gaps) + 1e-4
        return torch.cat([self.base.unsqueeze(0), self.base + torch.cumsum(gaps, dim=0)])


@dataclass
class ScopeOutput:
    ordinal_logits: Tensor
    global_score: Tensor
    raw_node_logits: Tensor
    projected_node_logits: Tensor
    criterion_scores: Tensor
    evidence_attention: Tensor
    modality_weights: Tensor
    cutpoints: Tensor

    @property
    def predicted_level(self) -> Tensor:
        return 1 + (self.ordinal_logits >= 0).sum(dim=-1)


class ScopeModel(nn.Module):
    """Standard-conditioned ordered partial-order encoder."""

    def __init__(
        self,
        graph: StandardGraph,
        *,
        modality_dims: Mapping[str, int],
        standard_feature_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        page_layers: int = 2,
        dropout: float = 0.1,
        max_pages: int = 64,
        minmax_groups: int = 8,
        minmax_pieces: int = 4,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.page_encoder = MultimodalPageEncoder(
            modality_dims,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=page_layers,
            dropout=dropout,
            max_pages=max_pages,
        )
        self.evidence_router = StandardConditionedEvidence(
            standard_feature_dim=standard_feature_dim,
            model_dim=model_dim,
            num_modalities=len(MultimodalPageEncoder.MODALITY_ORDER),
            dropout=dropout,
        )
        self.projection = GraphIsotonicProjection(graph.chain_indices())
        self.aggregator = MonotoneMinMaxAggregator(
            graph.num_criteria,
            groups=minmax_groups,
            pieces=minmax_pieces,
        )
        self.cutpoints = OrderedCutpoints(graph.level_count)
        modality_index = {
            name: index for index, name in enumerate(MultimodalPageEncoder.MODALITY_ORDER)
        }
        node_modality_prior = torch.zeros(
            graph.num_nodes,
            len(MultimodalPageEncoder.MODALITY_ORDER),
            dtype=torch.bool,
        )
        for node_index, node in enumerate(graph.nodes):
            for modality in node.modalities:
                node_modality_prior[node_index, modality_index[modality]] = True
        self.register_buffer("node_modality_prior", node_modality_prior, persistent=True)

    def node_logits_to_criterion_scores(self, node_logits: Tensor) -> Tensor:
        probabilities = torch.sigmoid(node_logits)
        scores = [
            probabilities.index_select(
                -1,
                torch.as_tensor(chain, device=probabilities.device, dtype=torch.long),
            ).mean(dim=-1)
            for chain in self.graph.chain_indices()
        ]
        return torch.stack(scores, dim=-1)

    def forward(
        self,
        modality_features: Mapping[str, Tensor],
        page_mask: Tensor,
        standard_node_features: Tensor,
        modality_mask: Tensor | None = None,
    ) -> ScopeOutput:
        if standard_node_features.shape[0] != self.graph.num_nodes:
            raise ValueError(
                f"Received {standard_node_features.shape[0]} standard nodes; "
                f"expected {self.graph.num_nodes}."
            )
        page_modalities, modality_presence = self.page_encoder(
            modality_features,
            page_mask,
            modality_mask,
        )
        raw_logits, evidence_attention, _, modality_weights = self.evidence_router(
            page_modalities,
            modality_presence,
            page_mask,
            standard_node_features,
            self.node_modality_prior,
        )
        projected_logits = self.projection(raw_logits)
        criterion_scores = self.node_logits_to_criterion_scores(projected_logits)
        global_score = self.aggregator(criterion_scores)
        cutpoints = self.cutpoints()
        ordinal_logits = global_score.unsqueeze(-1) - cutpoints
        return ScopeOutput(
            ordinal_logits=ordinal_logits,
            global_score=global_score,
            raw_node_logits=raw_logits,
            projected_node_logits=projected_logits,
            criterion_scores=criterion_scores,
            evidence_attention=evidence_attention,
            modality_weights=modality_weights,
            cutpoints=cutpoints,
        )
