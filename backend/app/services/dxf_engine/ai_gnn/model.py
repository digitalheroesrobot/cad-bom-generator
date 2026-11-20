from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


def _edge_key(edge_type):
    return "|".join(str(part) for part in edge_type)


class RelationMatchingNetwork(nn.Module):
    """Relation-specific edge-aware message and attention score."""

    def __init__(self, hidden_channels: int, edge_channels: int):
        super().__init__()
        self.source = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.edge = nn.Linear(max(1, edge_channels), hidden_channels, bias=False)
        self.score = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2 or 1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2 or 1, 1),
        )

    def forward(self, source_x, edge_attr):
        if edge_attr is None or edge_attr.size(-1) == 0:
            edge_attr = source_x.new_zeros((source_x.size(0), self.edge.in_features))
        projected_source = self.source(source_x)
        message = torch.tanh(projected_source + self.edge(edge_attr))
        score = self.score(torch.cat([message, projected_source], dim=-1))
        return message, score


class HeterogeneousMessagePassing(nn.Module):
    """Destination-normalized attention across all incoming typed relations.

    Unlike independent sigmoid gates, incoming edges compete in one softmax
    for each destination node. Relation modules use stable numeric keys, so
    punctuation in a relation name cannot create ModuleDict key collisions.
    """

    def __init__(self, node_types, edge_types, hidden_channels, edge_dims, dropout=0.2):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.edge_keys = {edge_type: f"relation_{i}" for i, edge_type in enumerate(self.edge_types)}
        self.messages = nn.ModuleDict()
        for edge_type in self.edge_types:
            width = int(edge_dims.get(_edge_key(edge_type), 1))
            self.messages[self.edge_keys[edge_type]] = RelationMatchingNetwork(hidden_channels, width)
        self.updates = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            ) for node_type in self.node_types
        })
        self.norms = nn.ModuleDict({n: nn.LayerNorm(hidden_channels) for n in self.node_types})
        self.dropout = dropout

    def forward(self, x_dict, data):
        incoming = {n: {"messages": [], "scores": [], "indices": []} for n in x_dict}
        for edge_type in self.edge_types:
            if edge_type not in data.edge_types:
                continue
            store = data[edge_type]
            edge_index = getattr(store, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_type, _, dst_type = edge_type
            src, dst = edge_index[0], edge_index[1]
            edge_attr = getattr(store, "edge_attr", None)
            message, score = self.messages[self.edge_keys[edge_type]](
                x_dict[src_type][src], edge_attr
            )
            incoming[dst_type]["messages"].append(message)
            incoming[dst_type]["scores"].append(score)
            incoming[dst_type]["indices"].append(dst)

        updated = {}
        for node_type, current in x_dict.items():
            bucket = incoming[node_type]
            if bucket["messages"]:
                messages = torch.cat(bucket["messages"], dim=0)
                scores = torch.cat(bucket["scores"], dim=0)
                indices = torch.cat(bucket["indices"], dim=0)
                weights = softmax(scores, indices, num_nodes=current.size(0))
                aggregate = current.new_zeros(current.shape)
                aggregate.index_add_(0, indices, messages * weights)
            else:
                aggregate = current.new_zeros(current.shape)
            delta = self.updates[node_type](torch.cat([current, aggregate], dim=-1))
            updated[node_type] = F.dropout(
                F.gelu(self.norms[node_type](current + delta)),
                p=self.dropout, training=self.training,
            )
        return updated


class GeometricReasoningNetwork(nn.Module):
    """Cross-node-type reasoning after typed message passing.

    Stacked heterogeneous layers explicitly support multi-hop paths such as
    Dimension -> Anchor -> Geometry. This layer then reasons over the fused
    evidence at each node without collapsing relation types prematurely.
    """

    def __init__(self, node_types, hidden_channels, dropout=0.2):
        super().__init__()
        self.reasoners = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LayerNorm(hidden_channels),
            ) for n in node_types
        })
        self.dropout = dropout

    def forward(self, x_dict):
        return {
            n: F.dropout(x + self.reasoners[n](x), p=self.dropout, training=self.training)
            for n, x in x_dict.items()
        }


class DXFSemanticGNN(nn.Module):
    """Relation-aware multi-task DXF GNN.

    Returns class logits, fused log1p measurements, node embeddings, and
    direct/relation/fusion evidence for explainable downstream inference.
    All regression branches are non-negative by construction.
    """

    def __init__(
        self, metadata, input_dims: dict[str, int],
        edge_dims: dict[str, int] | None = None,
        hidden_channels: int = 96, num_classes: int = 8,
        num_layers: int = 3, dropout: float = 0.20,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        edge_dims = edge_dims or {}
        self.input_proj = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(input_dims[n], hidden_channels),
                             nn.LayerNorm(hidden_channels), nn.GELU())
            for n in self.node_types
        })
        self.message_layers = nn.ModuleList([
            HeterogeneousMessagePassing(
                self.node_types, self.edge_types, hidden_channels,
                edge_dims, dropout,
            ) for _ in range(max(2, num_layers))
        ])
        self.relation_reasoner = GeometricReasoningNetwork(
            self.node_types, hidden_channels, dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, num_classes),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, 3),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, 3),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2 or 1),
            nn.GELU(), nn.Linear(hidden_channels // 2 or 1, 3),
        )

    def forward(self, data):
        x_dict = {n: self.input_proj[n](data[n].x) for n in self.node_types}
        # Local geometry-only representation is retained for the direct baseline.
        geometry_local = x_dict["geometry"]
        for layer in self.message_layers:
            x_dict = layer(x_dict, data)
        x_dict = self.relation_reasoner(x_dict)
        geometry_relation = x_dict["geometry"]
        class_logits = self.classifier(geometry_relation)

        # Direct branch sees local geometry only; relation branch sees the
        # Dimension/Anchor/Text/Geometry message-passing result.
        direct_raw = F.softplus(self.direct_head(geometry_local))
        relation_raw = F.softplus(self.relation_head(geometry_relation))
        fusion = torch.sigmoid(self.fusion_head(geometry_relation))
        fused_raw = fusion * direct_raw + (1.0 - fusion) * relation_raw
        evidence = {
            "direct_log": torch.log1p(direct_raw),
            "relation_log": torch.log1p(relation_raw),
            "fusion_weight": fusion,
            "direct_raw": direct_raw,
            "relation_raw": relation_raw,
            "fused_raw": fused_raw,
        }
        return class_logits, torch.log1p(fused_raw), x_dict, evidence
