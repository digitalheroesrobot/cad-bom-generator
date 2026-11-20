from __future__ import annotations

import re
import torch
from torch import nn
import torch.nn.functional as F


def _edge_key(edge_type):
    return "|".join(str(part) for part in edge_type)


def _module_key(edge_type):
    return re.sub(r"[^A-Za-z0-9_]", "_", _edge_key(edge_type))


class RelationMessage(nn.Module):
    """Edge-aware message function for one typed heterogeneous relation."""

    def __init__(self, hidden_channels: int, edge_channels: int):
        super().__init__()
        self.source = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.edge = nn.Linear(max(1, edge_channels), hidden_channels, bias=False)
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2 or 1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2 or 1, 1),
        )

    def forward(self, source_x, edge_attr):
        if edge_attr is None:
            edge_attr = source_x.new_zeros((source_x.size(0), 1))
        elif edge_attr.size(-1) == 0:
            edge_attr = source_x.new_zeros((edge_attr.size(0), 1))
        message = torch.tanh(self.source(source_x) + self.edge(edge_attr))
        weights = torch.sigmoid(self.attention(message))
        return message * weights


class HeterogeneousMessagePassing(nn.Module):
    """Typed, edge-aware message passing across geometry/dimension/anchor/text."""

    def __init__(self, node_types, edge_types, hidden_channels, edge_dims, dropout=0.2):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout
        self.messages = nn.ModuleDict()
        self.updates = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for edge_type in self.edge_types:
            key = _module_key(edge_type)
            edge_width = int(edge_dims.get(_edge_key(edge_type), 1))
            self.messages[key] = RelationMessage(hidden_channels, edge_width)
        for node_type in self.node_types:
            self.updates[node_type] = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.norms[node_type] = nn.LayerNorm(hidden_channels)

    def forward(self, x_dict, data):
        aggregate = {
            node_type: x.new_zeros(x.shape)
            for node_type, x in x_dict.items()
        }
        counts = {
            node_type: x.new_zeros((x.size(0), 1))
            for node_type, x in x_dict.items()
        }

        for edge_type in self.edge_types:
            if edge_type not in data.edge_types:
                continue
            store = data[edge_type]
            edge_index = getattr(store, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_type, _, dst_type = edge_type
            src_index, dst_index = edge_index[0], edge_index[1]
            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is None:
                edge_attr = x_dict[src_type].new_zeros((src_index.numel(), 1))
            key = _module_key(edge_type)
            messages = self.messages[key](
                x_dict[src_type][src_index], edge_attr
            )
            aggregate[dst_type].index_add_(0, dst_index, messages)
            counts[dst_type].index_add_(
                0, dst_index,
                torch.ones((dst_index.numel(), 1), device=dst_index.device,
                           dtype=messages.dtype),
            )

        updated = {}
        for node_type, current in x_dict.items():
            mean_message = aggregate[node_type] / counts[node_type].clamp_min(1.0)
            delta = self.updates[node_type](
                torch.cat([current, mean_message], dim=-1)
            )
            updated[node_type] = F.dropout(
                F.gelu(self.norms[node_type](current + delta)),
                p=self.dropout,
                training=self.training,
            )
        return updated


class RelationReasoningLayer(nn.Module):
    """Second-stage reasoning over the messages already exchanged by relations.

    The per-node MLP learns combinations such as multiple dimensions constraining
    one geometry, dimension chains, and dimension-anchor-geometry evidence.
    """

    def __init__(self, node_types, hidden_channels, dropout=0.2):
        super().__init__()
        self.node_types = list(node_types)
        self.reasoners = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            for node_type in self.node_types
        })
        self.dropout = dropout

    def forward(self, x_dict):
        return {
            node_type: F.dropout(
                x + self.reasoners[node_type](x),
                p=self.dropout,
                training=self.training,
            )
            for node_type, x in x_dict.items()
        }


class DXFSemanticGNN(nn.Module):
    """Multi-task heterogeneous GNN for geometry class and measurement inference.

    The regression head predicts log1p(length), log1p(area), and log1p(radius).
    Existing analytical length/area values are never read by this model.
    """

    def __init__(
        self,
        metadata,
        input_dims: dict[str, int],
        edge_dims: dict[str, int] | None = None,
        hidden_channels: int = 96,
        num_classes: int = 8,
        num_layers: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.dropout = dropout
        edge_dims = edge_dims or {}
        self.input_proj = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(input_dims[node_type], hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )
            for node_type in self.node_types
        })
        self.message_layers = nn.ModuleList([
            HeterogeneousMessagePassing(
                self.node_types, self.edge_types, hidden_channels,
                edge_dims=edge_dims, dropout=dropout,
            )
            for _ in range(max(1, num_layers))
        ])
        self.relation_reasoner = RelationReasoningLayer(
            self.node_types, hidden_channels, dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        self.measurement_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, data):
        x_dict = {
            node_type: self.input_proj[node_type](data[node_type].x)
            for node_type in self.node_types
        }
        for layer in self.message_layers:
            x_dict = layer(x_dict, data)
        x_dict = self.relation_reasoner(x_dict)
        geometry_embedding = x_dict["geometry"]
        class_logits = self.classifier(geometry_embedding)
        # Outputs are log1p-scaled positive measurement values.
        measurement_log = self.measurement_head(geometry_embedding)
        return class_logits, measurement_log, x_dict
