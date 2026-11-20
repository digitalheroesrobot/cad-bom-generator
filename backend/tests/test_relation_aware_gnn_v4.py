import math

import torch
from shapely.geometry import LineString, Point

from app.services.dxf_engine.ai_gnn.graph_builder import (
    _geometry_feature,
    _dimension_feature,
)
from app.services.dxf_engine.ai_gnn.labels import attach_geometry_targets
from app.services.dxf_engine.ai_gnn.model import DXFSemanticGNN
from torch_geometric.data import HeteroData


class _NodeStore(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _Graph:
    def __init__(self, geometry):
        self.geometry = _NodeStore(geometry)
        self.graph_metadata = {}

    def __getitem__(self, key):
        if key == "geometry":
            return self.geometry
        raise KeyError(key)


def test_geometry_and_dimension_feature_contract_v4():
    linear = {
        "geometry_2d": LineString([(0, 0), (10, 0)]),
        "geometry_type": "LINEAR_GEOMETRY",
        "source": {"entity_type": "LINE"},
        "is_closed": False,
        "is_valid": True,
        "radius": None,
    }
    circle = {
        "geometry_2d": Point(5, 5).buffer(3),
        "geometry_type": "CLOSED_PROFILE",
        "source": {"entity_type": "CIRCLE"},
        "is_closed": True,
        "is_valid": True,
        "radius": 3.0,
    }
    assert len(_geometry_feature(linear, 1.0, (0.0, 0.0))) == 24
    assert len(_geometry_feature(circle, 1.0, (0.0, 0.0))) == 25

    dimension = {
        "endpoints": [(0, 0), (0, 20)],
        "definition_points": [(2, 0), (2, 20)],
        "dimension_type": "LINEAR",
        "length": 20.0,
        "actual_length": None,
    }
    features = _dimension_feature(dimension, 1.0, (0.0, 0.0))
    assert len(features) == 17
    assert features[-2:] == [1.0, 0.0]
    # Endpoints, not definition-point displacement, define the arrow-line direction.
    assert features[5] == 0.0
    assert features[6] == 20.0


def test_label_masks_follow_geometry_semantics_and_preserve_real_zero():
    geometry = {
        "source_handle": ["L1", "P1", "C1", "A1"],
        "entity_type": ["LINE", "LWPOLYLINE", "CIRCLE", "ARC"],
        "is_closed": torch.tensor([False, True, True, False]),
    }
    graph = _Graph(geometry)
    targets = {
        "L1": {"length": 0.0, "area": 5.0, "radius": 2.0},
        "P1": {"length": 10.0, "area": 0.0, "radius": 3.0},
        "C1": {"length": 18.0, "area": 28.0, "radius": 0.0},
        "A1": {"length": 4.0, "area": 8.0, "radius": 2.0},
    }
    attach_geometry_targets(graph, targets)
    assert graph.geometry["length_mask"].tolist() == [True, True, True, True]
    assert graph.geometry["area_mask"].tolist() == [False, True, True, False]
    assert graph.geometry["radius_mask"].tolist() == [False, False, True, True]
    # A true zero remains a valid target and is encoded as log1p(0) == 0.
    assert graph.geometry["length_mask"][0]
    assert graph.geometry["length_y"][0].item() == 0.0
    assert graph.geometry["radius_mask"][2]
    assert graph.geometry["radius_y"][2].item() == 0.0


def test_non_finite_measurement_labels_are_ignored():
    geometry = {
        "source_handle": ["C1"],
        "entity_type": ["CIRCLE"],
        "is_closed": torch.tensor([True]),
    }
    graph = _Graph(geometry)
    attach_geometry_targets(graph, {"C1": {"length": float("nan"), "area": float("inf"), "radius": 5.0}})
    assert not graph.geometry["length_mask"][0]
    assert not graph.geometry["area_mask"][0]
    assert graph.geometry["radius_mask"][0]



def test_relation_aware_model_outputs_positive_hybrid_measurements():
    data = HeteroData()
    data["geometry"].x = torch.randn(2, 24)
    data["dimension"].x = torch.randn(1, 17)
    data["anchor"].x = torch.randn(1, 12)
    data["text"].x = torch.randn(1, 41)

    edge_dims = {}
    relations = [
        (("dimension", "measures_from", "anchor"), 2, [[0], [0]]),
        (("anchor", "rev_measures_from", "dimension"), 2, [[0], [0]]),
        (("dimension", "describes", "geometry"), 2, [[0], [0]]),
        (("text", "labels", "dimension"), 2, [[0], [0]]),
        (("geometry", "equal_length_candidate", "geometry"), 3, [[0], [1]]),
    ]
    for edge_type, width, indices in relations:
        data[edge_type].edge_index = torch.tensor(indices, dtype=torch.long)
        data[edge_type].edge_attr = torch.randn(1, width)
        edge_dims["|".join(edge_type)] = width

    model = DXFSemanticGNN(
        metadata=data.metadata(),
        input_dims={"geometry": 24, "dimension": 17, "anchor": 12, "text": 41},
        edge_dims=edge_dims,
        hidden_channels=32,
        num_classes=4,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        logits, fused_log, embeddings, evidence = model(data)

    assert logits.shape == (2, 4)
    assert fused_log.shape == (2, 3)
    assert embeddings["geometry"].shape == (2, 32)
    assert torch.all(evidence["direct_raw"] >= 0)
    assert torch.all(evidence["relation_raw"] >= 0)
    assert torch.all(evidence["fused_raw"] >= 0)
    assert torch.all((evidence["fusion_weight"] >= 0) & (evidence["fusion_weight"] <= 1))
