"""Optional AI/GNN subsystem for the DXF engine.

This package is deliberately separate from the deterministic analytical engine.
PyTorch Geometric is imported lazily by model/training modules.
"""
from .graph_builder import build_semantic_graph, build_semantic_graph_from_analysis
from .labels import (load_handle_labels, attach_labels, load_geometry_targets, attach_geometry_targets)
from .engineering_graph import build_engineering_graph

__all__ = [
    "build_semantic_graph",
    "build_semantic_graph_from_analysis",
    "load_handle_labels",
    "attach_labels",
    "load_geometry_targets",
    "attach_geometry_targets",
    "build_engineering_graph",
]
