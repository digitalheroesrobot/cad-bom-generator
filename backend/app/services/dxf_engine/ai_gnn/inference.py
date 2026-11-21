from __future__ import annotations

import math
import torch

from .model import DXFSemanticGNN


def _optional_float(values, index):
    try:
        value = float(values[index])
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, IndexError):
        return None


def _edge_candidates(data, edge_type, geometry_index):
    """Return source-node candidates attached to one geometry target."""
    if edge_type not in data.edge_types:
        return []
    store = data[edge_type]
    edge_index = store.edge_index
    edge_attr = getattr(store, "edge_attr", None)
    results = []
    for edge_i in range(edge_index.size(1)):
        source_index = int(edge_index[0, edge_i])
        target_index = int(edge_index[1, edge_i])
        if target_index != geometry_index:
            continue
        attrs = edge_attr[edge_i].detach().cpu().tolist() if edge_attr is not None else []
        results.append((source_index, attrs))
    return results


def _dimension_anchor_details(data, dimension_index, relation):
    edge_type = ("dimension", relation, "anchor")
    if edge_type not in data.edge_types:
        return None
    edge_index = data[edge_type].edge_index
    anchor_ids = data["anchor"].element_id
    coordinates = getattr(data["anchor"], "coordinates", [])
    kinds = getattr(data["anchor"], "anchor_kind", [])
    for edge_i in range(edge_index.size(1)):
        if int(edge_index[0, edge_i]) != dimension_index:
            continue
        anchor_index = int(edge_index[1, edge_i])
        return {
            "anchor_id": str(anchor_ids[anchor_index]),
            "anchor_kind": (
                str(kinds[anchor_index]) if anchor_index < len(kinds) else "UNKNOWN"
            ),
            "coordinates": (
                list(coordinates[anchor_index])
                if anchor_index < len(coordinates) else None
            ),
        }
    return None


def _related_dimensions(data, dimension_index):
    """Expose dimension-dimension candidate edges for each attached dimension."""
    records = []
    handles = data["dimension"].source_handle
    kinds = getattr(data["dimension"], "dimension_type", [])
    values = getattr(data["dimension"], "display_value", [])
    relations = ("near", "parallel", "same_axis", "chain_candidate")
    for relation in relations:
        edge_type = ("dimension", relation, "dimension")
        if edge_type not in data.edge_types:
            continue
        edge_index = data[edge_type].edge_index
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        for edge_i in range(edge_index.size(1)):
            src = int(edge_index[0, edge_i])
            dst = int(edge_index[1, edge_i])
            if src == dimension_index:
                other = dst
            elif dst == dimension_index:
                other = src
            else:
                continue
            attrs = edge_attr[edge_i].detach().cpu().tolist() if edge_attr is not None else []
            records.append({
                "dimension_handle": str(handles[other]),
                "relation": relation,
                "dimension_type": str(kinds[other]) if other < len(kinds) else "UNKNOWN",
                "display_value": _optional_float(values, other),
                "normalized_center_distance": attrs[0] if len(attrs) > 0 else None,
                "angle_difference": attrs[1] if len(attrs) > 1 else None,
                "endpoint_gap": attrs[2] if len(attrs) > 2 else None,
                "axis_gap": attrs[3] if len(attrs) > 3 else None,
                "candidate_confidence": attrs[4] if len(attrs) > 4 else None,
            })
    return records


def predict_geometry(data, checkpoint_path: str, device: str | None = None):
    """Predict class and independent geometry measurements.

    `predicted_*` values come from the learned GNN regression head.
    `analytic_*` values are merely reported for later fusion; they are not
    inputs to the GNN. Dimension and dimension-chain records are candidates,
    not guaranteed learned associations.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if data.graph_metadata.get("feature_version") != checkpoint.get("feature_version"):
        raise ValueError(
            "GNN feature version mismatch; rebuild graph and retrain checkpoint."
        )
    # Relation modules are named by their position in metadata.edge_types.
    # PyG may expose the same edge types in a different order when a new graph
    # is built, so recreate the training-time order from the checkpoint's
    # insertion-ordered edge_dims mapping before constructing the model.
    checkpoint_edge_dims = checkpoint.get(
        "edge_dims", data.graph_metadata.get("edge_feature_dims", {})
    )
    if checkpoint.get("edge_dims"):
        checkpoint_edge_types = [
            tuple(edge_key.split("|")) for edge_key in checkpoint["edge_dims"]
        ]
        if any(len(edge_type) != 3 for edge_type in checkpoint_edge_types):
            raise ValueError(
                "Checkpoint contains an invalid edge type key; expected "
                "'source|relation|target'."
            )
        current_edge_types = set(data.edge_types)
        checkpoint_edge_type_set = set(checkpoint_edge_types)
        if current_edge_types != checkpoint_edge_type_set:
            missing = sorted(checkpoint_edge_type_set - current_edge_types)
            unexpected = sorted(current_edge_types - checkpoint_edge_type_set)
            raise ValueError(
                "GNN relation schema mismatch. "
                f"Missing relations in input graph: {missing}; "
                f"unexpected relations: {unexpected}. "
                "Rebuild the graph with the same graph-builder version used "
                "for this checkpoint, or retrain the model."
            )
        model_metadata = (data.node_types, checkpoint_edge_types)
    else:
        # Compatibility path for older checkpoints that did not store edge_dims.
        model_metadata = data.metadata()

    model = DXFSemanticGNN(
        metadata=model_metadata,
        input_dims=checkpoint["input_dims"],
        edge_dims=checkpoint_edge_dims,
        hidden_channels=checkpoint["hidden_channels"],
        num_classes=max(1, max(checkpoint.get("class_to_id", {}).values(), default=-1) + 1),
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.20),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits, measurement_log, _, evidence = model(data)
        if checkpoint.get("class_to_id"):
            probabilities = logits.softmax(dim=-1)
            scores, class_ids = probabilities.max(dim=-1)
        else:
            scores = torch.zeros(logits.size(0), device=device)
            class_ids = torch.zeros(logits.size(0), dtype=torch.long, device=device)

    id_to_class = {v: k for k, v in checkpoint.get("class_to_id", {}).items()}
    geometry_handles = data["geometry"].source_handle
    dimension_handles = data["dimension"].source_handle
    text_handles = data["text"].source_handle
    dimension_types = getattr(data["dimension"], "dimension_type", [])
    dimension_values = getattr(data["dimension"], "display_value", [])
    dimension_actual_values = getattr(data["dimension"], "actual_value", [])
    text_contents = getattr(data["text"], "content", [])
    text_normalized = getattr(data["text"], "normalized_content", text_contents)
    text_semantics = getattr(data["text"], "semantic_type", ["UNKNOWN"] * len(text_handles))
    text_diameter_values = getattr(data["text"], "diameter_value", [])
    text_radius_values = getattr(data["text"], "radius_value", [])
    analytic_lengths = getattr(data["geometry"], "measured_length", [])
    analytic_areas = getattr(data["geometry"], "measured_area", [])
    analytic_radii = getattr(data["geometry"], "radius_measured", [])
    raw_dxf_radii = getattr(data["geometry"], "radius", [])
    entity_types = getattr(data["geometry"], "entity_type", ["OTHER"] * len(geometry_handles))
    closed_flags = getattr(data["geometry"], "is_closed", [False] * len(geometry_handles))
    length_valid = getattr(data["geometry"], "measured_length_valid", [False] * len(geometry_handles))
    area_valid = getattr(data["geometry"], "measured_area_valid", [False] * len(geometry_handles))
    radius_valid = getattr(data["geometry"], "radius_valid", [False] * len(geometry_handles))

    predictions = []
    for geometry_index in range(len(class_ids)):
        dimension_candidates = []
        for relation in ("measures_from", "measures_to", "describes"):
            edge_type = ("dimension", relation, "geometry")
            for dimension_index, attrs in _edge_candidates(data, edge_type, geometry_index):
                dimension_text_matches = []
                for text_index, text_attrs in _edge_candidates(
                    data, ("text", "labels", "dimension"), dimension_index
                ):
                    dimension_text_matches.append({
                        "text_handle": str(text_handles[text_index]),
                        "content": str(text_contents[text_index]) if text_index < len(text_contents) else "",
                        "normalized_distance": text_attrs[0] if len(text_attrs) > 0 else None,
                        "candidate_confidence": text_attrs[1] if len(text_attrs) > 1 else None,
                        "semantic_compatibility": text_attrs[2] if len(text_attrs) > 2 else None,
                        "has_diameter_symbol": bool(text_attrs[3]) if len(text_attrs) > 3 else False,
                        "has_radius_symbol": bool(text_attrs[4]) if len(text_attrs) > 4 else False,
                        "diameter_value": _optional_float(text_attrs, 6) if len(text_attrs) > 6 else None,
                        "radius_value": _optional_float(text_attrs, 8) if len(text_attrs) > 8 else None,
                    })
                record = {
                    "dimension_handle": str(dimension_handles[dimension_index]),
                    "dimension_text_candidates": dimension_text_matches,
                    "relation": relation,
                    "anchor": _dimension_anchor_details(data, dimension_index, relation),
                    "dimension_type": (
                        str(dimension_types[dimension_index])
                        if dimension_index < len(dimension_types) else "UNKNOWN"
                    ),
                    "display_value": _optional_float(dimension_values, dimension_index),
                    "actual_value": _optional_float(dimension_actual_values, dimension_index),
                    "normalized_distance": attrs[0] if len(attrs) > 0 else None,
                    "candidate_confidence": attrs[1] if len(attrs) > 1 else None,
                    "dimension_evidence": {
                        "is_diametric": bool(attrs[2]) if len(attrs) > 2 else False,
                        "is_radius": bool(attrs[3]) if len(attrs) > 3 else False,
                        "is_linear": bool(attrs[4]) if len(attrs) > 4 else False,
                        "is_angular": bool(attrs[5]) if len(attrs) > 5 else False,
                        "display_value_valid": bool(attrs[6]) if len(attrs) > 6 else False,
                        "display_value_normalized": attrs[7] if len(attrs) > 7 else None,
                        "actual_value_valid": bool(attrs[8]) if len(attrs) > 8 else False,
                        "actual_value_normalized": attrs[9] if len(attrs) > 9 else None,
                        "geometry_is_circle_or_arc": bool(attrs[10]) if len(attrs) > 10 else False,
                        "geometry_is_closed": bool(attrs[11]) if len(attrs) > 11 else False,
                    },
                    "related_dimensions": _related_dimensions(data, dimension_index),
                }
                dimension_candidates.append(record)

        unique_dimensions = {}
        for item in dimension_candidates:
            key = (item["dimension_handle"], item["relation"])
            unique_dimensions[key] = item

        text_candidates = []
        text_relations = (
            "dimension_value_candidate", "part_label_candidate",
            "geometry_description_candidate", "gdt_candidate", "describes",
        )
        for text_relation in text_relations:
            for text_index, attrs in _edge_candidates(
                data, ("text", text_relation, "geometry"), geometry_index
            ):
                text_candidates.append({
                    "text_handle": str(text_handles[text_index]),
                    "content": str(text_contents[text_index]) if text_index < len(text_contents) else "",
                    "normalized_content": str(text_normalized[text_index]) if text_index < len(text_normalized) else "",
                    "semantic_type": str(text_semantics[text_index]) if text_index < len(text_semantics) else "UNKNOWN",
                    "relation": text_relation,
                    "bbox_distance": attrs[0] if len(attrs) > 0 else None,
                    "text_center_geometry_distance": attrs[1] if len(attrs) > 1 else None,
                    "text_center_geometry_center_distance": attrs[2] if len(attrs) > 2 else None,
                    "normalized_distance": attrs[3] if len(attrs) > 3 else (attrs[0] if attrs else None),
                    "candidate_confidence": attrs[4] if len(attrs) > 4 else (attrs[1] if len(attrs) > 1 else None),
                    "semantic_compatibility": attrs[5] if len(attrs) > 5 else None,
                    "has_diameter_symbol": bool(attrs[6]) if len(attrs) > 6 else False,
                    "has_radius_symbol": bool(attrs[7]) if len(attrs) > 7 else False,
                    "diameter_value": _optional_float(attrs, 9),
                    "radius_value": _optional_float(attrs, 11),
                })

        predicted = torch.expm1(
            measurement_log[geometry_index].clamp(min=0.0, max=30.0)
        ).detach().cpu().tolist()
        predicted_length, predicted_area, predicted_radius = [float(value) for value in predicted]
        entity_type = str(entity_types[geometry_index]).upper()
        is_closed = bool(closed_flags[geometry_index])
        area_applicable = is_closed
        radius_applicable = entity_type in {"CIRCLE", "ARC"}
        direct_values = torch.expm1(evidence["direct_log"][geometry_index].clamp(0.0, 30.0)).cpu().tolist()
        relation_values = torch.expm1(evidence["relation_log"][geometry_index].clamp(0.0, 30.0)).cpu().tolist()
        fusion_weights = evidence["fusion_weight"][geometry_index].cpu().tolist()
        if not area_applicable:
            predicted_area = None
        if not radius_applicable:
            predicted_radius = None
        predicted_type = id_to_class.get(int(class_ids[geometry_index]), "UNKNOWN")

        predictions.append({
            "element_id": data["geometry"].element_id[geometry_index],
            "source_handle": str(geometry_handles[geometry_index]),
            "predicted_type": predicted_type,
            "confidence": float(scores[geometry_index].cpu()),
            "predicted_length": predicted_length,
            "predicted_area": predicted_area,
            "predicted_radius": predicted_radius,
            "direct_measurements": {
                "length": float(direct_values[0]),
                "area": float(direct_values[1]) if area_applicable else None,
                "radius": float(direct_values[2]) if radius_applicable else None,
            },
            "relation_measurements": {
                "length": float(relation_values[0]),
                "area": float(relation_values[1]) if area_applicable else None,
                "radius": float(relation_values[2]) if radius_applicable else None,
            },
            "fusion_weights": {
                "length": float(fusion_weights[0]),
                "area": float(fusion_weights[1]),
                "radius": float(fusion_weights[2]),
            },
            "measurement_validity": {
                "length": True,
                "area": area_applicable,
                "radius": radius_applicable,
            },
            "entity_type": entity_type,
            "is_closed": is_closed,
            "raw_dxf_radius": _optional_float(raw_dxf_radii, geometry_index),
            "analytic_length": _optional_float(analytic_lengths, geometry_index),
            "analytic_area": _optional_float(analytic_areas, geometry_index),
            "analytic_radius": _optional_float(analytic_radii, geometry_index),
            "raw_radius_valid": bool(radius_valid[geometry_index]) if geometry_index < len(radius_valid) else False,
            "analytic_length_valid": bool(length_valid[geometry_index]) if geometry_index < len(length_valid) else False,
            "analytic_area_valid": bool(area_valid[geometry_index]) if geometry_index < len(area_valid) else False,
            # Backward-compatible aliases remain analytic values. New code
            # should use explicit predicted_* / analytic_* field names.
            "length": _optional_float(analytic_lengths, geometry_index),
            "area": _optional_float(analytic_areas, geometry_index),
            "dimension_candidates": list(unique_dimensions.values()),
            "text_candidates": text_candidates,
        })
    return predictions
