from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import torch


def _read_label_rows(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = []
        for handle, value in payload.items():
            if isinstance(value, dict):
                rows.append({"source_handle": handle, **value})
            else:
                rows.append({"source_handle": handle, "label": value})
        return rows
    if not isinstance(payload, list):
        raise ValueError("Labels JSON must be an object or a list of records.")
    return payload


def load_handle_labels(path: str | Path) -> dict[str, str]:
    """Load optional geometry classification labels keyed by DXF handle."""
    result = {}
    for row in _read_label_rows(path):
        handle = row.get("source_handle") or row.get("handle")
        label = row.get("label") or row.get("class")
        if handle and label:
            result[str(handle).strip()] = str(label).strip()
    return result


def load_geometry_targets(path: str | Path) -> dict[str, dict[str, float]]:
    """Load reviewed independent measurement targets from CSV or JSON.

    Expected columns/keys: source_handle,length,area,radius.
    Do NOT populate these columns from the existing analytical geometry
    length/area if the goal is to train an independent AI estimator.
    Blank fields are treated as unavailable targets.
    """
    targets = {}
    for row in _read_label_rows(path):
        handle = row.get("source_handle") or row.get("handle")
        if not handle:
            continue
        values = {}
        for name in ("length", "area", "radius"):
            raw = row.get(name)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value) and value >= 0.0:
                values[name] = value
        if values:
            targets[str(handle).strip()] = values
    return targets


def attach_labels(data, handle_labels: dict[str, str], class_to_id: dict[str, int]):
    """Attach reviewed geometry classification labels; unlabeled nodes are masked."""
    handles = data["geometry"].source_handle
    y = [-100] * len(handles)
    mask = [False] * len(handles)
    for index, handle in enumerate(handles):
        label = handle_labels.get(str(handle))
        if label is not None and label in class_to_id:
            y[index] = int(class_to_id[label])
            mask[index] = True
    data["geometry"].y = torch.tensor(y, dtype=torch.long)
    data["geometry"].class_mask = torch.tensor(mask, dtype=torch.bool)
    # Backward-compatible alias used by the original classifier-label workflow.
    data["geometry"].train_mask = data["geometry"].class_mask.clone()
    return data


def load_text_geometry_labels(path: str | Path) -> dict[tuple[str, str, str], int]:
    """Load reviewed text-to-geometry correspondences from CSV/JSON rows.

    Required fields: text_handle, geometry_handle, is_match.
    Optional relation_type is a semantic candidate relation name; '*' labels
    every semantic relation for that text/geometry pair.
    """
    result = {}
    for row in _read_label_rows(path):
        text_handle = row.get("text_handle")
        geometry_handle = row.get("geometry_handle")
        raw_match = row.get("is_match")
        if not text_handle or not geometry_handle or raw_match in (None, ""):
            continue
        value = str(raw_match).strip().lower()
        if value in {"1", "true", "yes", "y", "match", "positive"}:
            label = 1
        elif value in {"0", "false", "no", "n", "nonmatch", "negative"}:
            label = 0
        else:
            continue
        relation = str(row.get("relation_type") or "*").strip()
        result[(str(text_handle).strip(), str(geometry_handle).strip(), relation)] = label
    return result


def attach_text_geometry_labels(data, labels):
    """Attach binary match targets to candidate text-to-geometry edges."""
    handles_text = [str(v) for v in data["text"].source_handle]
    handles_geometry = [str(v) for v in data["geometry"].source_handle]
    candidate_relations = {
        "dimension_value_candidate", "part_label_candidate",
        "geometry_description_candidate", "gdt_candidate",
    }
    counts = {"positive": 0, "negative": 0, "unlabeled": 0}
    for edge_type in data.edge_types:
        if edge_type[0] != "text" or edge_type[2] != "geometry" or edge_type[1] not in candidate_relations:
            continue
        edge_index = data[edge_type].edge_index
        targets, mask = [], []
        for edge_i in range(edge_index.size(1)):
            text_index, geometry_index = int(edge_index[0, edge_i]), int(edge_index[1, edge_i])
            text_handle, geometry_handle = handles_text[text_index], handles_geometry[geometry_index]
            label = labels.get(
                (text_handle, geometry_handle, edge_type[1]),
                labels.get((text_handle, geometry_handle, "*")),
            )
            if label is None:
                targets.append(0)
                mask.append(False)
                counts["unlabeled"] += 1
            else:
                targets.append(int(label))
                mask.append(True)
                counts["positive" if label else "negative"] += 1
        data[edge_type].match_y = torch.tensor(targets, dtype=torch.float)
        data[edge_type].match_mask = torch.tensor(mask, dtype=torch.bool)
    data.graph_metadata["text_geometry_supervision"] = {
        "provenance": "reviewed_text_geometry_correspondence_labels",
        "counts": counts,
    }
    return data


def attach_geometry_targets(data, targets: dict[str, dict[str, float]]):
    """Attach log1p regression targets with independent per-task masks."""
    handles = data["geometry"].source_handle
    entity_types = [str(v).upper() for v in getattr(data["geometry"], "entity_type", ["OTHER"] * len(handles))]
    closed = getattr(data["geometry"], "is_closed", torch.zeros(len(handles), dtype=torch.bool))
    for name in ("length", "area", "radius"):
        values = [0.0] * len(handles)
        mask = [False] * len(handles)
        for index, handle in enumerate(handles):
            raw_value = targets.get(str(handle), {}).get(name)
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(value) or value < 0.0:
                continue
            # Task-specific semantic validity prevents meaningless zero targets.
            if name == "area" and not bool(closed[index]):
                continue
            if name == "radius" and entity_types[index] not in {"CIRCLE", "ARC"}:
                continue
            values[index] = math.log1p(value)
            mask[index] = True
        data["geometry"][f"{name}_y"] = torch.tensor(values, dtype=torch.float)
        data["geometry"][f"{name}_mask"] = torch.tensor(mask, dtype=torch.bool)
    data.graph_metadata["target_encoding"] = "log1p"
    data.graph_metadata["target_validity"] = {
        "area": "is_closed == true",
        "radius": "entity_type in {CIRCLE, ARC}",
        "length": "finite reviewed target",
    }
    data.graph_metadata["target_provenance"] = "reviewed_external_ground_truth"
    return data


def load_class_mapping(path: str | Path) -> dict[str, int]:
    """Load class names and IDs from classes.json."""
    path = Path(path)
    class_to_id = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(class_to_id, dict):
        raise ValueError("classes.json must contain a JSON object.")
    class_to_id = {
        str(name).strip(): int(class_id)
        for name, class_id in class_to_id.items()
    }
    if len(set(class_to_id.values())) != len(class_to_id):
        raise ValueError("Duplicate class IDs found in classes.json.")
    return class_to_id


def build_training_file(
    graph_path: str | Path,
    labels_path: str | Path,
    classes_path: str | Path | None,
    output_path: str | Path,
):
    """Attach classification and independent length/area/radius ground truth."""
    graph_path = Path(graph_path)
    output_path = Path(output_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if data.graph_metadata.get("feature_version") != 7:
        raise ValueError("Rebuild the semantic graph with feature_version=7.")

    handle_labels = load_handle_labels(labels_path)
    class_to_id = load_class_mapping(classes_path) if classes_path else {}
    if class_to_id:
        data = attach_labels(data, handle_labels, class_to_id)
    else:
        count = len(data["geometry"].source_handle)
        data["geometry"].y = torch.full((count,), -100, dtype=torch.long)
        data["geometry"].class_mask = torch.zeros(count, dtype=torch.bool)
        data["geometry"].train_mask = data["geometry"].class_mask.clone()

    targets = load_geometry_targets(labels_path)
    data = attach_geometry_targets(data, targets)
    data = attach_text_geometry_labels(data, load_text_geometry_labels(labels_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    print("=" * 60)
    print("Labeled semantic graph created")
    print(f"Graph            : {graph_path}")
    print(f"Labels           : {labels_path}")
    print(f"Output           : {output_path}")
    print(f"Geometry nodes   : {len(data['geometry'].source_handle)}")
    print(f"Class labels     : {int(data['geometry'].class_mask.sum())}")
    for name in ("length", "area", "radius"):
        print(f"{name.title():16s}: {int(data['geometry'][name + '_mask'].sum())}")
    print(f"Text link labels : {data.graph_metadata.get('text_geometry_supervision', {}).get('counts', {})}")
    print("=" * 60)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attach reviewed labels to a semantic graph.")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--labels", required=True, help="CSV/JSON with source_handle,label,length,area,radius")
    parser.add_argument("--classes", default=None, help="Optional class-to-ID JSON mapping")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_training_file(args.graph, args.labels, args.classes, args.output)
