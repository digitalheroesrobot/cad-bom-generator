from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


# GeometryElement.geometry_type is assigned by geometry/entity_converter.py.
# Keep eight one-hot slots; the 24th feature is the permitted radius descriptor,
# but use the actual classifier vocabulary so known shapes are not all-zero.
GEOMETRY_TYPES = (
    "CLOSED_PROFILE", "LINEAR_GEOMETRY", "POINT", "OTHER",
    "UNKNOWN", "POLYGON", "LINESTRING", "MULTILINESTRING",
)
DIMENSION_TYPES = (
    "LINEAR", "ALIGNED", "ANGULAR", "DIAMETRIC",
    "RADIUS", "ANGULAR_3P", "ORDINATE", "UNKNOWN",
)

# Anchor kinds are deliberately compact and deterministic.
ANCHOR_TYPES = (
    "START", "END", "VERTEX", "CENTER",
    "INTERSECTION", "DIMENSION_ATTACH",
)


def _get(obj: Any, name: str, default=None):
    """Read a field from either a dataclass/object or a mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _hash_bucket(token: str, size: int) -> int:
    digest = hashlib.blake2b(
        token.encode("utf-8", errors="ignore"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % size


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _xy(point):
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None


def _bounds(geometry):
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    try:
        return tuple(float(v) for v in geometry.bounds)
    except Exception:
        return None


def _line_parts(geometry):
    if geometry is None:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        result = []
        for part in geometry.geoms:
            result.extend(_line_parts(part))
        return result
    if geometry.geom_type == "Polygon":
        return [geometry.exterior]
    if geometry.geom_type == "MultiPolygon":
        return [polygon.exterior for polygon in geometry.geoms]
    return []


def _geometry_feature(element, scale: float, origin):
    """Build 24 geometry inputs; length and area are intentionally excluded.

    Radius is retained as the explicitly permitted geometric descriptor.
    Ground-truth radius targets are stored separately and never copied from
    this input inside the model.
    """
    geometry = _get(element, "geometry_2d")
    bounds = _bounds(geometry)
    if bounds:
        minx, miny, maxx, maxy = bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        width, height = maxx - minx, maxy - miny
        try:
            vertex_count = sum(len(part.coords) for part in _line_parts(geometry))
            if vertex_count == 0 and hasattr(geometry, "exterior") and geometry.exterior:
                vertex_count = len(geometry.exterior.coords)
        except Exception:
            vertex_count = 0
    else:
        cx = cy = width = height = vertex_count = 0.0

    center_xy = _xy(_get(element, "center"))
    if center_xy:
        cx, cy = center_xy

    scale = max(abs(scale), 1e-9)
    ox, oy = origin
    geom_type = str(_get(element, "geometry_type", "OTHER")).upper()
    entity_type = str(_get(_get(element, "source"), "entity_type", "OTHER")).upper()
    type_vector = [1.0 if geom_type == item else 0.0 for item in GEOMETRY_TYPES]
    entity_hash = [0.0] * 8
    entity_hash[_hash_bucket(entity_type, 8)] = 1.0

    # Existing analytical length and area are deliberately NOT features.
    # Radius is an allowed shape descriptor; the supervised radius target is
    # separately supplied in geometry.radius_y with its own mask.
    radius = _safe_float(_get(element, "radius"), 0.0) / scale
    radius_valid = _get(element, "radius") is not None and math.isfinite(_safe_float(_get(element, "radius"), float("nan")))
    return [
        (cx - ox) / scale, (cy - oy) / scale,
        width / scale, height / scale,
        math.log1p(max(0.0, vertex_count)),
        float(bool(_get(element, "is_closed", False))),
        float(bool(_get(element, "is_valid", True))),
        *type_vector, *entity_hash, radius, float(radius_valid),
    ]


def _dimension_feature(dimension, scale: float, origin):
    """Keep the original 15-dimensional dimension feature contract."""
    scale = max(abs(scale), 1e-9)
    endpoints = _get(dimension, "endpoints")
    p1 = _xy(endpoints[0]) if endpoints else None
    p2 = _xy(endpoints[1]) if endpoints else None
    if p1 and p2:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = (p2[0] - p1[0]) / scale, (p2[1] - p1[1]) / scale
        span = math.hypot(dx, dy)
    else:
        mx = my = dx = dy = span = 0.0
    center = _xy(_get(dimension, "center"))
    if center:
        mx, my = center
    ox, oy = origin
    kind = str(_get(dimension, "dimension_type", "UNKNOWN")).upper()
    onehot = [1.0 if kind == item else 0.0 for item in DIMENSION_TYPES]
    value = _get(dimension, "length")
    actual = _get(dimension, "actual_length")
    value_valid = value is not None and math.isfinite(_safe_float(value, float("nan")))
    actual_valid = actual is not None and math.isfinite(_safe_float(actual, float("nan")))
    return [
        _safe_float(value) / scale,
        _safe_float(actual) / scale,
        span,
        (mx - ox) / scale,
        (my - oy) / scale,
        dx, dy,
        *onehot,
        float(value_valid), float(actual_valid),
    ]


def _text_feature(text, scale: float, origin, lexical_size: int = 32):
    content = str(_get(text, "text", "") or "")
    position = _xy(_get(text, "position")) or _xy(_get(text, "position_moved"))
    if position is None:
        position = (0.0, 0.0)
    bound = _get(text, "bound") or _get(text, "bound_moved")
    bounds = _bounds(bound)
    if bounds:
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    else:
        width = height = 0.0
    scale = max(abs(scale), 1e-9)
    ox, oy = origin
    tokens = re.findall(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[^\w\s]", content.lower())
    lexical = [0.0] * lexical_size
    for token in tokens:
        lexical[_hash_bucket(token, lexical_size)] += 1.0
    norm = max(1.0, math.sqrt(sum(v * v for v in lexical)))
    lexical = [v / norm for v in lexical]
    angle = _safe_float(_get(text, "angle", 0.0))
    return [
        (position[0] - ox) / scale,
        (position[1] - oy) / scale,
        width / scale, height / scale,
        angle,
        math.log1p(len(content)),
        math.log1p(sum(ch.isdigit() for ch in content)),
        float(bool(re.search(r"[Ø⌀φΦ]", content))),
        float(bool(re.search(r"\b(PL|PLATE|BEAM|COLUMN|BOLT|WELD|STEEL|H\d+)\b", content, re.I))),
        *lexical,
    ]


def _raw_text_items(raw_entities):
    """Include text-like RawEntity records not already represented by TextElement."""
    for raw in raw_entities or []:
        source = _get(raw, "source")
        entity_type = str(_get(source, "entity_type", "")).upper()
        if entity_type not in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
            continue
        content = _get(raw, "text")
        if not content:
            continue
        attrs = _get(raw, "attributes", {}) or {}
        position = attrs.get("center") or attrs.get("insert") or attrs.get("position")
        corners = attrs.get("corners")
        bound = None
        if corners and len(corners) >= 3:
            try:
                from shapely.geometry import Polygon
                bound = Polygon([(p[0], p[1]) for p in corners])
            except Exception:
                bound = None
        yield {
            "element_id": _get(source, "entity_handle", ""),
            "source_handle": _get(source, "entity_handle", ""),
            "source": source,
            "text": content,
            "position": Point(position[0], position[1]) if position else None,
            "position_moved": None,
            "bound": bound,
            "bound_moved": None,
            "angle": math.radians(_safe_float(attrs.get("angle", 0.0))),
        }


def _add_edge(edge_store, edge_type, source_idx, target_idx, attrs):
    """Append one typed edge with a numeric feature vector."""
    edge_store[edge_type].append(
        (int(source_idx), int(target_idx), [_safe_float(v) for v in attrs])
    )


def _angle_between(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    # Direction is unoriented for CAD line parallelism.
    return math.acos(min(1.0, abs(dot)))


def _segment_records(geometry):
    records = []
    for part in _line_parts(geometry):
        coords = list(part.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                records.append({
                    "start": (float(a[0]), float(a[1])),
                    "end": (float(b[0]), float(b[1])),
                    "length": length,
                    "direction": (dx / length, dy / length),
                })
    return records


def _endpoint_points(geometry):
    points = []
    if geometry is not None and geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return points
    for part in _line_parts(geometry):
        coords = list(part.coords)
        if len(coords) >= 2:
            points.extend([Point(coords[0]), Point(coords[-1])])
    if geometry is not None and geometry.geom_type == "Point":
        points.append(geometry)
    return points


def _representative_direction(geometry):
    # A single orientation is meaningful only for a straight open line.
    # Do not infer one from a circle, closed profile, bent polyline, or arc.
    if geometry is None or geometry.geom_type != "LineString":
        return None
    segments = _segment_records(geometry)
    if not segments:
        return None
    reference = max(segments, key=lambda item: item["length"])["direction"]
    tolerance = math.radians(3.0)
    if any(
        (_angle_between(reference, segment["direction"]) or 0.0) > tolerance
        for segment in segments
    ):
        return None
    return reference


def _collinear_score(geometry_a, geometry_b, direction_a, direction_b, tolerance):
    if direction_a is None or direction_b is None:
        return 0.0
    angle = _angle_between(direction_a, direction_b)
    if angle is None or angle > math.radians(3.0):
        return 0.0
    segments_a = _segment_records(geometry_a)
    segments_b = _segment_records(geometry_b)
    if not segments_a or not segments_b:
        return 0.0
    a = max(segments_a, key=lambda item: item["length"])
    b = max(segments_b, key=lambda item: item["length"])
    dx, dy = b["start"][0] - a["start"][0], b["start"][1] - a["start"][1]
    perpendicular = abs(dx * direction_a[1] - dy * direction_a[0])
    if perpendicular > max(tolerance, 1e-9):
        return 0.0
    return max(0.0, 1.0 - perpendicular / max(tolerance, 1e-9))


def _intersection_point_coords(geometry):
    """Return point-like coordinates from a Shapely intersection result."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geometry.geom_type in {"MultiPoint", "GeometryCollection"}:
        points = []
        for part in geometry.geoms:
            points.extend(_intersection_point_coords(part))
        return points
    # For overlapping linework, the overlap endpoints are useful anchors.
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        return [(float(coords[0][0]), float(coords[0][1])),
                (float(coords[-1][0]), float(coords[-1][1]))] if len(coords) >= 2 else []
    if geometry.geom_type == "MultiLineString":
        points = []
        for part in geometry.geoms:
            points.extend(_intersection_point_coords(part))
        return points
    return []


def _anchor_specs(element):
    """Create a small, deterministic set of meaningful anchors per geometry."""
    geom = _get(element, "geometry_2d")
    if geom is None or geom.is_empty:
        return []
    specs = []
    entity_type = str(_get(_get(element, "source"), "entity_type", "")).upper()

    center = _xy(_get(element, "center"))
    if center is None:
        try:
            c = geom.centroid
            center = (float(c.x), float(c.y))
        except Exception:
            center = None
    if center is not None and entity_type in {"CIRCLE", "ARC"}:
        specs.append((center, "CENTER", (0.0, 0.0), 0.0))

    # A flattened CIRCLE is represented as a polygon for geometric predicates.
    # Its approximation vertices are not useful semantic anchors.
    if entity_type == "CIRCLE":
        return specs

    segments = _segment_records(geom)
    if geom.geom_type == "Point":
        c = _xy(geom)
        if c:
            specs.append((c, "CENTER", (0.0, 0.0), 0.0))
    else:
        for part in _line_parts(geom):
            coords = list(part.coords)
            if len(coords) >= 2:
                first = coords[0]
                last = coords[-1]
                specs.append(((float(first[0]), float(first[1])), "START",
                              segments[0]["direction"] if segments else (0.0, 0.0), 0.0))
                specs.append(((float(last[0]), float(last[1])), "END",
                              segments[-1]["direction"] if segments else (0.0, 0.0), 0.0))
                # Interior vertices are useful for polyline and polygon corners.
                for vertex in coords[1:-1]:
                    specs.append(((float(vertex[0]), float(vertex[1])), "VERTEX",
                                  (0.0, 0.0), 0.0))

    # Remove duplicate coordinates while retaining the first semantic kind.
    unique = []
    seen = set()
    for coord, kind, tangent, curvature in specs:
        key = (round(coord[0], 8), round(coord[1], 8))
        if key not in seen:
            seen.add(key)
            unique.append((coord, kind, tangent, curvature))
    return unique


def _anchor_feature(coord, kind, tangent, curvature, scale, origin, center):
    scale = max(abs(scale), 1e-9)
    ox, oy = origin
    type_vector = [1.0 if kind == item else 0.0 for item in ANCHOR_TYPES]
    center_distance = (
        math.hypot(coord[0] - center[0], coord[1] - center[1]) / scale
        if center else 0.0
    )
    return [
        (coord[0] - ox) / scale,
        (coord[1] - oy) / scale,
        *type_vector,
        _safe_float(tangent[0] if tangent else 0.0),
        _safe_float(tangent[1] if tangent else 0.0),
        _safe_float(curvature),
        center_distance,
    ]


def _make_relation_attrs(distance, angle_diff, intersection_length,
                         endpoint_distance, overlap_ratio, confidence, scale):
    scale = max(abs(scale), 1e-9)
    return [
        _safe_float(distance) / scale,
        _safe_float(angle_diff) / math.pi if angle_diff is not None else 0.0,
        _safe_float(intersection_length) / scale,
        _safe_float(endpoint_distance) / scale,
        _safe_float(overlap_ratio),
        max(0.0, min(1.0, _safe_float(confidence))),
        float(endpoint_distance is not None and math.isfinite(_safe_float(endpoint_distance, float("nan")))),
    ]


def build_semantic_graph(
    geometry_elements,
    dimension_elements,
    text_elements,
    raw_entities=None,
    analytic_graph=None,
    spatial_index=None,
    scale: float = 1.0,
    spatial_tolerance: float = 300.0,
    dimension_tolerance: float = 2000.0,
    text_tolerance: float = 300.0,
    max_neighbors: int = 16,
    connection_tolerance: float = 200.0,
    angular_tolerance_degrees: float = 3.0,
):
    """Build a DXF engineering HeteroData graph.

    Existing geometry/dimension/text node types and their feature dimensions
    are preserved, with a radius descriptor added to the geometry features.
    The graph also includes anchors, dimension-to-geometry/anchor candidates,
    and dimension-to-dimension candidate relationships.

    Analytical length and area remain metadata only: they are never included
    in geometry.x. Independent reviewed regression targets are attached later
    by ai_gnn.labels and consumed by the multi-task GNN.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for the AI graph. Install the "
            "PyTorch build matching your environment, then install "
            "`torch-geometric`."
        ) from exc

    geometry_elements = list(geometry_elements or [])
    dimension_elements = list(dimension_elements or [])
    text_elements = list(text_elements or [])

    # Prefer rich TextElement instances, then include extracted raw MTEXT,
    # ATTRIB and ATTDEF records not represented by those objects.
    known_text_handles = {str(_get(t, "source_handle", "")) for t in text_elements}
    for item in _raw_text_items(raw_entities):
        if str(_get(item, "source_handle", "")) not in known_text_handles:
            text_elements.append(item)

    scale_safe = max(abs(_safe_float(scale, 1.0)), 1e-9)
    all_points = []
    for item in geometry_elements:
        b = _bounds(_get(item, "geometry_2d"))
        if b:
            all_points.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    for item in dimension_elements:
        ep = _get(item, "endpoints")
        if ep:
            all_points.extend([p for p in (_xy(ep[0]), _xy(ep[1])) if p])
        definition_points = _get(item, "definition_points")
        if definition_points:
            all_points.extend([p for p in (_xy(point) for point in definition_points) if p])
        for field in ("center", "measurement_point"):
            p = _xy(_get(item, field))
            if p:
                all_points.append(p)
    for item in text_elements:
        p = _xy(_get(item, "position")) or _xy(_get(item, "position_moved"))
        if p:
            all_points.append(p)
    origin = (
        (min(p[0] for p in all_points), min(p[1] for p in all_points))
        if all_points else (0.0, 0.0)
    )

    data = HeteroData()

    def feature_tensor(rows, width):
        if not rows:
            return torch.empty((0, width), dtype=torch.float)
        return torch.tensor(rows, dtype=torch.float).reshape(len(rows), width)

    data["geometry"].x = feature_tensor(
        [_geometry_feature(x, scale_safe, origin) for x in geometry_elements], 25
    )
    data["dimension"].x = feature_tensor(
        [_dimension_feature(x, scale_safe, origin) for x in dimension_elements], 17
    )
    data["text"].x = feature_tensor(
        [_text_feature(x, scale_safe, origin) for x in text_elements], 41
    )

    # Preserve traceability and expose exact deterministic geometry values.
    for node_type, items in (
        ("geometry", geometry_elements),
        ("dimension", dimension_elements),
        ("text", text_elements),
    ):
        data[node_type].element_id = [str(_get(x, "element_id", "")) for x in items]
        data[node_type].source_handle = [
            str(_get(x, "source_handle", "") or _get(_get(x, "source"), "entity_handle", ""))
            for x in items
        ]
        data[node_type].source_layer = [
            str(_get(_get(x, "source"), "layer", "")) for x in items
        ]

    data["geometry"].entity_type = [
        str(_get(_get(x, "source"), "entity_type", "OTHER")).upper()
        for x in geometry_elements
    ]
    data["geometry"].geometry_type = [
        str(_get(x, "geometry_type", "OTHER")).upper()
        for x in geometry_elements
    ]
    data["geometry"].is_closed = torch.tensor(
        [bool(_get(x, "is_closed", False)) for x in geometry_elements], dtype=torch.bool
    )
    def optional_measurement(item, field):
        raw = _get(item, field)
        try:
            value = float(raw)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    data["geometry"].measured_length = [optional_measurement(x, "length") for x in geometry_elements]
    data["geometry"].measured_area = [optional_measurement(x, "area") for x in geometry_elements]
    # radius is the raw DXF entity radius; analytic radius_measured is separate.
    data["geometry"].radius = [optional_measurement(x, "radius") for x in geometry_elements]
    data["geometry"].radius_measured = [optional_measurement(x, "radius_measured") for x in geometry_elements]
    def valid_source_value(item, field):
        raw = _get(item, field)
        try:
            return raw is not None and math.isfinite(float(raw))
        except (TypeError, ValueError):
            return False
    data["geometry"].measured_length_valid = torch.tensor(
        [valid_source_value(x, "length") for x in geometry_elements], dtype=torch.bool
    )
    data["geometry"].measured_area_valid = torch.tensor(
        [valid_source_value(x, "area") for x in geometry_elements], dtype=torch.bool
    )
    data["geometry"].radius_valid = torch.tensor(
        [valid_source_value(x, "radius") for x in geometry_elements], dtype=torch.bool
    )
    data["dimension"].dimension_type = [
        str(_get(x, "dimension_type", "UNKNOWN")) for x in dimension_elements
    ]
    data["dimension"].display_value = [
        _safe_float(_get(x, "length"), float("nan")) for x in dimension_elements
    ]
    data["dimension"].actual_value = [
        _safe_float(_get(x, "actual_length"), float("nan")) for x in dimension_elements
    ]
    data["dimension"].definition_points = [
        [
            _xy(point) for point in (
                _get(x, "definition_points")
                or _get(x, "endpoints")
                or ()
            )
        ] for x in dimension_elements
    ]
    data["text"].content = [
        str(_get(x, "text", "") or "") for x in text_elements
    ]

    geom_by_handle = {
        str(_get(x, "source_handle", "") or _get(_get(x, "source"), "entity_handle", "")): i
        for i, x in enumerate(geometry_elements)
    }

    def candidate_geometry_indices(query, distance):
        if spatial_index is None:
            return range(len(geometry_elements))
        if query is None or getattr(query, "is_empty", False):
            return []
        try:
            candidates = spatial_index.nearby(
                query, distance=distance, exclude_handle=None
            )
        except Exception:
            return range(len(geometry_elements))
        indices = []
        for candidate in candidates:
            handle = str(
                _get(candidate, "source_handle", "")
                or _get(_get(candidate, "source"), "entity_handle", "")
            )
            index = geom_by_handle.get(handle)
            if index is not None:
                indices.append(index)
        return sorted(set(indices))

    edge_store = defaultdict(list)
    gcount, dcount, tcount = len(geometry_elements), len(dimension_elements), len(text_elements)

    # --------------------------------------------------------------
    # Phase 1: Geometry-to-geometry semantic relations
    # --------------------------------------------------------------
    # Keep the existing analytical-graph proximity edges.
    if analytic_graph is not None:
        for u, v, attrs in analytic_graph.edges(data=True):
            if not (isinstance(u, tuple) and isinstance(v, tuple)
                    and u[0] == "G" and v[0] == "G"):
                continue
            # The existing topology graph can contain endpoint-connected,
            # crossing, and other relations. Do not relabel those as `near`.
            analytic_relation = str(attrs.get("relationship", "near")).lower()
            if analytic_relation not in {"near", "spatial_candidate"}:
                continue
            ui = geom_by_handle.get(str(u[1]))
            vi = geom_by_handle.get(str(v[1]))
            if ui is None or vi is None or ui == vi:
                continue
            ga = _get(geometry_elements[ui], "geometry_2d")
            gb = _get(geometry_elements[vi], "geometry_2d")
            distance = _safe_float(attrs.get("distance", 0.0))
            _add_edge(
                edge_store, ("geometry", "near", "geometry"), ui, vi,
                _make_relation_attrs(
                    distance, None, 0.0, float("inf"), 0.0,
                    attrs.get("confidence", 1.0), scale_safe
                ),
            )

    angular_tol = math.radians(max(0.0, angular_tolerance_degrees))
    pair_seen = set()

    for i, a in enumerate(geometry_elements):
        ga = _get(a, "geometry_2d")
        if ga is None or ga.is_empty:
            continue
        candidate_indices = candidate_geometry_indices(ga, spatial_tolerance)
        for j in candidate_indices:
            if j <= i:
                continue
            b = geometry_elements[j]
            gb = _get(b, "geometry_2d")
            if gb is None or gb.is_empty:
                continue

            try:
                distance = float(ga.distance(gb))
                intersects = bool(ga.intersects(gb))
                touches = bool(ga.touches(gb))
                overlaps = bool(ga.overlaps(gb))
                contains = bool(ga.contains(gb))
                within = bool(ga.within(gb))
                intersection = ga.intersection(gb) if intersects else None
                intersection_length = float(intersection.length) if intersection is not None else 0.0
                intersection_area = float(intersection.area) if intersection is not None else 0.0
                union_measure = max(float(ga.area), float(gb.area), 1e-9)
                overlap_ratio = intersection_area / union_measure if intersection_area else (
                    intersection_length / max(min(float(ga.length), float(gb.length)), 1e-9)
                    if intersection_length else 0.0
                )
            except Exception:
                continue

            dir_a = _representative_direction(ga)
            dir_b = _representative_direction(gb)
            angle_diff = _angle_between(dir_a, dir_b)
            endpoints_a = _endpoint_points(ga)
            endpoints_b = _endpoint_points(gb)
            endpoint_distance = min(
                (pa.distance(pb) for pa in endpoints_a for pb in endpoints_b),
                default=float("inf"),
            )
            collinear_score = _collinear_score(
                ga, gb, dir_a, dir_b, spatial_tolerance
            )

            relation_records = []
            if distance <= spatial_tolerance:
                relation_records.append(("near", max(0.0, 1.0 - distance / max(spatial_tolerance, 1e-9))))
            if angle_diff is not None and angle_diff <= angular_tol:
                relation_records.append(("parallel", max(0.0, 1.0 - angle_diff / max(angular_tol, 1e-9))))
            if angle_diff is not None and abs((math.pi / 2) - angle_diff) <= angular_tol:
                relation_records.append(("perpendicular", max(0.0, 1.0 - abs((math.pi / 2) - angle_diff) / max(angular_tol, 1e-9))))
            if intersects:
                relation_records.append(("intersects", 1.0))
            if touches:
                relation_records.append(("touches", 1.0))
            if overlaps:
                relation_records.append(("overlaps", 1.0))
            if contains:
                relation_records.append(("contains", 1.0))
            if within:
                relation_records.append(("within", 1.0))
            if endpoint_distance <= max(connection_tolerance, 0.0):
                relation_records.append((
                    "connected",
                    max(0.0, 1.0 - endpoint_distance / max(connection_tolerance, 1e-9))
                    if connection_tolerance > 0 else 1.0,
                ))
            if collinear_score > 0.0:
                relation_records.append(("collinear", collinear_score))

            for relation, confidence in relation_records:
                edge_type = ("geometry", relation, "geometry")
                attrs = _make_relation_attrs(
                    distance, angle_diff, intersection_length,
                    endpoint_distance, overlap_ratio, confidence, scale_safe
                )
                _add_edge(edge_store, edge_type, i, j, attrs)
                pair_seen.add((i, j, relation))

    # Preserve directed near edges from the original implementation and
    # analytic graph. Deduplicate all same-direction relation records.
    for edge_type, records in list(edge_store.items()):
        dedup = {}
        for src, dst, attrs in records:
            key = (src, dst)
            previous = dedup.get(key)
            if previous is None or attrs[-1] > previous[-1]:
                dedup[key] = attrs
        edge_store[edge_type] = [
            (src, dst, attrs) for (src, dst), attrs in dedup.items()
        ]

    # --------------------------------------------------------------
    # Phase 2: Geometry anchors and dimension measurement references
    # --------------------------------------------------------------
    anchor_rows = []
    anchor_owner = []
    anchor_coords = []
    anchor_kinds = []
    anchor_tangents = []
    geometry_centers = []

    for gi, element in enumerate(geometry_elements):
        geom = _get(element, "geometry_2d")
        center = None
        if geom is not None and not geom.is_empty:
            c = _xy(_get(element, "center"))
            if c is None:
                centroid = geom.centroid
                c = (float(centroid.x), float(centroid.y))
            center = c
        geometry_centers.append(center)
        for coord, kind, tangent, curvature in _anchor_specs(element):
            anchor_idx = len(anchor_rows)
            anchor_rows.append(_anchor_feature(
                coord, kind, tangent, curvature, scale_safe, origin, center
            ))
            anchor_owner.append(gi)
            anchor_coords.append(coord)
            anchor_kinds.append(kind)
            anchor_tangents.append(tangent)
            _add_edge(
                edge_store, ("geometry", "has_anchor", "anchor"),
                gi, anchor_idx, [0.0, 1.0],
            )

    # Reuse an existing endpoint/vertex/center anchor when the projected
    # dimension attachment coincides with it; otherwise add a dedicated anchor.
    def add_dimension_anchor(geometry_index, coord, distance, anchor_kind="DIMENSION_ATTACH"):
        for existing_index, (owner, existing_coord) in enumerate(zip(anchor_owner, anchor_coords)):
            if owner == geometry_index and math.hypot(
                coord[0] - existing_coord[0], coord[1] - existing_coord[1]
            ) <= 1e-6:
                return existing_index
        idx = len(anchor_rows)
        center = geometry_centers[geometry_index]
        anchor_rows.append(_anchor_feature(
            coord, anchor_kind, (0.0, 0.0), 0.0,
            scale_safe, origin, center
        ))
        anchor_owner.append(geometry_index)
        anchor_coords.append(coord)
        anchor_kinds.append(anchor_kind)
        anchor_tangents.append((0.0, 0.0))
        _add_edge(
            edge_store, ("geometry", "has_anchor", "anchor"),
            geometry_index, idx, [distance / scale_safe, 1.0],
        )
        return idx

    # Add shared intersection anchors to both participating geometries.
    # These are distinct graph nodes so message passing can learn which
    # geometric junctions are meaningful in the engineering drawing.
    for gi, element_a in enumerate(geometry_elements):
        ga = _get(element_a, "geometry_2d")
        if ga is None or ga.is_empty:
            continue
        for gj in candidate_geometry_indices(ga, spatial_tolerance):
            if gj <= gi:
                continue
            gb = _get(geometry_elements[gj], "geometry_2d")
            if gb is None or gb.is_empty:
                continue
            try:
                if not ga.intersects(gb):
                    continue
                intersection = ga.intersection(gb)
            except Exception:
                continue
            for coord in _intersection_point_coords(intersection):
                for owner in (gi, gj):
                    anchor_idx = len(anchor_rows)
                    center = geometry_centers[owner]
                    anchor_rows.append(_anchor_feature(
                        coord, "INTERSECTION", (0.0, 0.0), 0.0,
                        scale_safe, origin, center
                    ))
                    anchor_owner.append(owner)
                    anchor_coords.append(coord)
                    anchor_kinds.append("INTERSECTION")
                    anchor_tangents.append((0.0, 0.0))
                    _add_edge(
                        edge_store, ("geometry", "has_anchor", "anchor"),
                        owner, anchor_idx, [0.0, 1.0],
                    )

    data["anchor"].x = feature_tensor(anchor_rows, 12)
    data["anchor"].element_id = [
        f"{data['geometry'].source_handle[owner]}:A{index}"
        for index, owner in enumerate(anchor_owner)
    ]
    data["anchor"].source_handle = [
        str(data["geometry"].source_handle[owner]) for owner in anchor_owner
    ]
    data["anchor"].source_layer = [
        str(data["geometry"].source_layer[owner]) for owner in anchor_owner
    ]
    data["anchor"].owner_geometry_index = list(anchor_owner)
    data["anchor"].anchor_kind = list(anchor_kinds)
    data["anchor"].coordinates = [tuple(coord) for coord in anchor_coords]

    # Keep the original broad dimension->geometry "describes" candidates.
    # Add more specific measurement roles based on the two definition points.
    for di, dimension in enumerate(dimension_elements):
        dim_type = str(_get(dimension, "dimension_type", "UNKNOWN")).upper()
        endpoints = _get(dimension, "endpoints") or []
        definition_points = _get(dimension, "definition_points") or []
        attachment_points = []

        # Special dimension semantics must precede generic measured points.
        if dim_type in {"RADIUS", "DIAMETRIC"}:
            center = _xy(_get(dimension, "center"))
            measure = _xy(_get(dimension, "measurement_point"))
            if center:
                attachment_points.append(("measures_from", center, "CENTER"))
            if measure:
                attachment_points.append(("measures_to", measure, None))
            elif endpoints:
                p = _xy(endpoints[-1])
                if p:
                    attachment_points.append(("measures_to", p, None))
        elif dim_type in {"ANGULAR", "ANGULAR_3P"}:
            points = definition_points or endpoints
            for role, point in zip(("measures_from", "measures_to"), points[:2]):
                coord = _xy(point)
                if coord:
                    attachment_points.append((role, coord, None))
        else:
            # Definition points identify measured geometry; endpoints retain
            # the dimension arrow-line direction as an independent feature.
            measured_points = definition_points if len(definition_points) >= 2 else endpoints
            if len(measured_points) >= 2:
                attachment_points = [
                    ("measures_from", _xy(measured_points[0]), None),
                    ("measures_to", _xy(measured_points[1]), None),
                ]

        # Existing describes relation: nearest geometries to dimension
        # endpoints/center, bounded by dimension_tolerance.
        query_points = [Point(p) for _, p, _ in attachment_points if p is not None]
        if not query_points:
            center = _xy(_get(dimension, "center"))
            if center:
                query_points = [Point(center)]
        candidates = []
        candidate_indices = set()
        for query_point in query_points:
            candidate_indices.update(
                candidate_geometry_indices(query_point, dimension_tolerance)
            )
        for gi in sorted(candidate_indices):
            element = geometry_elements[gi]
            geom = _get(element, "geometry_2d")
            if geom is None or geom.is_empty or not query_points:
                continue
            dist = min(float(p.distance(geom)) for p in query_points)
            if dist <= dimension_tolerance:
                candidates.append((dist, gi))
        for dist, gi in sorted(candidates)[:max_neighbors]:
            _add_edge(
                edge_store, ("dimension", "describes", "geometry"),
                di, gi,
                [dist / scale_safe, max(0.0, 1.0 - dist / max(dimension_tolerance, 1e-9))],
            )

        # Role-specific association: each dimension definition point is
        # projected onto its nearest geometry and linked to a geometry anchor.
        # Radial center points attach to the geometry's center, not its boundary.
        for role, coord, requested_anchor_kind in attachment_points:
            if coord is None:
                continue
            point = Point(coord)
            best = None
            candidate_indices = (
                range(len(geometry_elements))
                if requested_anchor_kind == "CENTER"
                else candidate_geometry_indices(point, dimension_tolerance)
            )
            for gi in candidate_indices:
                element = geometry_elements[gi]
                geom = _get(element, "geometry_2d")
                if geom is None or geom.is_empty:
                    continue
                try:
                    if requested_anchor_kind == "CENTER":
                        element_center = _xy(_get(element, "center"))
                        if element_center is None:
                            centroid = geom.centroid
                            element_center = (float(centroid.x), float(centroid.y))
                        projected_coord = element_center
                        dist = math.hypot(coord[0] - element_center[0], coord[1] - element_center[1])
                    else:
                        projected = nearest_points(point, geom)[1]
                        projected_coord = (float(projected.x), float(projected.y))
                        dist = float(point.distance(projected))
                except Exception:
                    continue
                if dist <= dimension_tolerance and (best is None or dist < best[0]):
                    best = (dist, gi, projected_coord)
            if best is None:
                continue
            dist, gi, projected_coord = best
            anchor_idx = add_dimension_anchor(
                gi, projected_coord, dist,
                anchor_kind=requested_anchor_kind or "DIMENSION_ATTACH",
            )
            _add_edge(
                edge_store, ("dimension", role, "anchor"),
                di, anchor_idx,
                [dist / scale_safe, max(0.0, 1.0 - dist / max(dimension_tolerance, 1e-9))],
            )
            _add_edge(
                edge_store, ("anchor", "rev_" + role, "dimension"),
                anchor_idx, di,
                [dist / scale_safe, max(0.0, 1.0 - dist / max(dimension_tolerance, 1e-9))],
            )
            _add_edge(
                edge_store, ("dimension", role, "geometry"),
                di, gi,
                [dist / scale_safe, max(0.0, 1.0 - dist / max(dimension_tolerance, 1e-9))],
            )
            _add_edge(
                edge_store, ("geometry", "rev_" + role, "dimension"),
                gi, di,
                [dist / scale_safe, max(0.0, 1.0 - dist / max(dimension_tolerance, 1e-9))],
            )

    # Rebuild anchor tensors after dimension-attachment anchors were appended.
    data["anchor"].x = feature_tensor(anchor_rows, 12)
    data["anchor"].element_id = [
        f"{data['geometry'].source_handle[owner]}:A{index}"
        for index, owner in enumerate(anchor_owner)
    ]
    data["anchor"].source_handle = [
        str(data["geometry"].source_handle[owner]) for owner in anchor_owner
    ]
    data["anchor"].source_layer = [
        str(data["geometry"].source_layer[owner]) for owner in anchor_owner
    ]
    data["anchor"].owner_geometry_index = list(anchor_owner)
    data["anchor"].anchor_kind = list(anchor_kinds)
    data["anchor"].coordinates = [tuple(coord) for coord in anchor_coords]

    # Reverse ownership edges are materialized in the common reverse-edge pass.

    # --------------------------------------------------------------
    # Text -> geometry candidates: preserve original proximity behavior.
    # --------------------------------------------------------------
    for ti, text in enumerate(text_elements):
        query = _get(text, "bound") or _get(text, "bound_moved")
        if query is None:
            pos = _xy(_get(text, "position")) or _xy(_get(text, "position_moved"))
            query = Point(pos) if pos else None
        if query is None:
            continue
        candidates = []
        candidate_indices = candidate_geometry_indices(query, text_tolerance)
        for gi in candidate_indices:
            element = geometry_elements[gi]
            geom = _get(element, "geometry_2d")
            if geom is None or geom.is_empty:
                continue
            distance = float(query.distance(geom))
            if distance <= text_tolerance:
                candidates.append((distance, gi))
        for distance, gi in sorted(candidates)[:max_neighbors]:
            _add_edge(
                edge_store, ("text", "describes", "geometry"),
                ti, gi,
                [distance / scale_safe, max(0.0, 1.0 - distance / max(text_tolerance, 1e-9))],
            )

    # --------------------------------------------------------------
    # Dimension-to-dimension relationships
    # --------------------------------------------------------------
    # These are candidate relationships, not ground-truth constraints.
    # They let the GNN reason over dimension chains, parallel dimension
    # lines, and dimensions sharing a measurement axis.
    def _dimension_descriptor(item):
        endpoints = _get(item, "endpoints")
        p1 = _xy(endpoints[0]) if endpoints and len(endpoints) >= 2 else None
        p2 = _xy(endpoints[1]) if endpoints and len(endpoints) >= 2 else None
        center = _xy(_get(item, "center"))
        if center is None and p1 and p2:
            center = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        direction = None
        if p1 and p2:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 1e-9:
                direction = (dx / length, dy / length)
        return center, direction, p1, p2

    dimension_descriptors = [
        _dimension_descriptor(item) for item in dimension_elements
    ]
    cell_size = max(float(dimension_tolerance), float(connection_tolerance), 1e-6)
    dimension_buckets = defaultdict(list)
    for di, descriptor in enumerate(dimension_descriptors):
        center = descriptor[0]
        if center is not None:
            cell = (math.floor(center[0] / cell_size), math.floor(center[1] / cell_size))
            dimension_buckets[cell].append(di)

    dimension_pairs = set()
    for di, (center_a, direction_a, p1a, p2a) in enumerate(dimension_descriptors):
        if center_a is None:
            continue
        cell_x = math.floor(center_a[0] / cell_size)
        cell_y = math.floor(center_a[1] / cell_size)
        nearby_indices = set()
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                nearby_indices.update(dimension_buckets.get(
                    (cell_x + dx_cell, cell_y + dy_cell), ()
                ))
        scored = []
        for dj in nearby_indices:
            if dj <= di:
                continue
            center_b, direction_b, p1b, p2b = dimension_descriptors[dj]
            if center_b is None:
                continue
            center_distance = math.hypot(
                center_a[0] - center_b[0], center_a[1] - center_b[1]
            )
            if center_distance > dimension_tolerance:
                continue
            angle = _angle_between(direction_a, direction_b)
            endpoint_gap = min(
                (math.hypot(a[0] - b[0], a[1] - b[1])
                 for a in (p1a, p2a) if a is not None
                 for b in (p1b, p2b) if b is not None),
                default=float("inf"),
            )
            axis_gap = 0.0
            if direction_a is not None:
                axis_gap = abs(
                    (center_b[0] - center_a[0]) * direction_a[1]
                    - (center_b[1] - center_a[1]) * direction_a[0]
                )
            scored.append((center_distance, dj, angle, endpoint_gap, axis_gap))
        for center_distance, dj, angle, endpoint_gap, axis_gap in sorted(scored)[:max_neighbors]:
            normalized_distance = center_distance / scale_safe
            confidence = max(
                0.0,
                1.0 - center_distance / max(float(dimension_tolerance), 1e-9),
            )
            attrs = [
                normalized_distance,
                (angle / math.pi) if angle is not None else 0.0,
                (endpoint_gap / scale_safe) if math.isfinite(endpoint_gap) else 0.0,
                axis_gap / scale_safe,
                confidence,
            ]
            _add_edge(
                edge_store, ("dimension", "near", "dimension"), di, dj, attrs
            )
            if angle is not None and angle <= angular_tol:
                _add_edge(
                    edge_store, ("dimension", "parallel", "dimension"),
                    di, dj, attrs,
                )
                if axis_gap <= max(connection_tolerance, 1e-9):
                    _add_edge(
                        edge_store, ("dimension", "same_axis", "dimension"),
                        di, dj, attrs,
                    )
            if endpoint_gap <= max(connection_tolerance, 0.0):
                _add_edge(
                    edge_store, ("dimension", "chain_candidate", "dimension"),
                    di, dj, attrs,
                )

    # Keep a stable relation schema across drawings, including relations with
    # zero edges in a particular drawing. This is essential when training a
    # single model over separate train/validation/test drawings.
    canonical_edge_dims = {
        ("geometry", "near", "geometry"): 7,
        ("geometry", "parallel", "geometry"): 7,
        ("geometry", "perpendicular", "geometry"): 7,
        ("geometry", "intersects", "geometry"): 7,
        ("geometry", "touches", "geometry"): 7,
        ("geometry", "overlaps", "geometry"): 7,
        ("geometry", "contains", "geometry"): 7,
        ("geometry", "within", "geometry"): 7,
        ("geometry", "connected", "geometry"): 7,
        ("geometry", "collinear", "geometry"): 7,
        ("geometry", "has_anchor", "anchor"): 2,
        ("dimension", "describes", "geometry"): 2,
        ("dimension", "measures_from", "geometry"): 2,
        ("dimension", "measures_to", "geometry"): 2,
        ("dimension", "measures_from", "anchor"): 2,
        ("dimension", "measures_to", "anchor"): 2,
        ("text", "describes", "geometry"): 2,
        ("dimension", "near", "dimension"): 5,
        ("dimension", "parallel", "dimension"): 5,
        ("dimension", "same_axis", "dimension"): 5,
        ("dimension", "chain_candidate", "dimension"): 5,
    }
    for forward_type, width in canonical_edge_dims.items():
        edge_store[forward_type]  # ensure the canonical store exists


    # Materialize reverse relations in the store first. This also handles
    # explicitly authored role-reverse edges without accidentally dropping
    # both directions during tensor installation.
    for edge_type, records in list(edge_store.items()):
        if edge_type[1].startswith("rev_"):
            continue
        reverse_type = (edge_type[2], "rev_" + edge_type[1], edge_type[0])
        if reverse_type not in edge_store or not edge_store[reverse_type]:
            edge_store[reverse_type] = [
                (dst, src, list(attrs)) for src, dst, attrs in records
            ]

    # Install edge_index and edge_attr. Empty relations are materialized too,
    # keeping a stable schema across separate drawing graphs.
    for edge_type, records in list(edge_store.items()):
        if not records:
            width = canonical_edge_dims.get(
                edge_type,
                canonical_edge_dims.get(
                    (edge_type[2], edge_type[1].removeprefix("rev_"), edge_type[0]), 2
                ),
            )
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[edge_type].edge_attr = torch.empty((0, int(width)), dtype=torch.float)
            continue
        src = torch.tensor([[r[0] for r in records], [r[1] for r in records]], dtype=torch.long)
        widths = {len(r[2]) for r in records}
        if len(widths) != 1:
            raise ValueError(f"Inconsistent edge feature widths for {edge_type}: {widths}")
        attr = torch.tensor([r[2] for r in records], dtype=torch.float)
        data[edge_type].edge_index = src
        data[edge_type].edge_attr = attr

    data.graph_metadata = {
        "graph_type": "dxf_semantic_heterogeneous",
        "feature_version": 4,
        "origin": origin,
        "scale": float(scale_safe),
        "node_counts": {
            "geometry": gcount,
            "dimension": dcount,
            "text": tcount,
            "anchor": len(anchor_rows),
        },
        "feature_dims": {
            "geometry": 25,
            "dimension": 17,
            "text": 41,
            "anchor": 12,
        },
        "geometry_relation_edge_attr": [
            "normalized_distance",
            "normalized_angle_difference",
            "normalized_intersection_length",
            "normalized_endpoint_distance",
            "overlap_ratio",
            "confidence",
        ],
        "edge_feature_dims": {
            "|".join(edge_type): int(data[edge_type].edge_attr.size(-1))
            for edge_type in data.edge_types
        },
        "measurement_learning": {
            "targets": ["length", "area", "radius"],
            "analytic_length_area_used_as_input": False,
            "radius_input_feature": True,
            "radius_source": "raw_dxf_entity_radius",
            "radius_measured_is_separate": True,
            "area_validity": "is_closed",
            "radius_validity": "entity_type in {CIRCLE, ARC}",
            "target_encoding": "log1p",
        },
        "edge_counts": {
            str(edge_type): int(data[edge_type].edge_index.size(1))
            for edge_type in data.edge_types
        },
    }
    return data


def build_semantic_graph_from_analysis(
    geometry_elements,
    dimension_elements,
    text_elements,
    raw_entities=None,
    analytic_graph=None,
    **kwargs,
):
    """Named adapter for pipeline/service code; accepts already-extracted objects."""
    return build_semantic_graph(
        geometry_elements=geometry_elements,
        dimension_elements=dimension_elements,
        text_elements=text_elements,
        raw_entities=raw_entities,
        analytic_graph=analytic_graph,
        **kwargs,
    )
