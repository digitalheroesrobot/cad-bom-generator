from __future__ import annotations

import math
from typing import Any

# v7 geometry descriptors. None of these stores the supervised analytic
# Length/Area target itself. In particular, area, fill-ratio, solidity and
# circularity are deliberately excluded because they can directly reveal
# the Area target for closed profiles.
GEOMETRY_FEATURE_VERSION = 7
GEOMETRY_FEATURE_NAMES = (
    "bbox_diagonal_over_scale",
    "bbox_aspect_ratio",
    "perimeter_log_over_scale",
    "convex_hull_perimeter_log_over_scale",
    "segment_count_log",
    "segment_length_mean_over_scale",
    "segment_length_max_over_scale",
    "segment_length_std_over_scale",
    "principal_axis_major_over_scale",
    "principal_axis_minor_over_scale",
    "principal_axis_ratio",
    "hole_count_log",
    "hole_perimeter_log_over_scale",
    "hole_boundary_fraction",
    "boundary_to_hull_perimeter_ratio",
    "is_polygonal",
    "is_circle_or_arc",
    "has_perimeter",
    "has_hole_data",
)


def _get(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _line_parts(geometry):
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    kind = getattr(geometry, "geom_type", "")
    if kind in {"LineString", "LinearRing"}:
        return [geometry]
    if kind in {"MultiLineString", "GeometryCollection"}:
        result = []
        for part in geometry.geoms:
            result.extend(_line_parts(part))
        return result
    if kind == "Polygon":
        return [geometry.exterior, *list(geometry.interiors)]
    if kind == "MultiPolygon":
        result = []
        for polygon in geometry.geoms:
            result.extend([polygon.exterior, *list(polygon.interiors)])
        return result
    return []


def _polygonal_view(geometry, is_closed: bool):
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if is_closed and geometry.geom_type in {"LineString", "LinearRing"}:
        try:
            from shapely.geometry import Polygon
            candidate = Polygon(geometry)
            if candidate.is_valid and not candidate.is_empty:
                return candidate
        except Exception:
            pass
    return None


def _principal_axis_spans(parts):
    points = []
    for part in parts:
        try:
            points.extend((float(p[0]), float(p[1])) for p in part.coords)
        except Exception:
            continue
    if len(points) < 2:
        return 0.0, 0.0
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    xx = sum((p[0] - mx) ** 2 for p in points) / len(points)
    yy = sum((p[1] - my) ** 2 for p in points) / len(points)
    xy = sum((p[0] - mx) * (p[1] - my) for p in points) / len(points)
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    c, s = math.cos(angle), math.sin(angle)
    projected = [
        ((x - mx) * c + (y - my) * s,
         -(x - mx) * s + (y - my) * c)
        for x, y in points
    ]
    major = max(p[0] for p in projected) - min(p[0] for p in projected)
    minor = max(p[1] for p in projected) - min(p[1] for p in projected)
    return max(0.0, major), max(0.0, minor)


def extract_geometry_descriptors(element, scale: float = 1.0) -> list[float]:
    """Return v7 shape descriptors with explicit availability indicators.

    Length-like values use drawing-units / scale. Perimeter and segment
    statistics are boundary descriptors, not the supervised Length target.
    No analytic area or area-derived ratio is exposed to the model.
    """
    geometry = _get(element, "geometry_2d")
    scale = max(abs(_safe(scale, 1.0)), 1e-9)
    is_closed = bool(_get(element, "is_closed", False))
    entity_type = str(_get(_get(element, "source"), "entity_type", "")).upper()
    bounds = None
    if geometry is not None and not getattr(geometry, "is_empty", True):
        try:
            bounds = tuple(float(v) for v in geometry.bounds)
        except Exception:
            bounds = None

    if bounds and len(bounds) == 4:
        minx, miny, maxx, maxy = bounds
        width, height = max(0.0, maxx - minx), max(0.0, maxy - miny)
        diagonal = math.hypot(width, height) / scale
        aspect = min(width, height) / max(width, height, 1e-9)
    else:
        diagonal = aspect = 0.0

    parts = _line_parts(geometry)
    segment_lengths = []
    for part in parts:
        try:
            coords = list(part.coords)
        except Exception:
            continue
        for a, b in zip(coords[:-1], coords[1:]):
            length = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            if math.isfinite(length) and length > 1e-9:
                segment_lengths.append(length / scale)

    perimeter_raw = _safe(getattr(geometry, "length", 0.0)) if geometry is not None else 0.0
    hull_perimeter_raw = 0.0
    if geometry is not None and not getattr(geometry, "is_empty", True):
        try:
            hull_perimeter_raw = _safe(geometry.convex_hull.length)
        except Exception:
            pass
    has_perimeter = perimeter_raw > 1e-9
    perimeter_scaled = perimeter_raw / scale if has_perimeter else 0.0
    hull_perimeter_scaled = hull_perimeter_raw / scale if hull_perimeter_raw > 1e-9 else 0.0

    hole_count = 0
    hole_perimeter_raw = 0.0
    has_hole_data = False
    polygonal = _polygonal_view(geometry, is_closed)
    if polygonal is not None:
        try:
            polygons = list(polygonal.geoms) if polygonal.geom_type == "MultiPolygon" else [polygonal]
            for polygon in polygons:
                hole_count += len(polygon.interiors)
                hole_perimeter_raw += sum(float(ring.length) for ring in polygon.interiors)
            has_hole_data = True
        except Exception:
            hole_count = 0
            hole_perimeter_raw = 0.0
            has_hole_data = False

    count = len(segment_lengths)
    mean_length = sum(segment_lengths) / count if count else 0.0
    max_length = max(segment_lengths, default=0.0)
    std_length = (
        math.sqrt(sum((v - mean_length) ** 2 for v in segment_lengths) / count)
        if count else 0.0
    )
    major, minor = _principal_axis_spans(parts)
    major /= scale
    minor /= scale
    axis_ratio = minor / max(major, 1e-9) if major > 0.0 else 0.0
    hole_perimeter_scaled = hole_perimeter_raw / scale
    hole_fraction = (
        min(1.0, max(0.0, hole_perimeter_raw / perimeter_raw))
        if has_hole_data and perimeter_raw > 1e-9 else 0.0
    )
    hull_ratio = (
        min(1.0, max(0.0, perimeter_raw / hull_perimeter_raw))
        if hull_perimeter_raw > 1e-9 else 0.0
    )

    return [
        diagonal,
        aspect,
        math.log1p(perimeter_scaled),
        math.log1p(hull_perimeter_scaled),
        math.log1p(count),
        mean_length,
        max_length,
        std_length,
        major,
        minor,
        axis_ratio,
        math.log1p(hole_count),
        math.log1p(hole_perimeter_scaled),
        hole_fraction,
        hull_ratio,
        float(polygonal is not None),
        float(entity_type in {"CIRCLE", "ARC"}),
        float(has_perimeter),
        float(has_hole_data),
    ]
