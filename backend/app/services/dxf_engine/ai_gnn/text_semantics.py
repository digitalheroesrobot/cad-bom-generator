from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("text_semantics.json")


def load_text_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Text semantics configuration must be a JSON object.")
    return config


def normalize_text(content, config):
    value = str(content or "")
    for rule in config.get("replacements", []):
        pattern = str(rule.get("pattern", ""))
        if pattern:
            flags = re.IGNORECASE if rule.get("ignore_case", False) else 0
            value = re.sub(pattern, str(rule.get("replacement", "")), value, flags=flags)
    return value


def tokenize_text(content, config):
    value = normalize_text(content, config).lower()
    tokens = re.findall(
        r"%%c|[ø⌀φΦØ]|[a-z]+(?:[-_][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)?|[\u3400-\u9fff]+|[^\w\s]",
        value, flags=re.IGNORECASE,
    )
    options = config.get("chinese_tokenization", {})
    output = []
    for token in tokens:
        if re.search(r"[\u3400-\u9fff]", token):
            segmented = None
            if options.get("use_jieba", True):
                try:
                    import jieba
                    segmented = [part.strip() for part in jieba.cut(token) if part.strip()]
                except ImportError:
                    pass
            if segmented:
                output.extend(segmented)
            elif len(token) == 1:
                output.append(token)
            else:
                output.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            output.append(token)
    return output


def classify_text(content, config):
    normalized = normalize_text(content, config)
    patterns = config.get("category_patterns", {})
    # Notes take precedence so loads/manufacturing instructions never become
    # geometry evidence merely because they contain dimensions or numbers.
    for category in (
        "ANNOTATION", "TOLERANCE_GDT", "DIMENSION_VALUE",
        "PART_LABEL", "GEOMETRY_DESCRIPTION",
    ):
        for pattern in patterns.get(category, []):
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return category, normalized
    return "UNKNOWN", normalized


def parse_measurement_evidence(content, config):
    normalized = normalize_text(content, config)
    diameter_pattern = config.get(
        "diameter_pattern", r"(?:%%c|[ø⌀ØΦφ])\s*([0-9]+(?:\.[0-9]+)?)"
    )
    radius_pattern = config.get("radius_pattern", r"\bR\s*([0-9]+(?:\.[0-9]+)?)")
    diameter = re.search(diameter_pattern, normalized, flags=re.IGNORECASE)
    radius = re.search(radius_pattern, normalized, flags=re.IGNORECASE)
    return {
        "normalized": normalized,
        "has_diameter_symbol": bool(diameter),
        "diameter_value": float(diameter.group(1)) if diameter else None,
        "has_radius_symbol": bool(radius),
        "radius_value": float(radius.group(1)) if radius else None,
    }
