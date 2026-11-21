# Relation-aware DXF heterogeneous GNN

This optional AI subsystem runs alongside the deterministic DXF engine. It does
not replace DXF parsing, geometry conversion, analytical measurement,
dimension extraction, topology, classification rules, or BOM generation.

## Architecture

The graph contains four node types:

- Geometry
- Dimension
- Anchor
- Text

The relation-aware model uses relation-specific message functions and
destination-normalized attention. Incoming messages compete through a
softmax over all incoming edges for each destination node. Stacked
heterogeneous message-passing layers support multi-hop paths such as
Dimension -> Anchor -> Geometry. Separate direct and relation-based
measurement heads are combined by a learnable confidence/fusion head.

The output includes geometry classification, positive length/area/radius
estimates, direct and relation-based estimates, and per-task fusion weights.

## Feature schema (version 7)

| Node | Feature dimension |
|---|---:|
| geometry | 43 |
| dimension | 17 |
| text | 41 |
| anchor | 12 |

Geometry features preserve position, bounds, vertex count, is_closed,
is_valid, geometry-type one-hot, entity-type hash, and a radius-availability
flag. The raw DXF radius is deliberately excluded from model inputs because
it is the supervised radius target; this prevents target leakage. Analytical
length and area are not model inputs.

The v7 geometry vector adds scale-normalized boundary descriptors: bounding-box
diagonal/aspect ratio, perimeter and convex-hull perimeter, segment count and
length statistics, PCA principal-axis spans, hole count/perimeter and boundary
fractions, polygonal/circle/arc indicators, and explicit perimeter/hole-data
availability flags. Length-like descriptors are divided by the drawing scale.

Area itself and area-derived descriptors (area fill ratio, solidity and
circularity) are intentionally excluded from model inputs because they can
directly reveal the supervised Area target. The extractor also does not read
the source object's analytical length/area or raw DXF radius. For polygonal
closed geometry, hole topology is extracted from polygon interior rings; for
closed LineStrings, a polygonal view is attempted only when valid.

Dimension features include endpoint-derived direction/span and explicit
validity flags for displayed and actual dimension values. Definition points
are used to associate ordinary linear/aligned dimensions with measured
geometry; endpoints remain an independent source of dimension-line direction.

Special dimension semantics are handled before generic linear attachment:
RADIUS and DIAMETRIC use center/measurement-point semantics, while ANGULAR
and ANGULAR_3P use their available definition points (or endpoint fallback).
This prevents special dimensions from being silently consumed by the generic
measured-point branch.

Dimension-to-geometry edges now carry dimension-kind flags, displayed and
actual value validity/value, and geometry circle/arc/closed applicability.
These features preserve the Dimension -> Anchor -> Geometry evidence path,
including special RADIUS and DIAMETRIC attachment semantics.

The geometry relation builder exposes independent `spatial_tolerance`,
`collinear_tolerance`, `mirror_tolerance`, `connection_tolerance`, and
`text_tolerance` controls. Curvature and tangent continuity are edge/anchor
evidence, not hard constraints. Mirror candidates search unoriented axes in
5-degree coarse increments over 180 degrees, followed by a 1-degree local
refinement; adjust `mirror_tolerance` separately from the collinearity
tolerance.

Geometry-to-geometry edge attributes include a validity indicator for
endpoint distance. Missing/non-finite values are not treated as trustworthy
zero measurements; downstream code can distinguish unavailable evidence.

## Text semantics and evidence (schema version 7)

Text remains a first-class node type. The configurable
`text_semantics.json` file defines normalization substitutions, tokenization
settings, category patterns, diameter/radius parsing, and entity-type
compatibility. Edit this file for drawing-specific conventions; for example,
the sample `?10` to `I10` replacement is disabled by default. Enable it only
if that substitution is correct for your drawing convention.

The text categories are:

- `DIMENSION_VALUE`: diameter/radius/linear measurement strings.
- `PART_LABEL`: part names, section identifiers, and part numbers.
- `GEOMETRY_DESCRIPTION`: descriptions of geometric features.
- `ANNOTATION`: general notes and manufacturing/load instructions.
- `TOLERANCE_GDT`: tolerances and GD&T callouts.

Annotation text is preserved as a text node but has no outgoing text-to-dimension
or text-to-geometry edges, so general notes cannot indirectly influence a
geometry prediction through a nearby dimension.
The four geometry-link relation types are
`dimension_value_candidate`, `part_label_candidate`,
`geometry_description_candidate`, and `gdt_candidate`.

Text-to-geometry edge features combine three independent spatial measures (text
bounding-box to geometry bounding-box, text center to geometry, and text center
to geometry center), semantic compatibility, diameter/radius symbols and values,
geometry-type and entity-type indicators, closed-state, aspect ratio, bounding-box
span, circularity, and relative diameter/radius fit. The builder also inspects
existing Dimension -> Geometry measurement links (including the Dimension ->
Anchor -> Geometry path) and records whether a text diameter/radius value agrees
with a linked DIAMETRIC/RADIUS dimension. Candidate ranking uses a configurable
weighted combination of all three distances and semantic compatibility; a
nearby but semantically unsuitable geometry should not automatically win. A
candidate edge is evidence, not a verified text/geometry correspondence.

Chinese text uses jieba when installed. If it is unavailable, deterministic
Chinese character bigrams are used so graph generation does not require an
additional dependency. The JSON file includes editable rules for AutoCAD %%C,
%%D, and %%P symbols, mixed identifiers such as H294X200 and SJ-R-12000,
section-like labels such as 1#??, and optional OCR corrections such as ?10 ->
I10. OCR corrections that could change engineering meaning are disabled by
default and must be enabled only after checking the drawing convention. The
same file controls Chinese tokenization and candidate-ranking weights.

Geometry-to-geometry relations now use a separate `collinear_tolerance`,
plus tangent-continuity and curvature-similarity evidence. Mirror candidates
test multiple unoriented axes over the full 180-degree range, rather than
assuming only a vertical mirror. Curvature features are derived from
tessellated geometry; the raw DXF radius target is not copied into model
features.

Diameter/radius text and the existing Dimension -> Anchor -> Geometry paths
are both retained as evidence routes. A parsed diameter is not treated as a
verified radius/diameter target until a correspondence is reviewed.

## Measurement applicability and provenance

| Task | Validity rule |
|---|---|
| Length | Finite reviewed target |
| Area | Geometry is_closed == true |
| Radius | DXF entity_type is CIRCLE or ARC |

The raw DXF `radius` is kept separately as the supervised target and from
the independently calculated `radius_measured`. A missing value is represented by a numeric placeholder
plus a validity mask; the placeholder is never interpreted as a valid target.

For training labels, use reviewed ground truth in CSV/JSON keyed by DXF
`source_handle`. Radius labels should represent the raw CAD radius when that
is the desired prediction target. Do not copy analytic outputs into labels
without independent review.

Example CSV:

```csv
source_handle,label,length,area,radius
2A1,BEAM,1250.0,,
2A2,PLATE,,245000.0,
2A3,CIRCLE,,,25.0
```

Leave unavailable or inapplicable targets blank. The label builder masks area
for open geometry and radius for entities other than CIRCLE/ARC.

## Rebuild graphs and labels

Feature version 7 is incompatible with version 3, version 4, and version 5
graphs and checkpoints. The expanded 38-value text-to-geometry edge schema changes the model
checkpoint architecture, so retrain from scratch after rebuilding and
relabeling graphs.
Rebuild every graph, regenerate labeled graphs, and retrain before inference.

```python
from app.services.dxf_engine.ai_gnn import build_semantic_graph
import torch

graph = build_semantic_graph(
    geometry_elements=geometry_elements,
    dimension_elements=dimension_elements,
    text_elements=text_elements,
    raw_entities=raw_entities,
    analytic_graph=analytic_graph,
    spatial_index=spatial_index,
    scale=scale,
    spatial_tolerance=300.0,
    dimension_tolerance=2000.0,
    text_tolerance=300.0,
    collinear_tolerance=20.0,
    mirror_tolerance=5.0,
    text_config_path=None,  # defaults to ai_gnn/text_semantics.json
)
torch.save(graph, "graphs/drawing_001_v7.pt")
```

Attach labels:

```bash
python -m app.services.dxf_engine.ai_gnn.labels \
  --graph graphs/drawing_001_v7.pt \
  --labels labels/drawing_001.csv \
  --classes classes.json \
  --output labeled/drawing_001_v7.pt
```

### Reviewed text-to-geometry correspondence labels

The candidate edges are not treated as ground truth. To train the correspondence
head, add reviewed rows to the same CSV/JSON label file used for geometry
labels:

```csv
text_handle,geometry_handle,relation_type,is_match
T100,CIRCLE_01,dimension_value_candidate,1
T101,PLATE_03,part_label_candidate,1
T100,LINE_88,dimension_value_candidate,0
```

Use `is_match=1` for a reviewed correct correspondence and `0` for a reviewed
negative candidate. `relation_type=*` applies the reviewed pair label to all
semantic candidate relation types. Unlisted candidates remain masked; they are
not silently treated as negatives. Include hard negatives from the same
drawing when possible. The training objective adds a binary correspondence
loss only for reviewed edges, and validation reports text-link precision,
recall, F1, and labeled-edge count.

## Train / validation / test

Split by complete drawings or projects, never by geometry rows from the same
drawing. Keep all split graph files disjoint.

```bash
python -m app.services.dxf_engine.ai_gnn.train \
  --train-graphs labeled/train_001_v7.pt labeled/train_002_v7.pt \
  --val-graphs labeled/validation_001_v7.pt \
  --test-graphs labeled/test_001_v7.pt \
  --classes classes.json \
  --output models/dxf_relation_aware_gnn_v7.pt \
  --epochs 500 --patience 25 --checkpoint-every 100
```

Training checkpoints are written under `models/checkpoints/`:

- `checkpoint_epoch_0100.pt`, `checkpoint_epoch_0200.pt`, etc. retain
  numbered snapshots every 100 completed epochs.
- `checkpoint_latest.pt` is atomically replaced with the newest resumable
  snapshot.
- The final `dxf_relation_aware_gnn_v7.pt` remains the best-validation model
  intended for inference; it is not the same thing as the latest training
  snapshot.

To resume after an interruption, keep the same graph splits, class mapping,
and model configuration, then pass the latest checkpoint. The `--epochs`
value is the desired total epoch count, not the number of additional epochs:

```bash
python -m app.services.dxf_engine.ai_gnn.train \
  --train-graphs labeled/train_001_v7.pt labeled/train_002_v7.pt \
  --val-graphs labeled/validation_001_v7.pt \
  --test-graphs labeled/test_001_v7.pt \
  --classes classes.json \
  --output models/dxf_relation_aware_gnn_v7.pt \
  --epochs 500 --patience 25 --checkpoint-every 100 \
  --resume-from models/checkpoints/checkpoint_latest.pt
```

A resumable snapshot stores model weights, optimizer state, best-validation
weights/loss, early-stopping counter, and random-number-generator state.
Resume validation rejects mismatched graph splits, feature dimensions, class
mapping, or model architecture settings rather than silently continuing an
incompatible run. The periodic checkpoint is a training-resume artifact;
use the final best-validation model file for inference.

The loss supervises the fused estimate and applies smaller auxiliary losses
to the direct and relation-based heads. Checkpoint selection uses validation
loss; the test split is evaluated only after selection.

## Inference

```python
from app.services.dxf_engine.ai_gnn.inference import predict_geometry

predictions = predict_geometry(
    graph,
    "models/dxf_relation_aware_gnn_v7.pt",
)
```

Each prediction exposes:

- `predicted_length`, `predicted_area`, `predicted_radius`
- `direct_measurements` and `relation_measurements`
- `fusion_weights` (weight assigned to the direct branch)
- `measurement_validity`
- `raw_dxf_radius` and `analytic_radius`
- associated dimension, anchor, and text candidates
- text-candidate spatial distances, semantic compatibility, geometry/entity
  type indicators, shape descriptors, and linked DIAMETRIC/RADIUS dimension
  agreement evidence

Area and radius are returned as `None` when the geometry is not eligible.
Candidate relations remain evidence, not verified engineering constraints.

## Engineering limitations and next validation

The present relation matcher uses edge-conditioned attention to score and
aggregate candidate relationships. It is not yet a separately supervised
dimension-to-geometry correspondence classifier. Equal-length and mirrored
shape edges are geometric candidates, not hard constraints. Part-to-whole
sums, area composition, and exact Pythagorean equations are not yet solved
as explicit constraint systems; add these only with independently validated
evidence and provenance. Fusion weights are learned weights, not calibrated
probabilities.

Before production BOM use, evaluate by held-out drawings from different
projects, units, CAD conventions, and dimension styles. Report per-task MAE,
coverage, applicability-mask counts, calibration, and dimension-link
precision/recall. A plausible prediction alone is not proof of correctness.


### v7 semantic graph tests

Run the v7 text/geometry regression tests from the `backend` directory:

```bash
python -m pytest tests/test_text_semantics_v7.py tests/test_text_geometry_evidence_v7.py -v
```

The previous `test_text_semantics_v5.py` test module was replaced because its
expected graph version and text-to-geometry edge width no longer describe the
v6 schema.


## Geometry feature schema v7

The geometry node has 43 inputs: the preserved 24-value v6 prefix plus 19
versioned descriptors from `geometry_features.py`. The 19-value suffix is
ordered by `GEOMETRY_FEATURE_NAMES`; its final flags identify whether
perimeter and interior-ring information are available. Unavailable values
use numeric placeholders only alongside these validity flags.

Rebuild every graph, reattach reviewed labels, and retrain from scratch.
A v6 checkpoint cannot be used with v7 graphs. Existing v6 graphs and
checkpoints remain valid historical artifacts but are not migrated in place.
