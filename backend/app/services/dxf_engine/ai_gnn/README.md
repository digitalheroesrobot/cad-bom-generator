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

## Feature schema (version 4)

| Node | Feature dimension |
|---|---:|
| geometry | 24 |
| dimension | 17 |
| text | 41 |
| anchor | 12 |

Geometry features preserve position, bounds, vertex count, is_closed,
is_valid, geometry-type one-hot, entity-type hash, and a radius-availability
flag. The raw DXF radius is deliberately excluded from model inputs because
it is the supervised radius target; this prevents target leakage. Analytical
length and area are not model inputs.

Dimension features include endpoint-derived direction/span and explicit
validity flags for displayed and actual dimension values. Definition points
are used to associate ordinary linear/aligned dimensions with measured
geometry; endpoints remain an independent source of dimension-line direction.

Special dimension semantics are handled before generic linear attachment:
RADIUS and DIAMETRIC use center/measurement-point semantics, while ANGULAR
and ANGULAR_3P use their available definition points (or endpoint fallback).
This prevents special dimensions from being silently consumed by the generic
measured-point branch.

Geometry-to-geometry edge attributes include a validity indicator for
endpoint distance. Missing/non-finite values are not treated as trustworthy
zero measurements; downstream code can distinguish unavailable evidence.

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

Feature version 4 is incompatible with version 3 graphs and checkpoints.
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
)
torch.save(graph, "graphs/drawing_001_v4.pt")
```

Attach labels:

```bash
python -m app.services.dxf_engine.ai_gnn.labels \
  --graph graphs/drawing_001_v4.pt \
  --labels labels/drawing_001.csv \
  --classes classes.json \
  --output labeled/drawing_001_v4.pt
```

## Train / validation / test

Split by complete drawings or projects, never by geometry rows from the same
drawing. Keep all split graph files disjoint.

```bash
python -m app.services.dxf_engine.ai_gnn.train \
  --train-graphs labeled/train_001_v4.pt labeled/train_002_v4.pt \
  --val-graphs labeled/validation_001_v4.pt \
  --test-graphs labeled/test_001_v4.pt \
  --classes classes.json \
  --output models/dxf_relation_aware_gnn_v4.pt \
  --epochs 500 --patience 25 --checkpoint-every 100
```

Training checkpoints are written under `models/checkpoints/`:

- `checkpoint_epoch_0100.pt`, `checkpoint_epoch_0200.pt`, etc. retain
  numbered snapshots every 100 completed epochs.
- `checkpoint_latest.pt` is atomically replaced with the newest resumable
  snapshot.
- The final `dxf_relation_aware_gnn_v4.pt` remains the best-validation model
  intended for inference; it is not the same thing as the latest training
  snapshot.

To resume after an interruption, keep the same graph splits, class mapping,
and model configuration, then pass the latest checkpoint. The `--epochs`
value is the desired total epoch count, not the number of additional epochs:

```bash
python -m app.services.dxf_engine.ai_gnn.train \
  --train-graphs labeled/train_001_v4.pt labeled/train_002_v4.pt \
  --val-graphs labeled/validation_001_v4.pt \
  --test-graphs labeled/test_001_v4.pt \
  --classes classes.json \
  --output models/dxf_relation_aware_gnn_v4.pt \
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
    "models/dxf_relation_aware_gnn_v4.pt",
)
```

Each prediction exposes:

- `predicted_length`, `predicted_area`, `predicted_radius`
- `direct_measurements` and `relation_measurements`
- `fusion_weights` (weight assigned to the direct branch)
- `measurement_validity`
- `raw_dxf_radius` and `analytic_radius`
- associated dimension, anchor, and text candidates

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
