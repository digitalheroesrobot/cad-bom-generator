# DXF semantic GNN: independent measurement inference

This optional subsystem runs beside the existing deterministic DXF engine.
It does not replace DXF extraction, geometry conversion, the existing analytic
measurement calculator, rule-based classification, or BOM code.

## 1. What the model predicts

The GNN is now a multi-task heterogeneous model with three independent
measurement outputs and the existing geometry classification output:

- geometry class
- length
- area
- radius

**The existing analytical `GeometryElement.length` and `.area` are never used
as model inputs or regression targets.** They remain attached to the graph only
for reporting and later fusion with the AI predictions.

The 24-dimensional geometry input includes a radius descriptor, as requested.
The supervised radius target is stored separately in `radius_y` and
`radius_mask`. For a genuinely independent radius evaluation, ground-truth
radius labels must be reviewed values that are not copied blindly from that
same input descriptor.

Regression targets are encoded as `log1p(value)` during training. Inference
converts the model output back to the original measurement units.

## 2. Graph inputs and relationships

Node feature dimensions:

| Node | Features |
|---|---:|
| geometry | 24 |
| dimension | 15 |
| text | 41 |
| anchor | 12 |

The geometry feature vector contains position, bounding dimensions, vertex
count, closed/valid flags, geometry-type encoding, entity-type hash, and the
permitted radius descriptor. It does **not** contain analytical length or area.

The graph preserves geometry relations from the current builder, geometry
anchors, dimension-to-anchor and dimension-to-geometry candidate roles, and
text-to-geometry candidates. It also adds dimension-to-dimension candidates:

- `near`
- `parallel`
- `same_axis` (parallel with a small axis offset)
- `chain_candidate` (dimension endpoints close enough to form a possible chain)

These are candidate relations, not ground-truth constraints. Each drawing
contains the same canonical relation schema, including empty edge stores, so
graphs can be trained together. Edge attributes are consumed by the model's
relation-specific message functions.

## 3. Model architecture

`DXFSemanticGNN` now contains:

1. **Heterogeneous Message Passing** — typed messages move between geometry,
   dimension, anchor, and text nodes. Each relation has its own message
   function and uses edge attributes.
2. **Relation Reasoning Layer** — combines the accumulated evidence at each
   node. In particular, dimension nodes can exchange information with other
   dimensions and communicate through anchors to geometry nodes.
3. **Classification Head** — predicts the geometry category.
4. **Measurement Inference Head** — predicts log-length, log-area, and
   log-radius for every geometry node.

The model does not read `data["geometry"].measured_length` or
`data["geometry"].measured_area`. Those fields are kept only as analytical
outputs for comparison/fusion after inference.

## 4. Build and save semantic graphs

Use the existing pipeline objects:

```python
from app.services.dxf_engine.ai_gnn import build_semantic_graph
import torch

semantic_graph = build_semantic_graph(
    geometry_elements=geometry_elements,
    dimension_elements=dimension_elements,
    text_elements=text_elements,
    raw_entities=raw_entities,
    spatial_index=spatial_index,
    scale=scale,
    spatial_tolerance=300.0,
    dimension_tolerance=10000.0,
    text_tolerance=300.0,
    connection_tolerance=1.0,
)
torch.save(semantic_graph, "graphs/drawing_001.pt")
```

Or request graph export from `analyze_dxf`:

```python
result = analyze_dxf(
    dxf_path,
    ai_graph_output_path="graphs/drawing_001.pt",
    ai_graph_options={
        "spatial_tolerance": 300.0,
        "dimension_tolerance": 10000.0,
        "text_tolerance": 300.0,
        "connection_tolerance": 1.0,
        "angular_tolerance_degrees": 3.0,
    },
)
```

The AI graph is saved separately; the normal analysis result remains
JSON-serializable and contains an `ai_graph` summary when export is requested.

## 5. Ground-truth label format

Create one reviewed CSV per drawing. `source_handle` must match the original
DXF entity handle. The file can contain classification and measurement labels:

```csv
source_handle,label,length,area,radius
2A1,BEAM,1250.0,,
2A2,PLATE,,245000.0,
2A3,CIRCLE,,,25.0
2A4,PROFILE,850.0,12000.0,
```

Leave a measurement blank when it is not applicable or has no trusted
ground-truth value. Do not fill `length` or `area` from the existing analytic
engine if the purpose is to train an independent AI estimate. Use reviewed
engineering values, verified dimensions, or other independent ground truth.

Attach labels to each graph:

```python
from app.services.dxf_engine.ai_gnn.labels import build_training_file

build_training_file(
    graph_path="graphs/drawing_001.pt",
    labels_path="labels/drawing_001.csv",
    classes_path="classes.json",  # optional; may be None for regression-only
    output_path="labeled/drawing_001.pt",
)
```

```python
python app/services/dxf_engine/ai_gnn/labels.py --graph storage/training_graphs/drawing_001.pt --labels storage/labels/labels_001.csv --classes storage/classes.json --output storage/training_graphs/train_001.pt
```

`classes.json` example:

```json
{
  "BEAM": 0,
  "COLUMN": 1,
  "PLATE": 2,
  "BOLT": 3,
  "OTHER": 4
}
```

The resulting graph stores:
- `geometry.y` and `geometry.class_mask`
- `geometry.length_y` and `geometry.length_mask`
- `geometry.area_y` and `geometry.area_mask`
- `geometry.radius_y` and `geometry.radius_mask`

Regression targets are `log1p` encoded. Missing labels are masked and do not
contribute to the loss.

## 6. Train, validation, and test separately

Split by complete drawings or projects, never by individual geometry nodes
from the same drawing. This avoids leakage between related entities from one
drawing. Use disjoint graph files for the three splits.

Example:

```bash
python -m app.services.dxf_engine.ai_gnn.train --train-graphs storage/training_graphs/train_001.pt storage/training_graphs/train_002.pt storage/training_graphs/train_003.pt --val-graphs storage/training_graphs/validation_001.pt --test-graphs storage/training_graphs/test_001.pt --classes storage/classes.json --output storage/models/dxf_semantic_gnn_multitask.pt --epochs 200 --patience 25
```

The training loop:
- optimizes classification and available measurement regression losses;
- evaluates the validation set every epoch;
- saves the checkpoint with the best validation loss;
- stops early when validation loss does not improve;
- evaluates the test set only after checkpoint selection.

Reported regression metrics are MAE in original drawing units for length,
area, and radius. Classification accuracy is reported when class labels exist.
Every split must contain at least one supervised target. The training script
checks that graph files do not overlap across splits and that their feature
and relation schemas match.

## 7. Inference and later fusion

```python
from app.services.dxf_engine.ai_gnn.inference import predict_geometry

predictions = predict_geometry(
    semantic_graph,
    "models/dxf_semantic_gnn_multitask.pt",
)
```

Each geometry prediction contains separate fields:

- `predicted_length`, `predicted_area`, `predicted_radius`: GNN outputs
- `analytic_length`, `analytic_area`, `analytic_radius`: existing engine values
- `dimension_candidates`: candidate dimensions and anchor references
- `related_dimensions`: dimension-to-dimension candidate relationships
- `text_candidates`: nearby text candidates

For backward compatibility, `length` and `area` still alias the analytic
values. New fusion code should use the explicit `predicted_*` and `analytic_*`
names. No fusion algorithm is imposed here: it should be designed and
validated separately using confidence, residuals, and engineering constraints.

## 8. Important limitations

- The model predicts measurements only to the extent that training labels
  represent reliable ground truth and cover the drawing distribution.
- Dimension association edges are geometric candidates. The model can learn
  from their context, but the current implementation does not have a separate
  supervised dimension-link classification head.
- Angular, ordinate, radial, and diametric dimensions may need more precise
  DXF definition-point extraction and curated examples for reliable constraint
  interpretation.
- The GNN's regression output is an estimate, not a replacement for dimension
  verification or engineering review.
- Feature version 3 is incompatible with older graph files/checkpoints.
  Rebuild semantic graphs, attach labels again, and retrain.
