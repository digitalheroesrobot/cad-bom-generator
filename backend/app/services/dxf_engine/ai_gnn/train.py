from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import DXFSemanticGNN


REGRESSION_TARGETS = ("length", "area", "radius")


def _input_dims(data):
    return {node_type: int(data[node_type].x.size(-1)) for node_type in data.node_types}


def _edge_dims(data):
    return {
        "|".join(edge_type): int(data[edge_type].edge_attr.size(-1))
        for edge_type in data.edge_types
    }


def _load_graphs(paths):
    graphs = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.graph_metadata.get("feature_version") != 7:
            raise ValueError(
                f"{path}: expected semantic graph feature_version=7; rebuild this graph."
            )
        graphs.append((str(path), data))
    return graphs


def _validate_graph_compatibility(graphs, input_dims, edge_dims):
    for path, data in graphs:
        if _input_dims(data) != input_dims:
            raise ValueError(f"{path}: node feature dimensions differ from training graph.")
        if _edge_dims(data) != edge_dims:
            raise ValueError(f"{path}: relation/edge feature schema differs from training graph.")


def _has_supervision(data):
    return bool(_supervised_tasks(data))


def _supervised_tasks(data):
    geometry = data["geometry"]
    tasks = set()
    class_mask = getattr(geometry, "class_mask", None)
    if class_mask is not None and bool(class_mask.any()):
        tasks.add("classification")
    for name in REGRESSION_TARGETS:
        mask = getattr(geometry, f"{name}_mask", None)
        if mask is not None and bool(mask.any()):
            tasks.add(name)
    for edge_type in data.edge_types:
        if edge_type[0] == "text" and edge_type[2] == "geometry":
            match_mask = getattr(data[edge_type], "match_mask", None)
            if match_mask is not None and bool(match_mask.any()):
                tasks.add("text_geometry_matching")
                break
    return tasks


def _split_tasks(graphs):
    tasks = set()
    for _, data in graphs:
        tasks.update(_supervised_tasks(data))
    return tasks


def _loss_for_graph(logits, measurement_log, evidence, data, class_weight=1.0,
                    regression_weights=None):
    regression_weights = regression_weights or {
        "length": 1.0, "area": 1.0, "radius": 0.5
    }
    geometry = data["geometry"]
    losses = []
    details = {}

    class_mask = getattr(geometry, "class_mask", None)
    if class_mask is not None and bool(class_mask.any()):
        class_loss = F.cross_entropy(logits[class_mask], geometry.y[class_mask])
        losses.append(class_weight * class_loss)
        details["classification"] = class_loss.detach()

    for column, name in enumerate(REGRESSION_TARGETS):
        mask = getattr(geometry, f"{name}_mask", None)
        target = getattr(geometry, f"{name}_y", None)
        if mask is None or target is None or not bool(mask.any()):
            continue
        prediction = measurement_log[mask, column]
        task_loss = F.smooth_l1_loss(prediction, target[mask])
        direct_loss = F.smooth_l1_loss(evidence["direct_log"][mask, column], target[mask])
        relation_loss = F.smooth_l1_loss(evidence["relation_log"][mask, column], target[mask])
        weight = float(regression_weights.get(name, 1.0))
        losses.append(weight * (task_loss + 0.15 * direct_loss + 0.15 * relation_loss))
        details[name] = task_loss.detach()
        details[f"{name}_direct_aux"] = direct_loss.detach()
        details[f"{name}_relation_aux"] = relation_loss.detach()

    # Supervised semantic correspondence is applied only to edges with
    # reviewed labels. Unlabeled candidates remain useful message-passing
    # evidence but never become implicit negative examples.
    for edge_type, match_logits in evidence.get("text_match_logits", {}).items():
        store = data[edge_type]
        match_mask = getattr(store, "match_mask", None)
        match_targets = getattr(store, "match_y", None)
        if match_mask is None or match_targets is None or not bool(match_mask.any()):
            continue
        selected_targets = match_targets[match_mask]
        positive_count = selected_targets.sum()
        negative_count = selected_targets.numel() - positive_count
        positive_weight = (
            (negative_count / positive_count.clamp(min=1.0)).clamp(min=0.25, max=10.0)
            if positive_count > 0 else selected_targets.new_tensor(1.0)
        )
        match_loss = F.binary_cross_entropy_with_logits(
            match_logits[match_mask], selected_targets,
            pos_weight=positive_weight.reshape(1),
        )
        losses.append(0.5 * match_loss)
        details[f"text_match_{edge_type[1]}"] = match_loss.detach()

    if not losses:
        raise ValueError("Graph contains no supervised classification, measurement, or text-link targets.")
    return torch.stack(losses).sum(), details


@torch.no_grad()
def _evaluate(model, graphs, device, class_weight, regression_weights):
    model.eval()
    loss_values = []
    class_correct = 0
    class_total = 0
    regression_errors = {name: [] for name in REGRESSION_TARGETS}
    text_link_tp = text_link_fp = text_link_fn = text_link_count = 0

    for _, cpu_data in graphs:
        data = cpu_data.to(device)
        logits, measurement_log, _, evidence = model(data)
        loss, _ = _loss_for_graph(
            logits, measurement_log, evidence, data,
            class_weight=class_weight,
            regression_weights=regression_weights,
        )
        loss_values.append(float(loss.detach().cpu()))

        for edge_type, match_logits in evidence.get("text_match_logits", {}).items():
            store = data[edge_type]
            match_mask = getattr(store, "match_mask", None)
            match_targets = getattr(store, "match_y", None)
            if match_mask is None or match_targets is None or not bool(match_mask.any()):
                continue
            predicted_match = match_logits[match_mask] >= 0.0
            true_match = match_targets[match_mask] >= 0.5
            text_link_tp += int((predicted_match & true_match).sum().item())
            text_link_fp += int((predicted_match & ~true_match).sum().item())
            text_link_fn += int((~predicted_match & true_match).sum().item())
            text_link_count += int(match_mask.sum().item())

        geometry = data["geometry"]
        class_mask = getattr(geometry, "class_mask", None)
        if class_mask is not None and bool(class_mask.any()):
            predicted = logits[class_mask].argmax(dim=-1)
            class_correct += int((predicted == geometry.y[class_mask]).sum().item())
            class_total += int(class_mask.sum().item())

        for column, name in enumerate(REGRESSION_TARGETS):
            mask = getattr(geometry, f"{name}_mask", None)
            target = getattr(geometry, f"{name}_y", None)
            if mask is None or target is None or not bool(mask.any()):
                continue
            # Targets are log1p encoded. Report MAE in original drawing units.
            pred_value = torch.expm1(measurement_log[mask, column].clamp(min=0.0, max=30.0))
            true_value = torch.expm1(target[mask].clamp(min=0.0, max=30.0))
            regression_errors[name].extend(
                torch.abs(pred_value - true_value).detach().cpu().tolist()
            )

    metrics = {
        "loss": sum(loss_values) / max(1, len(loss_values)),
        "classification_accuracy": (
            class_correct / class_total if class_total else None
        ),
        "classification_count": class_total,
        "text_link_precision": text_link_tp / max(1, text_link_tp + text_link_fp) if text_link_count else None,
        "text_link_recall": text_link_tp / max(1, text_link_tp + text_link_fn) if text_link_count else None,
        "text_link_f1": (
            2.0 * text_link_tp / max(1, 2 * text_link_tp + text_link_fp + text_link_fn)
            if text_link_count else None
        ),
        "text_link_count": text_link_count,
    }
    for name, errors in regression_errors.items():
        metrics[f"{name}_mae"] = (
            sum(errors) / len(errors) if errors else None
        )
        metrics[f"{name}_count"] = len(errors)
    return metrics



def _cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _save_training_checkpoint(
    checkpoint_dir,
    epoch,
    model,
    optimizer,
    best_validation_loss,
    best_state,
    epochs_without_improvement,
    class_to_id,
    input_dims,
    edge_dims,
    hidden_channels,
    num_layers,
    dropout,
    text_matching_trained,
    train_graph_paths,
    validation_graph_paths,
    test_graph_paths,
):
    """Atomically save a resumable training snapshot and retain numbered copies."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": 1,
        "epoch": int(epoch),
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": float(best_validation_loss),
        "best_model_state_dict": best_state,
        "epochs_without_improvement": int(epochs_without_improvement),
        "class_to_id": class_to_id,
        "input_dims": input_dims,
        "edge_dims": edge_dims,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "text_matching_trained": bool(text_matching_trained),
        "feature_version": 7,
        "architecture": "relation_aware_hybrid_text_match_v4_geometry_v7",
        "target_encoding": "log1p",
        "regression_targets": list(REGRESSION_TARGETS),
        "split": {
            "train_graphs": list(train_graph_paths),
            "validation_graphs": list(validation_graph_paths),
            "test_graphs": list(test_graph_paths),
        },
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    latest_tmp = checkpoint_dir / "checkpoint_latest.pt.tmp"
    torch.save(checkpoint, latest_tmp)
    os.replace(latest_tmp, latest_path)

    numbered_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    numbered_tmp = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt.tmp"
    torch.save(checkpoint, numbered_tmp)
    os.replace(numbered_tmp, numbered_path)

    print(f"[CHECKPOINT SAVED] {numbered_path}")
    print(f"[LATEST CHECKPOINT] {latest_path}")


def _restore_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
    class_to_id,
    input_dims,
    edge_dims,
    hidden_channels,
    num_layers,
    dropout,
    text_matching_trained,
    train_graph_paths,
    validation_graph_paths,
    test_graph_paths,
):
    """Restore model, optimizer, early-stopping state, and RNG state."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    expected = {
        "feature_version": 7,
        "architecture": "relation_aware_hybrid_text_match_v4_geometry_v7",
        "input_dims": input_dims,
        "edge_dims": edge_dims,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "text_matching_trained": bool(text_matching_trained),
        "class_to_id": class_to_id,
        "split": {
            "train_graphs": list(train_graph_paths),
            "validation_graphs": list(validation_graph_paths),
            "test_graphs": list(test_graph_paths),
        },
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Resume checkpoint mismatch for '{key}'. "
                "Use the same model configuration, class mapping, and graph splits."
            )

    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(
            "This file is not a resumable training checkpoint "
            "(optimizer_state_dict is missing)."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rng_state = checkpoint.get("rng_state", {})
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"].cpu())
    cuda_rng = rng_state.get("cuda")
    if cuda_rng is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_rng)
        except (RuntimeError, ValueError):
            print("[WARN] Could not restore CUDA RNG state; continuing.")

    best_state = checkpoint.get("best_model_state_dict")
    if best_state is None:
        raise ValueError("Resume checkpoint does not contain best-model weights.")

    print(
        f"[RESUME] Restored epoch {checkpoint['epoch']} from {checkpoint_path}; "
        f"best validation loss={checkpoint['best_validation_loss']:.6f}"
    )
    return (
        int(checkpoint["epoch"]),
        float(checkpoint["best_validation_loss"]),
        best_state,
        int(checkpoint.get("epochs_without_improvement", 0)),
    )


def train_model(
    train_graph_paths,
    validation_graph_paths,
    test_graph_paths,
    class_to_id: dict[str, int] | None,
    output_path: str,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_channels: int = 96,
    num_layers: int = 2,
    dropout: float = 0.20,
    patience: int = 25,
    device: str | None = None,
    class_weight: float = 1.0,
    regression_weights: dict[str, float] | None = None,
    checkpoint_every: int = 100,
    resume_from: str | None = None,
):
    """Train on drawing-level train graphs, select by validation, test once.

    Split by complete drawing/project, never by geometry nodes from the same
    drawing. The test set is not used for optimization or checkpoint selection.
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1.")
    if not train_graph_paths or not validation_graph_paths or not test_graph_paths:
        raise ValueError(
            "Provide separate non-empty train, validation, and test graph lists."
        )
    train_graphs = _load_graphs(train_graph_paths)
    validation_graphs = _load_graphs(validation_graph_paths)
    test_graphs = _load_graphs(test_graph_paths)

    all_paths = (
        [p for p, _ in train_graphs]
        + [p for p, _ in validation_graphs]
        + [p for p, _ in test_graphs]
    )
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("A graph file appears in more than one split.")

    input_dims = _input_dims(train_graphs[0][1])
    edge_dims = _edge_dims(train_graphs[0][1])
    _validate_graph_compatibility(
        train_graphs + validation_graphs + test_graphs,
        input_dims, edge_dims,
    )

    if not any(_has_supervision(data) for _, data in train_graphs):
        raise ValueError("Training split contains no supervised labels.")
    if not any(_has_supervision(data) for _, data in validation_graphs):
        raise ValueError("Validation split contains no supervised labels.")
    if not any(_has_supervision(data) for _, data in test_graphs):
        raise ValueError("Test split contains no supervised labels.")

    train_tasks = _split_tasks(train_graphs)
    validation_tasks = _split_tasks(validation_graphs)
    test_tasks = _split_tasks(test_graphs)
    required_measurements = {"length", "area"}
    missing_train = required_measurements - train_tasks
    if missing_train:
        raise ValueError(
            "Training split needs independent ground-truth labels for: "
            + ", ".join(sorted(missing_train))
        )
    required_eval_tasks = {"length", "area", "radius", "classification", "text_geometry_matching"}
    missing_validation = (train_tasks & required_eval_tasks) - validation_tasks
    missing_test = (train_tasks & required_eval_tasks) - test_tasks
    if missing_validation:
        raise ValueError(
            "Validation split has no labels for trained task(s): "
            + ", ".join(sorted(missing_validation))
        )
    if missing_test:
        raise ValueError(
            "Test split has no labels for trained task(s): "
            + ", ".join(sorted(missing_test))
        )

    class_to_id = {
        str(name): int(class_id)
        for name, class_id in (class_to_id or {}).items()
    }
    if "classification" in train_tasks and not class_to_id:
        raise ValueError(
            "Classification labels are attached, but no class_to_id mapping was provided."
        )
    num_classes = max(1, max(class_to_id.values(), default=-1) + 1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    regression_weights = regression_weights or {
        "length": 5.0, "area": 0.1, "radius": 0.1
    }

    model = DXFSemanticGNN(
        metadata=train_graphs[0][1].metadata(),
        input_dims=input_dims,
        edge_dims=edge_dims,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    start_epoch = 0

    if resume_from:
        (
            start_epoch,
            best_validation_loss,
            best_state,
            epochs_without_improvement,
        ) = _restore_training_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            class_to_id=class_to_id,
            input_dims=input_dims,
            edge_dims=edge_dims,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            text_matching_trained=("text_geometry_matching" in train_tasks),
            train_graph_paths=[p for p, _ in train_graphs],
            validation_graph_paths=[p for p, _ in validation_graphs],
            test_graph_paths=[p for p, _ in test_graphs],
        )
        if start_epoch >= epochs:
            print(
                f"[RESUME] Checkpoint is already at epoch {start_epoch}; "
                f"requested total epochs={epochs}. No additional epochs will run."
            )

    checkpoint_dir = Path(output_path).parent / "checkpoints"

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_losses = []
        for _, cpu_data in train_graphs:
            data = cpu_data.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, measurement_log, _, evidence = model(data)
            loss, _ = _loss_for_graph(
                logits, measurement_log, evidence, data,
                class_weight=class_weight,
                regression_weights=regression_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_metrics = _evaluate(
            model, validation_graphs, device, class_weight, regression_weights
        )
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={sum(epoch_losses)/max(1,len(epoch_losses)):.5f} "
                f"val_loss={validation_loss:.5f} "
                f"val_len_MAE={validation_metrics['length_mae']} "
                f"val_area_MAE={validation_metrics['area_mae']} "
                f"val_radius_MAE={validation_metrics['radius_mae']} "
                f"val_text_link_F1={validation_metrics['text_link_f1']}"
            )
        if epoch % checkpoint_every == 0:
            _save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_validation_loss=best_validation_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                class_to_id=class_to_id,
                input_dims=input_dims,
                edge_dims=edge_dims,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                text_matching_trained=("text_geometry_matching" in train_tasks),
                train_graph_paths=[p for p, _ in train_graphs],
                validation_graph_paths=[p for p, _ in validation_graphs],
                test_graph_paths=[p for p, _ in test_graphs],
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best val loss={best_validation_loss:.5f}")
            break

    final_epoch = epoch if epochs > start_epoch else start_epoch
    if final_epoch > 0 and final_epoch % checkpoint_every != 0:
        _save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=final_epoch,
            model=model,
            optimizer=optimizer,
            best_validation_loss=best_validation_loss,
            best_state=best_state,
            epochs_without_improvement=epochs_without_improvement,
            class_to_id=class_to_id,
            input_dims=input_dims,
            edge_dims=edge_dims,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            text_matching_trained=("text_geometry_matching" in train_tasks),
            train_graph_paths=[p for p, _ in train_graphs],
            validation_graph_paths=[p for p, _ in validation_graphs],
            test_graph_paths=[p for p, _ in test_graphs],
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    # Evaluate test split only after selecting the checkpoint on validation.
    test_metrics = _evaluate(
        model, test_graphs, device, class_weight, regression_weights
    )
    print("TEST METRICS:", json.dumps(test_metrics, indent=2))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "class_to_id": class_to_id,
        "input_dims": input_dims,
        "edge_dims": edge_dims,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "text_matching_trained": ("text_geometry_matching" in train_tasks),
        "feature_version": 7,
        "architecture": "relation_aware_hybrid_text_match_v4_geometry_v7",
        "target_encoding": "log1p",
        "regression_targets": list(REGRESSION_TARGETS),
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
        "split": {
            "train_graphs": [p for p, _ in train_graphs],
            "validation_graphs": [p for p, _ in validation_graphs],
            "test_graphs": [p for p, _ in test_graphs],
        },
    }, output)
    return {
        "checkpoint": str(output),
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the DXF multi-task heterogeneous GNN."
    )
    parser.add_argument("--train-graphs", nargs="+", required=True)
    parser.add_argument("--val-graphs", nargs="+", required=True)
    parser.add_argument("--test-graphs", nargs="+", required=True)
    parser.add_argument("--classes", default=None, help="Optional classes.json")
    parser.add_argument("--output", default="models/dxf_semantic_gnn_multitask.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from checkpoint_latest.pt or a numbered training checkpoint.",
    )
    args = parser.parse_args()
    classes = (
        json.loads(Path(args.classes).read_text(encoding="utf-8"))
        if args.classes else {}
    )
    train_model(
        args.train_graphs,
        args.val_graphs,
        args.test_graphs,
        classes,
        args.output,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
