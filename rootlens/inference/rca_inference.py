#!/usr/bin/env python3
"""Frozen GraphSAGE RCA v2 inference with no runtime fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, SAGEConv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO_ROOT / "rootlens/models/graphsage_rca_v2/model_bundle.json"


class InferenceConfigurationError(RuntimeError):
    """The frozen model bundle is incomplete or inconsistent."""


class TelemetryShapeError(ValueError):
    """A telemetry window does not match the frozen model contract."""


class RootLensGraphSAGEV2(nn.Module):
    """Node-identity-preserving architecture used by the canonical run."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float,
        mlp_hidden: int,
        node_count: int,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_count = node_count
        self.dropout_p = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(node_count * hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout_p, training=self.training)
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x = x1 + x2
        batch_size = int(batch.max().item()) + 1
        expected = batch_size * self.node_count
        if x.shape[0] != expected:
            raise TelemetryShapeError(
                f"Expected {expected} batched nodes, found {x.shape[0]}"
            )
        return self.classifier(
            x.view(batch_size, self.node_count * self.hidden_channels)
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RCAInference:
    """Load once and infer one frozen-order system telemetry window at a time."""

    def __init__(
        self,
        bundle_path: Path | str = DEFAULT_BUNDLE,
        device: str = "cpu",
    ) -> None:
        self.bundle_path = Path(bundle_path).resolve()
        self.bundle = _load_json(self.bundle_path)
        if device == "cuda" and not torch.cuda.is_available():
            raise InferenceConfigurationError("CUDA requested but unavailable")
        self.device = torch.device(device)
        self.node_order = list(self.bundle["node_order"])
        self.feature_order = list(self.bundle["feature_order"])
        self.class_to_index = {
            str(k): int(v) for k, v in self.bundle["class_to_index"].items()
        }
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self._validate_bundle_contract()
        self.model, self.edge_index, self.mean, self.std = self._load_artifacts()

    def _repo_path(self, relative: str) -> Path:
        return (REPO_ROOT / relative).resolve()

    def _validate_hash(self, key: str, hash_key: str) -> Path:
        path = self._repo_path(str(self.bundle[key]))
        if not path.is_file():
            raise InferenceConfigurationError(f"Missing artifact: {path}")
        observed = _sha256(path)
        expected = str(self.bundle[hash_key]).lower()
        if observed != expected:
            raise InferenceConfigurationError(
                f"Checksum mismatch for {path}: expected {expected}, got {observed}"
            )
        return path

    def _validate_bundle_contract(self) -> None:
        if len(self.node_order) != int(self.bundle["expected_nodes"]):
            raise InferenceConfigurationError("Bundle node count is inconsistent")
        if len(self.feature_order) != int(self.bundle["expected_node_features"]):
            raise InferenceConfigurationError("Bundle feature count is inconsistent")
        if sorted(self.class_to_index.values()) != list(range(len(self.class_to_index))):
            raise InferenceConfigurationError("Class indices must be contiguous")

    def _load_artifacts(
        self,
    ) -> tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
        checkpoint_path = self._validate_hash("checkpoint_path", "checkpoint_sha256")
        topology_path = self._validate_hash("topology_config_path", "topology_sha256")
        normalization_path = self._validate_hash(
            "normalization_path", "normalization_sha256"
        )
        graph_metadata_path = self._validate_hash(
            "graph_metadata_path", "graph_metadata_sha256"
        )
        del topology_path  # Its frozen hash is validated; edge_index comes from graph metadata.
        graph_metadata = _load_json(graph_metadata_path)
        if graph_metadata["node_order"] != self.node_order:
            raise InferenceConfigurationError("Graph and bundle node orders differ")
        if graph_metadata["node_features"] != self.feature_order:
            raise InferenceConfigurationError("Graph and bundle feature orders differ")
        edge_index = torch.tensor(graph_metadata["edge_index"], dtype=torch.long)
        if edge_index.shape != (2, int(self.bundle["expected_message_passing_edges"])):
            raise InferenceConfigurationError(
                f"Expected edge_index [2,44], got {list(edge_index.shape)}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["class_to_index"] != self.class_to_index:
            raise InferenceConfigurationError("Checkpoint class mapping differs")
        config = checkpoint["model_config"]
        model = RootLensGraphSAGEV2(
            **config,
            node_count=len(self.node_order),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(self.bundle["parameter_count"]):
            raise InferenceConfigurationError(
                f"Expected {self.bundle['parameter_count']} parameters, got {parameter_count}"
            )
        model.to(self.device).eval()
        mean = checkpoint["feature_mean"].float()
        std = checkpoint["feature_std"].float()
        normalization = _load_json(normalization_path)
        recorded_mean = torch.tensor(
            [normalization[name]["mean"] for name in self.feature_order],
            dtype=torch.float32,
        )
        recorded_std = torch.tensor(
            [normalization[name]["std"] for name in self.feature_order],
            dtype=torch.float32,
        )
        if not torch.allclose(mean, recorded_mean) or not torch.allclose(std, recorded_std):
            raise InferenceConfigurationError(
                "Checkpoint and normalization metadata differ"
            )
        if mean.shape != (len(self.feature_order),) or std.shape != mean.shape:
            raise InferenceConfigurationError("Invalid normalization tensor shape")
        if torch.any(std <= 0):
            raise InferenceConfigurationError("Normalization std must be positive")
        return model, edge_index.to(self.device), mean.to(self.device), std.to(self.device)

    def build_feature_tensor(
        self,
        telemetry_window: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if isinstance(telemetry_window, Mapping):
            by_service = telemetry_window
        else:
            by_service = {
                str(row.get("service")): row for row in telemetry_window
            }
        observed = set(by_service)
        expected = set(self.node_order)
        if observed != expected:
            raise TelemetryShapeError(
                f"Service set mismatch; missing={sorted(expected-observed)}, "
                f"unexpected={sorted(observed-expected)}"
            )
        values: list[list[float]] = []
        zero_fill = {"latency_ms", "error_rate"}
        for service in self.node_order:
            row = by_service[service]
            node: list[float] = []
            for feature in self.feature_order:
                value = row.get(feature)
                if value is None and feature in zero_fill:
                    value = 0.0
                if value is None:
                    raise TelemetryShapeError(
                        f"{service}.{feature} is missing; frozen policy does not impute it"
                    )
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise TelemetryShapeError(
                        f"{service}.{feature} is not numeric: {value!r}"
                    ) from exc
                if not torch.isfinite(torch.tensor(number)):
                    raise TelemetryShapeError(f"{service}.{feature} is non-finite")
                node.append(number)
            values.append(node)
        tensor = torch.tensor(values, dtype=torch.float32)
        expected_shape = (len(self.node_order), len(self.feature_order))
        if tuple(tensor.shape) != expected_shape:
            raise TelemetryShapeError(
                f"Expected feature tensor {expected_shape}, got {tuple(tensor.shape)}"
            )
        return tensor

    def predict(
        self,
        telemetry_window: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        raw_x = self.build_feature_tensor(telemetry_window).to(self.device)
        x = (raw_x - self.mean) / self.std
        batch = torch.zeros(len(self.node_order), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x, self.edge_index, batch)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()
        predicted_index = int(torch.argmax(probabilities).item())
        probability_map = {
            self.index_to_class[index]: float(probabilities[index].item())
            for index in range(len(probabilities))
        }
        return {
            "predicted_class": self.index_to_class[predicted_index],
            "confidence": probability_map[self.index_to_class[predicted_index]],
            "probabilities": probability_map,
            "graph_shape": list(raw_x.shape),
            "device": str(self.device),
        }
