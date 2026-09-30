from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image as PilImage

from .errors import GraphValidationError, RenderCancelled
from .graph import Graph
from .nodes import NODE_TYPES, NodeType

CACHE_VERSION = 2


@dataclass(frozen=True)
class RenderTarget:
    output_node: str
    path: str
    frame: int = 0
    width: int | None = None
    height: int | None = None


@dataclass
class RenderContext:
    is_cancelled: Callable[[], bool] = lambda: False
    evaluated: list[str] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise RenderCancelled("render was cancelled")


class EvaluationCache:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get(self, signature: str) -> object | None:
        return self._values.get(signature)

    def store(self, signature: str, value: object) -> None:
        self._values[signature] = value


class TextureEngine:
    def __init__(self, graph: Graph, cache: EvaluationCache | None = None) -> None:
        graph.validate_references()
        self.graph = graph
        self.cache = cache or EvaluationCache()

    def render_image(self, output_node: str, context: RenderContext | None = None):
        value = self.evaluate(output_node, context)
        return to_pil_image(value)

    def evaluate(
        self,
        output_node: str,
        context: RenderContext | None = None,
        frame: int = 0,
        width: int | None = None,
        height: int | None = None,
    ):
        context = context or RenderContext()
        width = int(width or self.graph.width)
        height = int(height or self.graph.height)
        if width <= 0 or height <= 0:
            raise GraphValidationError("render width and height must be positive")
        order = self.graph.topological_order(output_node)
        signatures = self._signatures(order, frame, width, height)
        values: dict[str, object] = {}

        for node_id in order:
            context.raise_if_cancelled()
            signature = signatures[node_id]
            cached = self.cache.get(signature)
            if cached is not None:
                values[node_id] = copy.deepcopy(cached)
                context.cache_hits.append(node_id)
                continue

            node = self.graph.nodes[node_id]
            node_type = NODE_TYPES[node.type]
            inputs = self._prepare_inputs(node_id, node_type, values)
            value = np.asarray(
                node_type.execute(
                    inputs,
                    self.graph.params_at(node_id, frame),
                    width,
                    height,
                ),
                dtype=np.float32,
            )
            expected_shape = expected_output_shape(
                node_type.output_type,
                width,
                height,
            )
            if tuple(value.shape) != expected_shape:
                raise GraphValidationError(
                    f"node '{node_id}' returned shape {tuple(value.shape)}, expected {expected_shape}"
                )
            values[node_id] = value
            self.cache.store(signature, value.copy())
            context.evaluated.append(node_id)

        return values[output_node]

    def render_batch(
        self,
        output_nodes: list[str] | list[RenderTarget],
        paths: dict[str, str] | None = None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        targets = normalize_targets(output_nodes, paths, self.graph)
        for target in targets:
            if target.output_node not in self.graph.nodes:
                raise GraphValidationError(f"unknown output node: {target.output_node}")
        context = RenderContext(is_cancelled=is_cancelled)
        rendered: list[tuple[RenderTarget, object]] = []
        for target in targets:
            rendered.append(
                (
                    target,
                    self.evaluate(
                        target.output_node,
                        context,
                        frame=target.frame,
                        width=target.width,
                        height=target.height,
                    ),
                )
            )
        context.raise_if_cancelled()
        for target, value in rendered:
            write_png_atomic(value, target.path)

    def _prepare_inputs(self, node_id: str, node_type: NodeType, values):
        incoming = self.graph.inputs_for(node_id)
        inputs = {}
        allowed = {port.name: port for port in node_type.inputs}
        for port in node_type.inputs:
            connection = incoming.get(port.name)
            if connection is None:
                if port.required:
                    raise GraphValidationError(
                        f"node '{node_id}' is missing required input '{port.name}'"
                    )
                continue
            source = self.graph.nodes[connection.source_node]
            source_type = NODE_TYPES[source.type]
            if source_type.output_type != port.type:
                raise GraphValidationError(
                    f"cannot connect {source_type.output_type} to {port.type}: "
                    f"{connection.source_node}.{connection.source_port} -> "
                    f"{node_id}.{port.name}"
                )
            if connection.source_port != "value":
                raise GraphValidationError(
                    f"unknown output port '{connection.source_port}' on '{connection.source_node}'"
                )
            inputs[port.name] = values[connection.source_node]

        for connection in incoming.values():
            if connection.target_port not in allowed:
                raise GraphValidationError(
                    f"node '{node_id}' has no input port '{connection.target_port}'"
                )
        return inputs

    def _signatures(
        self,
        order: list[str],
        frame: int,
        width: int,
        height: int,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for node_id in order:
            node = self.graph.nodes[node_id]
            node_type = NODE_TYPES[node.type]
            input_signatures = {}
            for port_name, connection in self.graph.inputs_for(node_id).items():
                input_signatures[port_name] = result[connection.source_node]
            payload = {
                "cache_version": CACHE_VERSION,
                "size": [width, height],
                "type": node.type,
                "output_type": node_type.output_type,
                "params": stable_json(self.graph.params_at(node_id, frame)),
                "inputs": input_signatures,
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            result[node_id] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return result


def normalize_targets(
    output_nodes: list[str] | list[RenderTarget],
    paths: dict[str, str] | None,
    graph: Graph,
) -> list[RenderTarget]:
    if output_nodes and all(isinstance(item, RenderTarget) for item in output_nodes):
        if paths is not None:
            raise ValueError("paths cannot be combined with RenderTarget entries")
        return list(output_nodes)
    if paths is None or set(output_nodes) != set(paths):
        raise ValueError("every output node needs exactly one output path")
    return [
        RenderTarget(str(node_id), paths[str(node_id)])
        for node_id in output_nodes
    ]


def stable_json(value) -> object:
    if isinstance(value, dict):
        return {str(key): stable_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [stable_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def expected_output_shape(output_type: str, width: int, height: int) -> tuple[int, ...]:
    if output_type == "image":
        return (height, width, 4)
    if output_type == "mask":
        return (height, width)
    if output_type == "coordinates":
        return (height, width, 2)
    raise GraphValidationError(f"unsupported output type: {output_type}")


def to_pil_image(value) -> PilImage.Image:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    channels = np.rint(clipped * 255.0).astype(np.uint8)
    return PilImage.fromarray(channels, mode="RGBA")


def write_png_atomic(value, path: str) -> None:
    import os

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        image = to_pil_image(value)
        image.save(temporary_path, format="PNG", compress_level=6)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
