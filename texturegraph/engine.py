from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image as PilImage

from .errors import GraphValidationError, RenderCancelled
from .graph import Graph
from .nodes import NODE_TYPES, NodeType

# Bump when node execution semantics or the signature payload change so old
# cached masks/frames can never be mistaken for current results.
CACHE_VERSION = 2


@dataclass(frozen=True)
class FrameOutput:
    """One image written as part of a frame: node, path and resolution."""

    node_id: str
    path: str
    width: int
    height: int


@dataclass(frozen=True)
class FrameRequest:
    """A set of outputs rendered together at one point in time."""

    time: float
    outputs: tuple[FrameOutput, ...]

    @property
    def node_ids(self) -> list[str]:
        return [output.node_id for output in self.outputs]


@dataclass(frozen=True)
class FrameResult:
    time: float
    outputs: dict[str, str]
    skipped: bool = False


class RenderContext:
    def __init__(
        self,
        is_cancelled: Callable[[], bool] = lambda: False,
        time: float = 0.0,
    ) -> None:
        self.is_cancelled = is_cancelled
        self.time = time
        self.evaluated: list[str] = []
        self.cache_hits: list[str] = []

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise RenderCancelled("render was cancelled")


class EvaluationCache:
    """Persistent cache keyed by the full node signature.

    Signatures include resolution, CACHE_VERSION and the resolved (possibly
    animated) parameters, so a mask cached for frame 0 can never satisfy frame
    1 and a preview-size value can never satisfy a high-resolution request.
    """

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

    def render_image(
        self,
        output_node: str,
        context: RenderContext | None = None,
        width: int | None = None,
        height: int | None = None,
        time: float | None = None,
    ):
        value = self.evaluate(output_node, context, width, height, time)
        return to_pil_image(value)

    def evaluate(
        self,
        output_node: str,
        context: RenderContext | None = None,
        width: int | None = None,
        height: int | None = None,
        time: float | None = None,
        scope: dict[str, object] | None = None,
    ):
        context = context or RenderContext()
        if time is not None:
            context.time = time
        width = int(self.graph.width if width is None else width)
        height = int(self.graph.height if height is None else height)
        if width <= 0 or height <= 0:
            raise GraphValidationError("width and height must be positive")

        order = self.graph.topological_order(output_node)
        signatures = self._signatures(order, width, height, context.time)
        values: dict[str, object] = {}

        for node_id in order:
            context.raise_if_cancelled()
            signature = signatures[node_id]
            cached = scope.get(signature) if scope is not None else None
            source = "scope"
            if cached is None:
                cached = self.cache.get(signature)
                source = "cache"
            if cached is not None:
                values[node_id] = copy.deepcopy(cached)
                context.cache_hits.append(node_id)
                if scope is not None:
                    scope[signature] = cached
                continue

            node = self.graph.nodes[node_id]
            node_type = NODE_TYPES[node.type]
            inputs = self._prepare_inputs(node_id, node_type, values)
            params = self.graph.params_at(node_id, context.time)
            value = np.asarray(
                node_type.execute(inputs, params, width, height),
                dtype=np.float32,
            )
            expected_shape = expected_output_shape(node_type.output_type, width, height)
            if tuple(value.shape) != expected_shape:
                raise GraphValidationError(
                    f"node {node_id} returned shape {tuple(value.shape)}, expected {expected_shape}"
                )
            values[node_id] = value
            stored = value.copy()
            self.cache.store(signature, stored)
            if scope is not None:
                scope[signature] = stored
            context.evaluated.append(node_id)

        return values[output_node]

    def render_batch(
        self,
        output_nodes: list[str],
        paths: dict[str, str],
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        if set(output_nodes) != set(paths):
            raise ValueError("every output node needs exactly one output path")
        frame = FrameRequest(
            0.0,
            tuple(
                FrameOutput(node_id, paths[node_id], self.graph.width, self.graph.height)
                for node_id in output_nodes
            ),
        )
        self.render_frames([frame], is_cancelled=is_cancelled, resume=False)

    def render_frames(
        self,
        frames: list[FrameRequest],
        is_cancelled: Callable[[], bool] = lambda: False,
        resume: bool = True,
    ) -> list[FrameResult]:
        """Render and atomically publish frames.

        Each frame is an independent publish: outputs committed by earlier
        frames are never overwritten or removed.  With resume enabled (the
        default), frames whose PNGs already exist at the requested node and
        resolution are skipped, so a cancelled batch can continue without
        repeating or dropping frames and without mixing resolutions.
        """

        results: list[FrameResult] = []
        for frame in frames:
            context = RenderContext(is_cancelled=is_cancelled, time=frame.time)
            existing = {
                output.node_id: output.path
                for output in frame.outputs
                if resume and published_png_is_valid(output)
            }
            pending = [
                output for output in frame.outputs if output.node_id not in existing
            ]
            if not pending:
                results.append(FrameResult(frame.time, existing, skipped=True))
                continue

            # Share values across outputs of THIS frame (same time and same
            # output resolution) so a shared mask is computed only once, while
            # distinct resolutions still get distinct signatures.
            scope: dict[str, object] = {}
            rendered: list[tuple[FrameOutput, object]] = []
            for output in pending:
                value = self.evaluate(
                    output.node_id,
                    context,
                    output.width,
                    output.height,
                    frame.time,
                    scope,
                )
                rendered.append((output, value))
            for output, value in rendered:
                write_png_atomic(value, output.path)
                existing[output.node_id] = output.path
            results.append(
                FrameResult(
                    frame.time,
                    {output.node_id: output.path for output in frame.outputs},
                )
            )
        return results

    def _prepare_inputs(self, node_id: str, node_type: NodeType, values):
        incoming = self.graph.inputs_for(node_id)
        inputs = {}
        allowed = {port.name: port for port in node_type.inputs}
        for port in node_type.inputs:
            connection = incoming.get(port.name)
            if connection is None:
                if port.required:
                    raise GraphValidationError(
                        f"node {node_id} is missing required input {port.name}"
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
                    f"unknown output port {connection.source_port} on {connection.source_node}"
                )
            inputs[port.name] = values[connection.source_node]

        for connection in incoming.values():
            if connection.target_port not in allowed:
                raise GraphValidationError(
                    f"node {node_id} has no input port {connection.target_port}"
                )
        return inputs

    def _is_time_dependent(self, node_id: str) -> bool:
        """True for animated nodes and anything downstream of them."""
        animated_nodes = {track_node for track_node, _ in self.graph.tracks}
        outgoing: dict[str, list[str]] = {}
        for connection in self.graph.connections:
            outgoing.setdefault(connection.source_node, []).append(
                connection.target_node
            )
        seen: set[str] = set()
        pending = list(animated_nodes)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(outgoing.get(current, []))
        return node_id in seen

    def _signatures(
        self, order: list[str], width: int, height: int, time: float
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
                "params": stable_json(self.graph.params_at(node_id, time)),
                "inputs": input_signatures,
            }
            if self._is_time_dependent(node_id):
                # Explicitly stamp time on animated nodes and their downstream
                # nodes; unrelated branches stay shared across times.
                payload["time"] = float(time)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            result[node_id] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return result


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


def published_png_is_valid(output: FrameOutput) -> bool:
    """A published file counts only at the requested resolution."""
    import os

    if not os.path.exists(output.path):
        return False
    try:
        with PilImage.open(output.path) as image:
            image.load()
            return tuple(image.size) == (output.width, output.height)
    except Exception:
        return False


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
