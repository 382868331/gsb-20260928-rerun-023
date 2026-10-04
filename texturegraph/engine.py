from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image as PilImage
from PIL.PngImagePlugin import PngInfo

from .errors import GraphValidationError, RenderCancelled
from .graph import Graph
from .nodes import NODE_TYPES, NodeType, _neighborhood_box, neighborhood_radius

# Bump when node execution semantics or the signature payload change so old
# cached masks/frames can never be mistaken for current results.
CACHE_VERSION = 4


@dataclass(frozen=True)
class FrameOutput:
    """One image written as part of a frame: node, path and resolution."""

    node_id: str
    path: str
    width: int
    height: int
    label: str = ""


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
    reused: dict[str, str] = field(default_factory=dict)
    rendered: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


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

    def evaluate_region(
        self,
        output_node: str,
        region: tuple[int, int, int, int],
        context: RenderContext | None = None,
        width: int | None = None,
        height: int | None = None,
        time: float | None = None,
        scope: dict[str, object] | None = None,
        overrides: dict[str, tuple[str, object]] | None = None,
    ) -> np.ndarray:
        """Evaluate one output only inside ``region`` = (x0, y0, x1, y1).

        Neighbourhood nodes (e.g. box blur) transparently pull the full halo
        of pixels across the region border, so boundary pixels are identical
        to a full-canvas recompute.  ``overrides`` maps ``node_id`` to
        ``{param: value}`` replacements applied to a throwaway copy of the
        graph; unchanged upstream nodes keep their signatures and cache.
        """
        return evaluate_region(
            self,
            output_node,
            region,
            context=context,
            width=width,
            height=height,
            time=time,
            scope=scope,
            overrides=overrides,
        )

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

        Each frame is an independent publish.  With resume enabled (the
        default), a PNG is reused only when its embedded signature matches the
        output's current node, size, time, parameters and upstream signatures,
        and its decoded pixels match the digest captured at publication.
        """

        results: list[FrameResult] = []
        for frame in frames:
            context = RenderContext(is_cancelled=is_cancelled, time=frame.time)
            reused: dict[str, str] = {}
            rendered_paths: dict[str, str] = {}
            reasons: dict[str, str] = {}
            pending: list[FrameOutput] = []
            for output in frame.outputs:
                expected_signature = self.signature_for(
                    output.node_id,
                    output.width,
                    output.height,
                    frame.time,
                )
                if resume and published_png_is_valid(output, expected_signature):
                    reused[output.node_id] = output.path
                    reasons[output.node_id] = (
                        "reused published PNG: size and complete dependency "
                        "signature match, and pixel digest is intact"
                    )
                else:
                    pending.append(output)
                    reasons[output.node_id] = (
                        "rendered: file is missing or its size, dependency "
                        "signature, or pixel digest does not match current input"
                    )
            if not pending:
                results.append(
                    FrameResult(
                        frame.time,
                        {output.node_id: output.path for output in frame.outputs},
                        skipped=True,
                        reused=reused,
                        reasons=reasons,
                    )
                )
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
            context.raise_if_cancelled()
            for output, value in rendered:
                signature = self.signature_for(
                    output.node_id,
                    output.width,
                    output.height,
                    frame.time,
                )
                write_png_atomic(value, output.path, signature, label=output.label)
                rendered_paths[output.node_id] = output.path
            all_outputs = {}
            all_outputs.update(reused)
            all_outputs.update(rendered_paths)
            results.append(
                FrameResult(
                    frame.time,
                    all_outputs,
                    skipped=False,
                    reused=reused,
                    rendered=rendered_paths,
                    reasons=reasons,
                )
            )
        return results

    def signature_for(
        self,
        output_node: str,
        width: int,
        height: int,
        time: float,
    ) -> str:
        order = self.graph.topological_order(output_node)
        return self._signatures(order, width, height, time)[output_node]

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
                "node_id": node_id,
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


def _clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _crop_or_pad(
    value: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Extract ``target_box`` from a value known over ``source_box``.

    Missing strips come from edge replication, which is exactly how the
    box blur pads the canvas border.
    """
    width, height = canvas_size
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    channels = value.shape[2:]
    result = np.empty((ty1 - ty0, tx1 - tx0) + channels, dtype=np.float32)
    for row in range(ty0, ty1):
        source_row = min(max(row, sy0), sy1 - 1)
        for col in range(tx0, tx1):
            source_col = min(max(col, sx0), sx1 - 1)
            result[row - ty0, col - tx0] = value[
                source_row - sy0, source_col - sx0
            ]
    return result


def _exact_crop(
    value: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
) -> np.ndarray:
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    return value[
        ty0 - sy0 : ty1 - sy0,
        tx0 - sx0 : tx1 - sx0,
    ].copy()


def evaluate_region(
    engine: "TextureEngine",
    output_node: str,
    region: tuple[int, int, int, int],
    context: RenderContext | None = None,
    width: int | None = None,
    height: int | None = None,
    time: float | None = None,
    scope: dict[str, object] | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
) -> np.ndarray:
    """Windowed evaluation with exact cross-boundary neighbourhood halos.

    The requested region is dilated upstream through every neighbourhood
    node (using its current parameter-derived radius) before any node runs,
    so blur output at a region edge averages the full (2r+1)x(2r+1)
    neighbourhood, matching a full-canvas recompute pixel for pixel.
    """
    graph = engine.graph
    overrides = overrides or {}
    if overrides:
        graph = copy.deepcopy(graph)
        for node_id, params in overrides.items():
            if node_id not in graph.nodes:
                raise GraphValidationError(f"override for unknown node: {node_id}")
            for name, value in params.items():
                graph.set_param(node_id, name, value)
    region_engine = TextureEngine(graph, engine.cache)

    context = context or RenderContext()
    if time is not None:
        context.time = time
    pixel_width = int(graph.width if width is None else width)
    pixel_height = int(graph.height if height is None else height)
    if pixel_width <= 0 or pixel_height <= 0:
        raise GraphValidationError("width and height must be positive")
    x0, y0, x1, y1 = _clip_box(
        (int(region[0]), int(region[1]), int(region[2]), int(region[3])),
        pixel_width,
        pixel_height,
    )
    if x1 <= x0 or y1 <= y0:
        raise GraphValidationError("region must contain at least one pixel")

    order = graph.topological_order(output_node)
    signatures = region_engine._signatures(
        order, pixel_width, pixel_height, context.time
    )

    # Back-propagate the needed window from output to sources.  Each
    # neighbourhood node dilates by its radius (clamped to the canvas); each
    # pointwise node passes the same window on to every input.
    windows: dict[str, tuple[int, int, int, int]] = {output_node: (x0, y0, x1, y1)}
    pending = [output_node]
    while pending:
        node_id = pending.pop()
        box = windows[node_id]
        node = graph.nodes[node_id]
        node_type = NODE_TYPES[node.type]
        params = graph.params_at(node_id, context.time)
        radius = neighborhood_radius(node.type, params)
        needed = _clip_box(
            (box[0] - radius, box[1] - radius, box[2] + radius, box[3] + radius),
            pixel_width,
            pixel_height,
        )
        for connection in graph.inputs_for(node_id).values():
            source = connection.source_node
            if source in windows:
                old = windows[source]
                merged = (
                    min(old[0], needed[0]),
                    min(old[1], needed[1]),
                    max(old[2], needed[2]),
                    max(old[3], needed[3]),
                )
                windows[source] = merged
            else:
                windows[source] = needed
                pending.append(source)

    values: dict[str, np.ndarray] = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for node_id in order:
        context.raise_if_cancelled()
        box = windows[node_id]
        signature = signatures[node_id]
        node = graph.nodes[node_id]
        node_type = NODE_TYPES[node.type]
        params = graph.params_at(node_id, context.time)
        cached = scope.get(signature) if scope is not None else None
        source_kind = "scope"
        if cached is None:
            cached = region_engine.cache.get(signature)
            source_kind = "cache"
        if cached is not None and tuple(np.asarray(cached).shape[:2]) == (
            pixel_height,
            pixel_width,
        ):
            full = np.asarray(cached, dtype=np.float32)
            values[node_id] = full[box[1]:box[3], box[0]:box[2]].copy()
            boxes[node_id] = box
            context.cache_hits.append(node_id)
            if scope is not None:
                scope[signature] = cached
            continue

        inputs = {}
        for port in node_type.inputs:
            connection = graph.inputs_for(node_id).get(port.name)
            if connection is None:
                if port.required:
                    raise GraphValidationError(
                        f"node {node_id} is missing required input {port.name}"
                    )
                continue
            source_node = connection.source_node
            inputs[port.name] = _exact_crop(
                values[source_node], boxes[source_node], box
            )

        value: np.ndarray
        if node.type == "uv_coordinates":
            xs = (np.arange(box[0], box[2], dtype=np.float32) + 0.5) / pixel_width * 2.0 - 1.0
            ys = (np.arange(box[1], box[3], dtype=np.float32) + 0.5) / pixel_height * 2.0 - 1.0
            xx, yy = np.meshgrid(xs, ys)
            value = np.stack([xx, yy], axis=-1).astype(np.float32)
        elif node.type == "solid_color":
            value = node_type.execute(inputs, params, box[2] - box[0], box[3] - box[1])
        elif node.type == "transform_coordinates":
            # Pointwise remap; the already-fetched input window equals this
            # node's window, so the generic execution path is exact.
            value = node_type.execute(
                inputs, params, box[2] - box[0], box[3] - box[1]
            )
        elif node.type == "box_blur":
            radius = neighborhood_radius(node.type, params)
            source_node = graph.inputs_for(node_id)["image"].source_node
            dilated = _clip_box(
                (
                    box[0] - radius,
                    box[1] - radius,
                    box[2] + radius,
                    box[3] + radius,
                ),
                pixel_width,
                pixel_height,
            )
            padded = _crop_or_pad(
                values[source_node],
                boxes[source_node],
                dilated,
                (pixel_width, pixel_height),
            )
            blurred = _neighborhood_box(padded, radius)
            value = blurred[
                box[1] - dilated[1] : box[3] - dilated[1],
                box[0] - dilated[0] : box[2] - dilated[0],
            ].astype(np.float32)
        else:
            value = node_type.execute(
                inputs, params, box[2] - box[0], box[3] - box[1]
            )
        expected = expected_output_shape(
            node_type.output_type, box[2] - box[0], box[3] - box[1]
        )
        if tuple(value.shape) != expected:
            raise GraphValidationError(
                f"node {node_id} returned shape {tuple(value.shape)}, expected {expected}"
            )
        values[node_id] = np.asarray(value, dtype=np.float32)
        boxes[node_id] = box
        context.evaluated.append(node_id)

    return values[output_node]


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


def published_png_is_valid(output: FrameOutput, expected_signature: str) -> bool:
    """Validate resolution, dependency signature and decoded pixel content."""
    import os

    if not os.path.exists(output.path):
        return False
    try:
        with PilImage.open(output.path) as image:
            image.load()
            if tuple(image.size) != (output.width, output.height):
                return False
            if image.info.get("TextureGraph-Signature") != expected_signature:
                return False
            if output.label and image.info.get("TextureGraph-Label") != output.label:
                return False
            pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            digest = image_digest(pixels)
            return image.info.get("TextureGraph-Content-SHA256") == digest
    except Exception:
        return False


def image_digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def write_png_atomic(value, path: str, signature: str, label: str = "") -> None:
    import os

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        pixels = np.rint(
            np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        image = PilImage.fromarray(pixels, mode="RGBA")
        metadata = PngInfo()
        metadata.add_text("TextureGraph-Signature", signature)
        metadata.add_text("TextureGraph-Content-SHA256", image_digest(pixels))
        if label:
            metadata.add_text("TextureGraph-Label", label)
        image.save(
            temporary_path,
            format="PNG",
            compress_level=6,
            pnginfo=metadata,
        )
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
