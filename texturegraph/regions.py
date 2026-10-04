"""Local region candidates and zoned (partial) publishing.

The normal engine always recomputes a whole image.  This module adds:

* region candidates: recompute one rectangle of an output using parameter
  overrides, then inspect the result without publishing;
* zoned publish: adopt a candidate for one region while every other region
  keeps its previously published pixels and their recorded source;
* resumable partial state: the adopted ranges and the exact input versions
  they came from are persisted, so a same-named same-sized input change or a
  damaged cache forces the affected range to be judged and recomputed again.

Neighbourhood nodes (box_blur) advertise their radius; every compute window
is dilated through such a node so an output pixel on either side of a region
border still sees its complete real neighbourhood.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .engine import (
    CACHE_VERSION,
    EvaluationCache,
    expected_output_shape,
    image_digest,
    stable_json,
    write_png_atomic,
)
from .errors import GraphValidationError
from .graph import Graph
from .nodes import (
    NODE_TYPES,
    box_blur_window,
    crop_box,
    neighborhood_radius,
)

REGION_STATE_VERSION = 1


class RegionError(ValueError):
    """Raised for invalid regions or ambiguous candidate adoption."""


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    top, left, bottom, right = box
    top = max(0, min(top, height))
    left = max(0, min(left, width))
    bottom = max(0, min(bottom, height))
    right = max(0, min(right, width))
    if top >= bottom or left >= right:
        return None
    return (top, left, bottom, right)


def box_intersect(box, other):
    top = max(box[0], other[0])
    left = max(box[1], other[1])
    bottom = min(box[2], other[2])
    right = min(box[3], other[3])
    if top >= bottom or left >= right:
        return None
    return (top, left, bottom, right)


def boxes_overlap(box, other) -> bool:
    return box_intersect(box, other) is not None


@dataclass(frozen=True)
class RegionBox:
    """Half-open pixel rectangle [top, bottom) x [left, right)."""

    top: int
    left: int
    bottom: int
    right: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.top, self.left, self.bottom, self.right)

    @classmethod
    def from_tuple(cls, box: Iterable[int]) -> "RegionBox":
        top, left, bottom, right = (int(value) for value in box)
        return cls(top, left, bottom, right)

    def __post_init__(self) -> None:
        if self.top < 0 or self.left < 0:
            raise RegionError("region origin must be non-negative")
        if self.bottom <= self.top or self.right <= self.left:
            raise RegionError("region must be non-empty")

    def overlaps(self, other: "RegionBox") -> bool:
        return boxes_overlap(self.as_tuple(), other.as_tuple())

    def to_list(self) -> list[int]:
        return [self.top, self.left, self.bottom, self.right]


@dataclass(frozen=True)
class RegionCandidate:
    """A locally computed, not-yet-published result for one output rectangle."""

    candidate_id: str
    output_node: str
    box: RegionBox
    pixels: np.ndarray
    time: float
    width: int
    height: int
    overrides: dict[tuple[str, str], Any]
    base_signature: str
    override_signature: str
    verified: bool
    digest: str


def _params_with_overrides(
    graph: Graph,
    node_id: str,
    time: float,
    overrides: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    params = graph.params_at(node_id, time)
    for (override_node, param_name), value in overrides.items():
        if override_node == node_id:
            params[param_name] = value
    return params


def _time_scoped_nodes(graph: Graph) -> set[str]:
    animated_nodes = {track_node for track_node, _ in graph.tracks}
    outgoing: dict[str, list[str]] = {}
    for connection in graph.connections:
        outgoing.setdefault(connection.source_node, []).append(
            connection.target_node
        )
    scoped: set[str] = set()
    pending = list(animated_nodes)
    while pending:
        current = pending.pop()
        if current in scoped:
            continue
        scoped.add(current)
        pending.extend(outgoing.get(current, []))
    return scoped


def compute_signatures(
    graph: Graph,
    order: list[str],
    width: int,
    height: int,
    time: float,
    overrides: dict[tuple[str, str], Any] | None = None,
) -> dict[str, str]:
    """Node signatures, optionally with local parameter overrides layered on."""
    overrides = overrides or {}
    time_scoped = _time_scoped_nodes(graph)
    result: dict[str, str] = {}
    for node_id in order:
        node = graph.nodes[node_id]
        node_type = NODE_TYPES[node.type]
        input_signatures = {
            port_name: result[connection.source_node]
            for port_name, connection in graph.inputs_for(node_id).items()
        }
        payload = {
            "cache_version": CACHE_VERSION,
            "node_id": node_id,
            "size": [width, height],
            "type": node.type,
            "output_type": node_type.output_type,
            "params": stable_json(
                _params_with_overrides(graph, node_id, time, overrides)
            ),
            "inputs": input_signatures,
        }
        if node_id in time_scoped:
            payload["time"] = float(time)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        result[node_id] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return result


def _dilate(
    box: tuple[int, int, int, int], radius: int, width: int, height: int
) -> tuple[int, int, int, int]:
    top, left, bottom, right = box
    return clamp_box(
        (top - radius, left - radius, bottom + radius, right + radius),
        width,
        height,
    )


def _union(boxes: list[tuple[int, int, int, int]]):
    top = min(box[0] for box in boxes)
    left = min(box[1] for box in boxes)
    bottom = max(box[2] for box in boxes)
    right = max(box[3] for box in boxes)
    return (top, left, bottom, right)


def _uv_window(box, width: int, height: int) -> np.ndarray:
    top, left, bottom, right = box
    x = (np.arange(left, right, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    y = (np.arange(top, bottom, dtype=np.float32) + 0.5) / height * 2.0 - 1.0
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def _execute_window(
    graph: Graph,
    node_id: str,
    node_type,
    inputs: dict[str, np.ndarray],
    params: dict[str, Any],
    out_box: tuple[int, int, int, int],
    input_origins: dict[str, tuple[int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    top, left, bottom, right = out_box
    crop_width = right - left
    crop_height = bottom - top
    if node_type.name == "uv_coordinates":
        return _uv_window(out_box, width, height)
    if node_type.name == "box_blur":
        return box_blur_window(
            inputs["image"], params, out_box, input_origins["image"]
        )
    value = np.asarray(
        node_type.execute(inputs, params, crop_width, crop_height),
        dtype=np.float32,
    )
    expected = expected_output_shape(node_type.output_type, crop_width, crop_height)
    if tuple(value.shape) != expected:
        raise GraphValidationError(
            f"node {node_id} returned shape {tuple(value.shape)}, expected {expected}"
        )
    return value


class LocalEvaluator:
    """Evaluates rectangles of an output, dilating windows at halo nodes."""

    def __init__(
        self,
        graph: Graph,
        width: int,
        height: int,
        time: float,
        overrides: dict[tuple[str, str], Any] | None = None,
        cache: EvaluationCache | None = None,
    ) -> None:
        graph.validate_references()
        self.graph = graph
        self.width = width
        self.height = height
        self.time = time
        self.overrides = dict(overrides or {})
        self.cache = cache or EvaluationCache()
        # Persistable store of computed crops keyed by node signature.
        self.crop_cache: dict[str, tuple[tuple[int, int, int, int], np.ndarray]] = {}
        self.evaluated: list[str] = []
        self.cache_hits: list[str] = []

    def _params(self, node_id: str) -> dict[str, Any]:
        return _params_with_overrides(
            self.graph, node_id, self.time, self.overrides
        )

    def _needs(
        self,
        order: list[str],
        output_node: str,
        target: tuple[int, int, int, int],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Back-propagate the output window.

        A node's stored/compute window is dilated by its own neighbourhood
        radius before requesting its inputs, so every produced pixel (including
        pixels on the border of the requested region) sees a complete halo.
        """
        needs: dict[str, tuple[int, int, int, int]] = {output_node: target}
        incoming = {node_id: self.graph.inputs_for(node_id) for node_id in order}
        for node_id in reversed(order):
            if node_id not in needs:
                continue
            node_type = NODE_TYPES[self.graph.nodes[node_id].type]
            radius = neighborhood_radius(node_type.name, self._params(node_id))
            wanted = _dilate(needs[node_id], radius, self.width, self.height)
            for connection in incoming[node_id].values():
                source = connection.source_node
                if source in needs:
                    wanted = _union([needs[source], wanted])
                needs[source] = wanted
        return needs

    def _cached_slice(
        self, signature: str, wanted: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        if signature in self.crop_cache:
            source_box, array = self.crop_cache[signature]
            if box_intersect(source_box, wanted) == wanted:
                return crop_box(array, source_box, wanted)
            return None
        full = self.cache.get(signature)
        if full is not None:
            array = np.asarray(full, dtype=np.float32)
            self.crop_cache[signature] = (
                (0, 0, self.height, self.width),
                array,
            )
            return crop_box(
                array, (0, 0, self.height, self.width), wanted
            )
        return None

    def evaluate_region(
        self,
        output_node: str,
        target: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]], dict[str, str]]:
        target = target or (0, 0, self.height, self.width)
        target = clamp_box(target, self.width, self.height)
        if target is None:
            raise RegionError("target region is outside the canvas")
        order = self.graph.topological_order(output_node)
        signatures = compute_signatures(
            self.graph,
            order,
            self.width,
            self.height,
            self.time,
            self.overrides,
        )
        needs = self._needs(order, output_node, target)
        stored: dict[str, tuple[tuple[int, int, int, int], np.ndarray]] = {}
        incoming = {node_id: self.graph.inputs_for(node_id) for node_id in order}

        for node_id in order:
            if node_id not in needs:
                continue
            node = self.graph.nodes[node_id]
            node_type = NODE_TYPES[node.type]
            inputs_for = incoming[node_id]
            compute_box = needs[node_id]

            signature = signatures[node_id]
            cached = self._cached_slice(signature, compute_box)
            if cached is not None:
                stored[node_id] = (compute_box, cached)
                self.cache_hits.append(node_id)
                continue

            own_radius = neighborhood_radius(node_type.name, self._params(node_id))
            inputs: dict[str, np.ndarray] = {}
            input_origins: dict[str, tuple[int, int]] = {}
            for port in node_type.inputs:
                connection = inputs_for[port.name]
                source = connection.source_node
                source_box, source_array = stored[source]
                wanted = _dilate(
                    compute_box, own_radius, self.width, self.height
                )
                inputs[port.name] = crop_box(source_array, source_box, wanted)
                input_origins[port.name] = (wanted[0], wanted[1])
            value = _execute_window(
                self.graph,
                node_id,
                node_type,
                inputs,
                self._params(node_id),
                compute_box,
                input_origins,
                self.width,
                self.height,
            )
            stored[node_id] = (compute_box, value)
            self.crop_cache[signature] = (compute_box, value.copy())
            self.evaluated.append(node_id)

        out_box, value = stored[output_node]
        return crop_box(value, out_box, target), needs, signatures


def evaluate_full(
    graph: Graph,
    output_node: str,
    width: int,
    height: int,
    time: float,
    overrides: dict[tuple[str, str], Any] | None = None,
    cache: EvaluationCache | None = None,
) -> np.ndarray:
    """Whole-canvas evaluation with parameter overrides (reference path)."""
    overrides = dict(overrides or {})
    order = graph.topological_order(output_node)
    signatures = compute_signatures(
        graph, order, width, height, time, overrides
    )
    cache = cache or EvaluationCache()
    values: dict[str, np.ndarray] = {}
    for node_id in order:
        signature = signatures[node_id]
        cached = cache.get(signature)
        if cached is not None:
            values[node_id] = np.asarray(cached, dtype=np.float32)
            continue
        node = graph.nodes[node_id]
        node_type = NODE_TYPES[node.type]
        inputs = {
            port.name: values[graph.inputs_for(node_id)[port.name].source_node]
            for port in node_type.inputs
        }
        value = np.asarray(
            node_type.execute(
                inputs,
                _params_with_overrides(graph, node_id, time, overrides),
                width,
                height,
            ),
            dtype=np.float32,
        )
        values[node_id] = value
        cache.store(signature, value.copy())
    return values[output_node]


def _pixel_identical(a: np.ndarray, b: np.ndarray) -> bool:
    a8 = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    b8 = np.rint(np.clip(b, 0.0, 1.0) * 255.0).astype(np.uint8)
    return bool(np.array_equal(a8, b8))


def changed_nodes(
    graph: Graph,
    output_node: str,
    width: int,
    height: int,
    time: float,
    overrides: dict[tuple[str, str], Any] | None = None,
) -> list[str]:
    """Nodes whose signature actually differs under the overrides."""
    order = graph.topological_order(output_node)
    base = compute_signatures(graph, order, width, height, time)
    changed_graph = compute_signatures(
        graph, order, width, height, time, overrides or {}
    )
    return [node_id for node_id in order if base[node_id] != changed_graph[node_id]]


def influence_box(
    graph: Graph,
    output_node: str,
    width: int,
    height: int,
    time: float,
    overrides: dict[tuple[str, str], Any] | None = None,
    tolerance: float = 1.0 / 510.0,
) -> tuple[int, int, int, int] | None:
    """Bounding box of pixels that actually change under the overrides.

    Returns None when the overrides change nothing visible.  This is the
    judged "affected range", independent of the rectangle the user asked to
    publish.
    """
    before = evaluate_full(graph, output_node, width, height, time)
    after = evaluate_full(
        graph, output_node, width, height, time, overrides or {}
    )
    if _pixel_identical(before, after):
        return None
    difference = np.any(np.abs(before - after) > tolerance, axis=-1)
    rows = np.any(difference, axis=1)
    cols = np.any(difference, axis=0)
    top = int(np.argmax(rows))
    bottom = int(height - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(width - np.argmax(cols[::-1]))
    return (top, left, bottom, right)


@dataclass(frozen=True)
class Zone:
    candidate_id: str
    output_node: str
    box: RegionBox
    time: float
    width: int
    height: int
    override_signature: str
    digest: str
    source_path: str

@dataclass
class RegionWorkspace:
    graph: Graph
    output_node: str
    output_path: str
    width: int
    height: int
    time: float = 0.0
    cache: EvaluationCache = field(default_factory=EvaluationCache)
    state_path: str | None = None
    base_signature: str | None = None
    base_digest: str | None = None
    candidates: dict[str, RegionCandidate] = field(default_factory=dict)
    zones: list[Zone] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.graph.validate_references()
        if self.output_node not in self.graph.nodes:
            raise GraphValidationError('unknown output node: ' + self.output_node)
        if self.width <= 0 or self.height <= 0:
            raise GraphValidationError('width and height must be positive')
        order = self.graph.topological_order(self.output_node)
        signatures = compute_signatures(
            self.graph, order, self.width, self.height, self.time
        )
        self._order = order
        self._base_signatures = signatures
        if self.base_signature is None:
            self.base_signature = signatures[self.output_node]
        if self.state_path is None:
            self.state_path = self.output_path + '.zones/state.json'

    @property
    def zones_directory(self) -> str:
        return self.output_path + '.zones'

    def create_candidate(self, candidate_id, box, overrides=None):
        if candidate_id in self.candidates:
            raise RegionError('duplicate candidate id: ' + candidate_id)
        region = box if isinstance(box, RegionBox) else RegionBox.from_tuple(box)
        clamped = clamp_box(region.as_tuple(), self.width, self.height)
        if clamped is None:
            raise RegionError('candidate region is outside the canvas')
        if clamped != region.as_tuple():
            raise RegionError('candidate region extends beyond the canvas')
        overrides = self._checked_overrides(dict(overrides or {}))
        local = LocalEvaluator(
            self.graph, self.width, self.height, self.time, overrides, self.cache
        )
        pixels, _, override_signatures = local.evaluate_region(
            self.output_node, region.as_tuple()
        )
        reference = evaluate_full(
            self.graph, self.output_node, self.width, self.height, self.time,
            overrides, EvaluationCache(),
        )
        top, left, bottom, right = region.as_tuple()
        if not _pixel_identical(pixels, reference[top:bottom, left:right]):
            raise RegionError(
                'local region result does not match the full recompute '
                'pixel by pixel (candidate ' + candidate_id + ')'
            )
        digest = image_digest(
            np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        candidate = RegionCandidate(
            candidate_id=candidate_id, output_node=self.output_node, box=region,
            pixels=pixels, time=self.time, width=self.width, height=self.height,
            overrides=overrides,
            base_signature=self._base_signatures[self.output_node],
            override_signature=override_signatures[self.output_node],
            verified=True, digest=digest,
        )
        self.candidates[candidate_id] = candidate
        return candidate

    def _checked_overrides(self, overrides):
        for node_id, _ in overrides:
            if node_id not in self.graph.nodes:
                raise GraphValidationError('override targets unknown node: ' + node_id)
        return overrides

    def discard_candidate(self, candidate_id: str) -> None:
        if candidate_id not in self.candidates:
            raise RegionError('unknown candidate: ' + candidate_id)
        del self.candidates[candidate_id]

    def _zone_path(self, candidate_id: str) -> str:
        return os.path.join(
            self.zones_directory, self.output_node, candidate_id + '.png'
        )

    def _overlapping_pending(self, box: RegionBox):
        return [
            cid for cid, candidate in self.candidates.items()
            if candidate.box.overlaps(box)
        ]

    def publish_candidate(self, candidate_id: str, replace=()):
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise RegionError('unknown candidate: ' + candidate_id)
        replaced = set(replace)
        for name in replaced:
            if name not in self.candidates:
                raise RegionError('cannot replace unknown candidate: ' + name)
        pending_overlap = [
            name for name in self._overlapping_pending(candidate.box)
            if name != candidate_id
        ]
        unresolved = [name for name in pending_overlap if name not in replaced]
        if unresolved:
            raise RegionError(
                'candidate overlaps other pending candidates; choose one or '
                'cancel them explicitly: ' + ', '.join(sorted(unresolved))
            )
        for existing in self.zones:
            if existing.box.overlaps(candidate.box):
                raise RegionError(
                    'old published region ' + existing.candidate_id +
                    ' cannot be overwritten by candidate ' + candidate_id
                )
        order = self.graph.topological_order(self.output_node)
        current = compute_signatures(
            self.graph, order, self.width, self.height, self.time
        )
        if current[self.output_node] != candidate.base_signature:
            raise RegionError(
                'candidate ' + candidate_id + ' is stale: inputs changed since it '
                'was computed; recreate it to re-judge the affected range'
            )
        source_path = self._zone_path(candidate_id)
        write_png_atomic(
            candidate.pixels, source_path, candidate.override_signature
        )
        zone = Zone(
            candidate_id=candidate_id, output_node=self.output_node,
            box=candidate.box, time=candidate.time, width=candidate.width,
            height=candidate.height, override_signature=candidate.override_signature,
            digest=candidate.digest, source_path=source_path,
        )
        self.zones.append(zone)
        for name in replaced:
            self.discard_candidate(name)
        del self.candidates[candidate_id]
        self.compose()
        return zone

    def compose(self) -> np.ndarray:
        """Lay published zones over the base image and publish atomically."""
        base = evaluate_full(
            self.graph, self.output_node, self.width, self.height, self.time,
            None, self.cache,
        )
        canvas = base.copy()
        # Zones never overlap (enforced at publish time); later adoption wins
        # only over base pixels, never over an older zone.
        for zone in self.zones:
            top, left, bottom, right = zone.box.as_tuple()
            region = self._load_zone_pixels(zone)
            if region.shape[:2] != (bottom - top, right - left):
                raise RegionError(
                    'damaged region cache for ' + zone.candidate_id +
                    ': size does not match the adopted range'
                )
            canvas[top:bottom, left:right] = region
        write_png_atomic(canvas, self.output_path, self.base_signature)
        self.base_digest = image_digest(
            np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        return canvas

    @staticmethod
    def _load_zone_pixels(zone: Zone) -> np.ndarray:
        from PIL import Image as PilImage
        if not os.path.exists(zone.source_path):
            raise RegionError('missing region source: ' + zone.source_path)
        with PilImage.open(zone.source_path) as image:
            image.load()
            if image.info.get('TextureGraph-Signature') != zone.override_signature:
                raise RegionError('damaged region source signature: ' + zone.source_path)
            pixels = np.asarray(image.convert('RGBA'), dtype=np.uint8)
            if image.info.get('TextureGraph-Content-SHA256') != image_digest(pixels):
                raise RegionError('damaged region source pixels: ' + zone.source_path)
        return (pixels.astype(np.float32) / 255.0)

    def sources(self):
        """Per-pixel provenance: candidate id, or 'base' for untouched areas."""
        labels = np.full((self.height, self.width), 'base', dtype=object)
        for zone in self.zones:
            top, left, bottom, right = zone.box.as_tuple()
            labels[top:bottom, left:right] = zone.candidate_id
        return labels

    def zone_boxes(self):
        return {zone.candidate_id: zone.box.as_tuple() for zone in self.zones}

    def save_state(self) -> str:
        """Persist partial computation state (adopted ranges + input versions)."""
        document = {
            'version': REGION_STATE_VERSION,
            'output_node': self.output_node,
            'output_path': os.path.abspath(self.output_path),
            'width': self.width,
            'height': self.height,
            'time': float(self.time),
            'base_signature': self.base_signature,
            'base_digest': self.base_digest,
            'zones': [
                {
                    'candidate_id': zone.candidate_id,
                    'box': zone.box.to_list(),
                    'time': float(zone.time),
                    'width': zone.width,
                    'height': zone.height,
                    'override_signature': zone.override_signature,
                    'digest': zone.digest,
                    'source_path': os.path.abspath(zone.source_path),
                }
                for zone in self.zones
            ],
        }
        state_path = self.state_path
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        temporary = state_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, state_path)
        return state_path

def load_region_workspace(
    graph: Graph,
    output_node: str,
    output_path: str,
    width: int,
    height: int,
    time: float = 0.0,
    cache: EvaluationCache | None = None,
    state_path: str | None = None,
) -> tuple[RegionWorkspace, dict[str, Any]]:
    """Restore a partially computed region workspace.

    Adopted ranges and their input versions are revalidated.  When a
    same-named, same-sized input changed (base signature differs) or a
    region source is damaged, the affected zones are dropped so their
    ranges can be judged and recomputed again; intact zones stay adopted.
    """
    cache = cache or EvaluationCache()
    workspace = RegionWorkspace(
        graph=graph, output_node=output_node, output_path=output_path,
        width=width, height=height, time=time, cache=cache,
        state_path=state_path,
    )
    path = workspace.state_path
    report = {'restored': [], 'dropped_stale': [], 'dropped_damaged': []}
    if not os.path.exists(path):
        return workspace, report
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            document = json.load(handle)
        if int(document.get('version', 0)) != REGION_STATE_VERSION:
            raise RegionError('unsupported region state version')
        if document.get('output_node') != output_node:
            raise RegionError('state belongs to a different output node')
        if int(document.get('width')) != width or int(document.get('height')) != height:
            raise RegionError('state canvas size differs')
        if abs(float(document.get('time', 0.0)) - float(time)) > 1e-9:
            raise RegionError('state time differs')
        saved_zones = document.get('zones', [])
    except (OSError, ValueError, KeyError, RegionError, json.JSONDecodeError):
        # Corrupted state: re-judge everything, but keep nothing blindly.
        return workspace, {
            'restored': [], 'dropped_stale': [], 'dropped_damaged': ['*']
        }

    current_base = document.get('base_signature')
    base_changed = current_base != workspace.base_signature
    for item in saved_zones:
        try:
            box = RegionBox.from_tuple(item['box'])
            source_path = item['source_path']
            zone = Zone(
                candidate_id=item['candidate_id'], output_node=output_node,
                box=box, time=float(item['time']), width=int(item['width']),
                height=int(item['height']),
                override_signature=item['override_signature'],
                digest=item['digest'], source_path=source_path,
            )
            pixels = RegionWorkspace._load_zone_pixels(zone)
            expected_shape = (box.bottom - box.top, box.right - box.left, 4)
            if tuple(pixels.shape) != expected_shape:
                raise RegionError('region size mismatch')
            if base_changed:
                report['dropped_stale'].append(zone.candidate_id)
                continue
            workspace.zones.append(zone)
            report['restored'].append(zone.candidate_id)
        except (OSError, ValueError, KeyError, RegionError):
            report['dropped_damaged'].append(item.get('candidate_id', '?'))

    # If anything survived, rebuild the composed output from current base
    # plus the adopted regions.  Otherwise the published PNG is stale and
    # must be recomposed/rejudged by the caller.
    if workspace.zones:
        workspace.compose()
    return workspace, report

def judge_affected_range(
    graph: Graph,
    output_node: str,
    width: int,
    height: int,
    time: float,
    overrides=None,
):
    """Return the exact pixel bbox changed by the overrides (or None)."""
    return influence_box(
        graph, output_node, width, height, time, overrides or {}
    )
