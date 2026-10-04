from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image as PilImage
from PIL.PngImagePlugin import PngInfo

from .engine import (
    EvaluationCache,
    RenderContext,
    TextureEngine,
    image_digest,
    stable_json,
    to_pil_image,
)
from .errors import GraphValidationError, RenderCancelled
from .graph import Graph

REGION_STATE_VERSION = 1


@dataclass(frozen=True)
class Region:
    """An axis-aligned pixel rectangle: inclusive start, exclusive end."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def intersects(self, other: "Region") -> bool:
        return (
            self.x0 < other.x1
            and other.x0 < self.x1
            and self.y0 < other.y1
            and other.y0 < self.y1
        )

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values) -> "Region":
        x0, y0, x1, y1 = (int(v) for v in values)
        return cls(x0, y0, x1, y1)


@dataclass(frozen=True)
class RegionTarget:
    """A published output that partitions can be applied to."""

    node_id: str
    width: int
    height: int
    time: float = 0.0
    label: str = ""

    def key(self) -> str:
        digest = hashlib.sha256(
            f"{self.node_id}|{self.width}x{self.height}|{self.time:g}|{self.label}".encode()
        ).hexdigest()[:16]
        return f"{self.node_id}-{self.width}x{self.height}-t{self.time:g}-{digest}"


def _to_uint8(value: np.ndarray) -> np.ndarray:
    return np.rint(
        np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _read_png(path: str) -> np.ndarray:
    with PilImage.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _write_png(
    pixels: np.ndarray, path: str, metadata: dict[str, str]
) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        image = PilImage.fromarray(pixels, mode="RGBA")
        info = PngInfo()
        for key, value in sorted(metadata.items()):
            info.add_text(key, value)
        image.save(temporary, format="PNG", compress_level=6, pnginfo=info)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: str, document: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


@dataclass
class Candidate:
    id: str
    target: RegionTarget
    region: Region
    overrides: dict[str, dict[str, object]]
    signature: str
    content_digest: str
    tile_path: str
    meta_path: str
    full_digest_match: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": {
                "node_id": self.target.node_id,
                "width": self.target.width,
                "height": self.target.height,
                "time": self.target.time,
                "label": self.target.label,
            },
            "region": self.region.as_list(),
            "overrides": {
                node_id: {name: stable_json(value) for name, value in params.items()}
                for node_id, params in self.overrides.items()
            },
            "signature": self.signature,
            "content_digest": self.content_digest,
            "tile_path": self.tile_path,
            "meta_path": self.meta_path,
            "full_digest_match": self.full_digest_match,
        }

    @classmethod
    def from_dict(cls, document: dict) -> "Candidate":
        target_document = document["target"]
        target = RegionTarget(
            node_id=target_document["node_id"],
            width=int(target_document["width"]),
            height=int(target_document["height"]),
            time=float(target_document.get("time", 0.0)),
            label=target_document.get("label", ""),
        )
        return cls(
            id=document["id"],
            target=target,
            region=Region.from_list(document["region"]),
            overrides={
                node_id: dict(params)
                for node_id, params in document.get("overrides", {}).items()
            },
            signature=document["signature"],
            content_digest=document["content_digest"],
            tile_path=document["tile_path"],
            meta_path=document["meta_path"],
            full_digest_match=bool(document.get("full_digest_match", False)),
        )

    def tile_intact(self) -> bool:
        """The cached candidate tile is still readable and untampered."""
        try:
            if not os.path.exists(self.tile_path):
                return False
            with PilImage.open(self.tile_path) as image:
                image.load()
                expected_size = (
                    self.region.x1 - self.region.x0,
                    self.region.y1 - self.region.y0,
                )
                if tuple(image.size) != expected_size:
                    return False
                if image.info.get("Region-Candidate-Signature") != self.signature:
                    return False
                pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                return image_digest(pixels) == self.content_digest
        except Exception:
            return False


@dataclass(eq=False)
class AdoptedTile:
    candidate_id: str
    target: RegionTarget
    region: Region
    overrides: dict[str, dict[str, object]]
    signature: str
    content_digest: str
    published_version: int
    tile_path: str
    valid: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "target": {
                "node_id": self.target.node_id,
                "width": self.target.width,
                "height": self.target.height,
                "time": self.target.time,
                "label": self.target.label,
            },
            "region": self.region.as_list(),
            "overrides": {
                node_id: {name: stable_json(value) for name, value in params.items()}
                for node_id, params in self.overrides.items()
            },
            "signature": self.signature,
            "content_digest": self.content_digest,
            "published_version": self.published_version,
            "tile_path": self.tile_path,
            "valid": self.valid,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, document: dict) -> "AdoptedTile":
        target_document = document["target"]
        target = RegionTarget(
            node_id=target_document["node_id"],
            width=int(target_document["width"]),
            height=int(target_document["height"]),
            time=float(target_document.get("time", 0.0)),
            label=target_document.get("label", ""),
        )
        return cls(
            candidate_id=document["candidate_id"],
            target=target,
            region=Region.from_list(document["region"]),
            overrides={
                node_id: dict(params)
                for node_id, params in document.get("overrides", {}).items()
            },
            signature=document["signature"],
            content_digest=document["content_digest"],
            published_version=int(document["published_version"]),
            tile_path=document["tile_path"],
            valid=bool(document.get("valid", True)),
            reason=document.get("reason", ""),
        )


class RegionWorkspace:
    """Region-candidate scratch area plus partitioned publishing.

    Each published target keeps an immutable, signed full-canvas baseline
    PNG.  Candidates are signed tiles computed with a modified graph copy;
    adopting one writes a new full-canvas version in which only the
    candidate rectangle changes, while the old version, other regions and
    their provenance are preserved.
    """

    def __init__(
        self,
        root: str,
        graph: Graph,
        cache: EvaluationCache | None = None,
    ) -> None:
        self.root = str(Path(root))
        self.graph = graph
        self.cache = cache or EvaluationCache()
        self.candidates: dict[str, Candidate] = {}
        self.adopted: list[AdoptedTile] = []
        self.versions: dict[str, int] = {}
        self.baselines: dict[str, dict] = {}
        self._state_path = os.path.join(self.root, "region-state.json")

    # ------------------------------------------------------------------
    # paths / signatures
    # ------------------------------------------------------------------

    def _target_dir(self, target: RegionTarget) -> str:
        return os.path.join(self.root, "targets", _safe_name(target.key()))

    def _published_path(self, target: RegionTarget, version: int) -> str:
        return os.path.join(
            self._target_dir(target), f"published-v{version:04d}.png"
        )

    def _published_meta_path(self, target: RegionTarget, version: int) -> str:
        return os.path.join(
            self._target_dir(target), f"published-v{version:04d}.json"
        )

    def _adopted_tile_path(self, target: RegionTarget, candidate_id: str) -> str:
        """Stable location of an adopted region's own pixel source.

        Adopted tiles keep their pixels independently of the published
        canvases, so a corrupted publish can be rebuilt while intact regions
        keep their content and provenance byte for byte.
        """
        return os.path.join(
            self._target_dir(target),
            "adopted",
            f"{_safe_name(candidate_id)}.png",
        )

    def _current_version(self, target: RegionTarget) -> int | None:
        return self.versions.get(target.key())

    def _node_signatures(
        self, target: RegionTarget, overrides: dict[str, dict[str, object]] | None
    ) -> tuple[TextureEngine, dict[str, str], list[str]]:
        graph = self.graph
        if overrides:
            graph = copy.deepcopy(self.graph)
            for node_id, params in overrides.items():
                if node_id not in graph.nodes:
                    raise GraphValidationError(
                        f"override for unknown node: {node_id}"
                    )
                for name, value in params.items():
                    graph.set_param(node_id, name, value)
        engine = TextureEngine(graph, self.cache)
        order = graph.topological_order(target.node_id)
        signatures = engine._signatures(
            order, target.width, target.height, target.time
        )
        return engine, signatures, order

    def _candidate_signature(
        self,
        target: RegionTarget,
        region: Region,
        overrides: dict[str, dict[str, object]],
        node_signatures: dict[str, str],
    ) -> str:
        payload = {
            "region_state_version": REGION_STATE_VERSION,
            "target": {
                "node_id": target.node_id,
                "width": target.width,
                "height": target.height,
                "time": target.time,
                "label": target.label,
            },
            "region": region.as_list(),
            "overrides": {
                node_id: {name: stable_json(value) for name, value in params.items()}
                for node_id, params in sorted(overrides.items())
            },
            "output_signature": node_signatures[target.node_id],
            "node_signatures": dict(sorted(node_signatures.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # baseline publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        target: RegionTarget,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> AdoptedTile | None:
        """Publish the current graph as a partitioned target baseline.

        Returns ``None`` when the target already has an identical current
        baseline; old published versions are never overwritten.
        """
        key = target.key()
        engine, signatures, _ = self._node_signatures(target, None)
        signature = signatures[target.node_id]
        existing_version = self.versions.get(key)
        if existing_version is not None:
            meta_path = self._published_meta_path(target, existing_version)
            if os.path.exists(meta_path):
                meta = _read_json(meta_path)
                if meta.get("signature") == signature and self._png_matches(
                    self._published_path(target, existing_version),
                    signature,
                    target,
                ):
                    return None
        value = engine.evaluate(
            target.node_id,
            RenderContext(is_cancelled=is_cancelled, time=target.time),
            target.width,
            target.height,
            target.time,
        )
        pixels = _to_uint8(value)
        version = (existing_version or 0) + 1
        self._write_published(target, version, pixels, signature, {}, signatures)
        self.versions[key] = version
        self.baselines[key] = {
            "signature": signature,
            "version": version,
            "node_signatures": dict(sorted(signatures.items())),
        }
        return None

    def _write_published(
        self,
        target: RegionTarget,
        version: int,
        pixels: np.ndarray,
        signature: str,
        tiles: dict[str, dict],
        node_signatures: dict[str, str],
    ) -> None:
        path = self._published_path(target, version)
        metadata = {
            "Region-Target": target.node_id,
            "Region-Version": str(version),
            "Region-Signature": signature,
            "Region-Content-SHA256": image_digest(pixels),
            "Region-Time": repr(float(target.time)),
        }
        if target.label:
            metadata["Region-Label"] = target.label
        _write_png(pixels, path, metadata)
        _write_json(
            self._published_meta_path(target, version),
            {
                "version": version,
                "target": {
                    "node_id": target.node_id,
                    "width": target.width,
                    "height": target.height,
                    "time": target.time,
                    "label": target.label,
                },
                "signature": signature,
                "content_digest": image_digest(pixels),
                "tiles": tiles,
                "node_signatures": dict(sorted(node_signatures.items())),
            },
        )

    def _png_matches(
        self, path: str, signature: str, target: RegionTarget
    ) -> bool:
        try:
            if not os.path.exists(path):
                return False
            with PilImage.open(path) as image:
                image.load()
                if tuple(image.size) != (target.width, target.height):
                    return False
                if image.info.get("Region-Signature") != signature:
                    return False
                pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                return image.info.get("Region-Content-SHA256") == image_digest(pixels)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # candidates
    # ------------------------------------------------------------------

    def create_candidate(
        self,
        candidate_id: str,
        target: RegionTarget,
        region: Region,
        overrides: dict[str, dict[str, object]],
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> Candidate:
        """Compute a local candidate and verify it against a full recompute.

        The candidate tile is stored separately; nothing published changes
        until ``adopt_candidate`` is called.  Cross-boundary blur halos are
        included by the windowed evaluator, and the tile is compared pixel
        by pixel with a full-canvas recompute under the same overrides.
        """
        if candidate_id in self.candidates:
            raise ValueError(f"candidate already exists: {candidate_id}")
        if self._current_version(target) is None:
            raise GraphValidationError(
                f"target {target.node_id} must be published before creating candidates"
            )
        region = self._validated_region(region, target)
        overlapping = sorted(
            other.id
            for other in self.candidates.values()
            if other.target.key() == target.key()
            and other.region.intersects(region)
        )
        if overlapping:
            raise ValueError(
                "candidate overlaps pending candidate(s) that must be chosen "
                f"or cancelled explicitly: {', '.join(overlapping)}"
            )
        overrides = {node_id: dict(params) for node_id, params in overrides.items()}
        engine, signatures, _ = self._node_signatures(target, overrides)
        signature = self._candidate_signature(target, region, overrides, signatures)

        context = RenderContext(is_cancelled=is_cancelled, time=target.time)
        local_value = engine.evaluate_region(
            target.node_id,
            region.tuple,
            context=context,
            width=target.width,
            height=target.height,
            time=target.time,
            overrides=overrides,
        )
        context.raise_if_cancelled()
        verify_context = RenderContext(
            is_cancelled=is_cancelled, time=target.time
        )
        full_value = engine.evaluate(
            target.node_id,
            verify_context,
            target.width,
            target.height,
            target.time,
        )
        full_tile = _to_uint8(
            full_value[
                region.y0 : region.y1, region.x0 : region.x1
            ]
        )
        tile = _to_uint8(local_value)
        match = bool(np.array_equal(tile, full_tile))
        if not match:
            raise RenderCancelled(
                "region candidate does not match full recompute pixel by pixel"
            )

        tile_path = os.path.join(
            self._target_dir(target),
            "candidates",
            f"{_safe_name(candidate_id)}.png",
        )
        meta_path = os.path.join(
            self._target_dir(target),
            "candidates",
            f"{_safe_name(candidate_id)}.json",
        )
        digest = image_digest(tile)
        _write_png(
            tile,
            tile_path,
            {
                "Region-Candidate": candidate_id,
                "Region-Candidate-Signature": signature,
                "Region-Candidate-Content-SHA256": digest,
            },
        )
        candidate = Candidate(
            id=candidate_id,
            target=target,
            region=region,
            overrides=overrides,
            signature=signature,
            content_digest=digest,
            tile_path=tile_path,
            meta_path=meta_path,
            full_digest_match=match,
        )
        _write_json(meta_path, candidate.to_dict())
        self.candidates[candidate_id] = candidate
        return candidate

    def cancel_candidate(self, candidate_id: str) -> None:
        """Discard a pending candidate; published content is untouched."""
        candidate = self.candidates.pop(candidate_id, None)
        if candidate is None:
            raise KeyError(f"no such candidate: {candidate_id}")
        for path in (candidate.tile_path, candidate.meta_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def adopt_candidate(
        self,
        candidate_id: str,
        replace: bool = False,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> AdoptedTile:
        """Publish a new version containing exactly one candidate region.

        Overlapping adopted regions are rejected unless ``replace`` is set,
        in which case the earlier tiles are dropped from the new version.
        Pending candidates overlapping the adopted rectangle must be
        cancelled or adopted explicitly first.
        """
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"no such candidate: {candidate_id}")
        target = candidate.target
        if not candidate.tile_intact():
            raise ValueError(
                f"candidate {candidate_id} tile is missing or damaged; recompute it"
            )
        current_version = self._current_version(target)
        if current_version is None:
            raise GraphValidationError("target baseline is missing")

        pending_others = sorted(
            other.id
            for other in self.candidates.values()
            if other.id != candidate.id
            and other.target.key() == target.key()
        )
        if pending_others:
            raise ValueError(
                "other pending candidate(s) for the same target must be "
                "chosen or cancelled explicitly first: "
                f"{', '.join(pending_others)}"
            )

        active = [
            tile for tile in self.adopted if tile.valid and tile.target.key() == target.key()
        ]
        overlapping = [
            tile
            for tile in active
            if tile.region.intersects(candidate.region)
        ]
        if overlapping and not replace:
            ids = ", ".join(sorted(tile.candidate_id for tile in overlapping))
            raise ValueError(
                f"candidate overlaps adopted region(s): {ids}; adopt with "
                "replace=True or cancel the overlap explicitly"
            )
        context = RenderContext(is_cancelled=is_cancelled, time=target.time)
        context.raise_if_cancelled()
        pixels = _read_png(
            self._published_path(target, current_version)
        ).copy()
        tile_pixels = _read_png(candidate.tile_path)
        pixels[
            candidate.region.y0 : candidate.region.y1,
            candidate.region.x0 : candidate.region.x1,
        ] = tile_pixels

        kept_tiles = [
            tile
            for tile in active
            if not (replace and tile.region.intersects(candidate.region))
        ]
        tile_records: dict[str, dict] = {}
        for tile in kept_tiles:
            tile_records[tile.candidate_id] = tile.to_dict()
        adopted_record = {
            "candidate_id": candidate.id,
            "region": candidate.region.as_list(),
            "overrides": candidate.overrides,
            "signature": candidate.signature,
            "content_digest": candidate.content_digest,
        }
        tile_records[candidate.id] = adopted_record

        engine, signatures, _ = self._node_signatures(target, None)
        published_signature = self._partition_signature(
            target, current_version + 1, tile_records
        )
        self._write_published(
            target,
            current_version + 1,
            pixels,
            published_signature,
            tile_records,
            signatures,
        )
        adopted_tile_path = self._adopted_tile_path(target, candidate.id)
        _write_png(
            tile_pixels,
            adopted_tile_path,
            {
                "Region-Candidate": candidate.id,
                "Region-Candidate-Signature": candidate.signature,
                "Region-Candidate-Content-SHA256": candidate.content_digest,
            },
        )
        adopted_tile = AdoptedTile(
            candidate_id=candidate.id,
            target=target,
            region=candidate.region,
            overrides=dict(candidate.overrides),
            signature=candidate.signature,
            content_digest=candidate.content_digest,
            published_version=current_version + 1,
            tile_path=adopted_tile_path,
        )
        self.adopted.append(adopted_tile)
        if replace:
            for tile in overlapping:
                tile.valid = False
                tile.reason = (
                    f"replaced by candidate {candidate.id} at version {current_version + 1}"
                )
        self.versions[target.key()] = current_version + 1
        for path in (candidate.meta_path,):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        try:
            os.unlink(candidate.tile_path)
        except FileNotFoundError:
            pass
        del self.candidates[candidate.id]
        return adopted_tile

    def _partition_signature(
        self,
        target: RegionTarget,
        version: int,
        tile_records: dict[str, dict],
    ) -> str:
        payload = {
            "kind": "partitioned",
            "region_state_version": REGION_STATE_VERSION,
            "target": {
                "node_id": target.node_id,
                "width": target.width,
                "height": target.height,
                "time": target.time,
                "label": target.label,
            },
            "version": version,
            "tiles": {
                key: {
                    "region": record["region"],
                    "signature": record["signature"],
                    "content_digest": record["content_digest"],
                    "overrides": {
                        node_id: {
                            name: stable_json(value) for name, value in params.items()
                        }
                        for node_id, params in record["overrides"].items()
                    },
                }
                for key, record in sorted(tile_records.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validated_region(self, region: Region, target: RegionTarget) -> Region:
        x0 = max(0, min(target.width, int(region.x0)))
        y0 = max(0, min(target.height, int(region.y0)))
        x1 = max(0, min(target.width, int(region.x1)))
        y1 = max(0, min(target.height, int(region.y1)))
        if x1 <= x0 or y1 <= y0:
            raise GraphValidationError("region must contain at least one pixel")
        return Region(x0, y0, x1, y1)

    def published_path(self, target: RegionTarget) -> str | None:
        version = self._current_version(target)
        if version is None:
            return None
        return self._published_path(target, version)

    def read_published(self, target: RegionTarget) -> np.ndarray | None:
        path = self.published_path(target)
        if path is None or not os.path.exists(path):
            return None
        return _read_png(path)

    # ------------------------------------------------------------------
    # state persistence and recovery
    # ------------------------------------------------------------------

    def save_state(self) -> str:
        """Persist partial computation state: adopted ranges and input versions.

        Pending (possibly half-finished) candidates, adopted rectangles and
        the complete per-node input signatures are recorded so a later
        session can restore exactly what was adopted and recompute only the
        ranges affected by same-name/same-size input changes or damage.
        """
        targets: dict[str, dict] = {}
        def _target_document(target: RegionTarget) -> dict:
            return {
                "node_id": target.node_id,
                "width": target.width,
                "height": target.height,
                "time": target.time,
                "label": target.label,
            }

        target_keys: dict[str, RegionTarget] = {}
        for candidate in self.candidates.values():
            target_keys.setdefault(candidate.target.key(), candidate.target)
        for tile in self.adopted:
            target_keys.setdefault(tile.target.key(), tile.target)
        for key in set(self.versions) | set(target_keys):
            targets[key] = {
                "version": self.versions.get(key),
                "baseline": self.baselines.get(key),
                "target": _target_document(target_keys[key]),
                "adopted": [
                    tile.to_dict()
                    for tile in self.adopted
                    if tile.target.key() == key
                ],
            }
        document = {
            "state_version": REGION_STATE_VERSION,
            "targets": targets,
            "candidates": [
                candidate.to_dict() for candidate in self.candidates.values()
            ],
        }
        _write_json(self._state_path, document)
        return self._state_path

    @classmethod
    def open(
        cls,
        root: str,
        graph: Graph,
        cache: EvaluationCache | None = None,
        recover: bool = True,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> "RegionWorkspace":
        """Restore adopted ranges, input versions and pending candidates.

        No new version is published here; call ``recover_target`` to
        rejudge the affected ranges explicitly.  Pending candidates whose
        tile is damaged are flagged unless ``recover`` is disabled.
        """
        workspace = cls(root, graph, cache)
        state_path = workspace._state_path
        if not os.path.exists(state_path):
            return workspace
        document = _read_json(state_path)
        if int(document.get("state_version", 1)) != REGION_STATE_VERSION:
            raise ValueError("unsupported region state version")
        for candidate_document in document.get("candidates", []):
            candidate = Candidate.from_dict(candidate_document)
            workspace.candidates[candidate.id] = candidate
        target_by_key: dict[str, RegionTarget] = {}
        for key, target_document in document.get("targets", {}).items():
            if target_document.get("version") is not None:
                workspace.versions[key] = int(target_document["version"])
            baseline = target_document.get("baseline")
            if baseline:
                workspace.baselines[key] = baseline
            for tile_document in target_document.get("adopted", []):
                tile = AdoptedTile.from_dict(tile_document)
                workspace.adopted.append(tile)
                target_by_key.setdefault(key, tile.target)
            target_info = target_document.get("target")
            if key not in target_by_key and target_info:
                target_by_key[key] = RegionTarget(
                    node_id=target_info["node_id"],
                    width=int(target_info["width"]),
                    height=int(target_info["height"]),
                    time=float(target_info.get("time", 0.0)),
                    label=target_info.get("label", ""),
                )
        if recover:
            for candidate in list(workspace.candidates.values()):
                if not candidate.tile_intact():
                    candidate.full_digest_match = False
        return workspace

    def recover_target(
        self,
        target: RegionTarget,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> dict:
        """Revalidate one target after input changes or cache/file damage.

        Returns a report listing changed nodes, affected adopted regions and
        whether a new partitioned version was published.  Outside affected
        ranges the previously published pixels and sources are reused byte
        for byte.
        """
        key = target.key()
        version = self.versions.get(key)
        report = {
            "target": target.node_id,
            "changed_nodes": [],
            "damaged": False,
            "affected_tiles": [],
            "recomputed": False,
        }
        if version is None:
            return report
        published_path = self._published_path(target, version)
        meta_path = self._published_meta_path(target, version)
        baseline_info = self.baselines.get(key) or {}
        old_node_signatures: dict[str, str] = dict(
            baseline_info.get("node_signatures", {})
        )
        engine, current_signatures, order = self._node_signatures(target, None)
        changed_nodes = {
            node_id
            for node_id in order
            if node_id in old_node_signatures
            and old_node_signatures[node_id] != current_signatures[node_id]
        }
        report["changed_nodes"] = sorted(changed_nodes)

        published_signature = None
        tile_records: dict[str, dict] = {}
        if os.path.exists(meta_path):
            meta = _read_json(meta_path)
            published_signature = meta.get("signature")
            tile_records = dict(meta.get("tiles", {}))
            old_node_signatures = dict(
                meta.get("node_signatures", old_node_signatures)
            )
            changed_nodes = {
                node_id
                for node_id in order
                if node_id in old_node_signatures
                and old_node_signatures[node_id] != current_signatures[node_id]
            }
            report["changed_nodes"] = sorted(changed_nodes)

        published_ok = self._png_matches(
            published_path, published_signature or "", target
        )
        active_tiles = [
            tile
            for tile in self.adopted
            if tile.target.key() == key and tile.valid
        ]

        affected: list[AdoptedTile] = []
        for tile in active_tiles:
            influences = self._nodes_influencing_region(tile, changed_nodes, target)
            tile_damaged = not self._adopted_tile_pixels_intact(tile, target)
            if influences or tile_damaged or not published_ok:
                affected.append(tile)
                report["affected_tiles"].append(tile.candidate_id)
        if not published_ok:
            report["damaged"] = True

        if not published_ok:
            # File/signature damage: rebuild baseline fully, but only
            # re-evaluate tiles whose stored pixels cannot be trusted.
            damaged_tiles = {
                tile for tile in active_tiles if not self._adopted_tile_pixels_intact(tile, target)
            }
            return self._rebuild_target(
                target,
                engine,
                current_signatures,
                active_tiles,
                damaged_tiles,
                report,
                is_cancelled,
                full_rebuild=True,
            )
        if not affected:
            return report
        return self._rebuild_target(
            target,
            engine,
            current_signatures,
            active_tiles,
            set(affected),
            report,
            is_cancelled,
        )

    def _nodes_influencing_region(
        self,
        tile: AdoptedTile,
        changed_nodes: set[str],
        target: RegionTarget,
    ) -> set[str]:
        """Changed nodes whose value can reach the output inside the tile."""
        if not changed_nodes:
            return set()
        # Every reachable upstream node contributes to every output pixel
        # here (pointwise graph); neighbourhood dilation only changes the
        # evaluation window, not the dependency set.
        order = self.graph.topological_order(target.node_id)
        ancestors = set(order)
        return changed_nodes & ancestors

    def _adopted_tile_pixels_intact(
        self, tile: AdoptedTile, target: RegionTarget
    ) -> bool:
        """The independently stored tile pixels are still untampered.

        Provenance is stored separately from the published canvases, so a
        corrupted or missing publish does not invalidate an intact tile.
        """
        path = tile.tile_path
        try:
            if not os.path.exists(path):
                return False
            with PilImage.open(path) as image:
                image.load()
                expected_size = (
                    tile.region.x1 - tile.region.x0,
                    tile.region.y1 - tile.region.y0,
                )
                if tuple(image.size) != expected_size:
                    return False
                if image.info.get("Region-Candidate-Signature") != tile.signature:
                    return False
                pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                return image_digest(pixels) == tile.content_digest
        except Exception:
            return False

    def _rebuild_target(
        self,
        target: RegionTarget,
        base_engine: TextureEngine,
        current_signatures: dict[str, str],
        active_tiles: list[AdoptedTile],
        affected: set[AdoptedTile],
        report: dict,
        is_cancelled: Callable[[], bool],
        full_rebuild: bool = False,
    ) -> dict:
        """Recompute only affected ranges; republish preserving other regions."""
        current_version = self.versions[target.key()]
        source_path = self._published_path(target, current_version)
        if os.path.exists(source_path) and not full_rebuild:
            pixels = _read_png(source_path).copy()
        else:
            pixels = _to_uint8(
                base_engine.evaluate(
                    target.node_id,
                    RenderContext(
                        is_cancelled=is_cancelled, time=target.time
                    ),
                    target.width,
                    target.height,
                    target.time,
                )
            )

        # The base canvas comes from the current inputs for a full rebuild
        # (damaged publish) or from the previous publish otherwise.
        # Recompute each affected tile's overrides locally; intact tiles
        # (including during a full-canvas rebuild after publish damage) are
        # pasted back from their independently stored pixels, keeping their
        # published content and provenance byte for byte.
        new_tiles: list[AdoptedTile] = []
        for tile in active_tiles:
            if tile in affected:
                report["recomputed"] = True
                context = RenderContext(
                    is_cancelled=is_cancelled, time=target.time
                )
                local_value = base_engine.evaluate_region(
                    target.node_id,
                    tile.region.tuple,
                    context=context,
                    width=target.width,
                    height=target.height,
                    time=target.time,
                    overrides=tile.overrides or None,
                )
                new_pixels = _to_uint8(local_value)
                pixels[
                    tile.region.y0 : tile.region.y1,
                    tile.region.x0 : tile.region.x1,
                ] = new_pixels
                digest = image_digest(new_pixels)
                engine, signatures, _ = self._node_signatures(
                    target, tile.overrides or None
                )
                signature = self._candidate_signature(
                    target, tile.region, tile.overrides or {}, signatures
                )
                tile_path = self._adopted_tile_path(target, tile.candidate_id)
                _write_png(
                    new_pixels,
                    tile_path,
                    {
                        "Region-Candidate": tile.candidate_id,
                        "Region-Candidate-Signature": signature,
                        "Region-Candidate-Content-SHA256": digest,
                    },
                )
                new_tile = AdoptedTile(
                    candidate_id=tile.candidate_id,
                    target=target,
                    region=tile.region,
                    overrides=dict(tile.overrides),
                    signature=signature,
                    content_digest=digest,
                    published_version=current_version + 1,
                    tile_path=tile_path,
                )
                new_tiles.append(new_tile)
            else:
                stored_pixels = _read_png(tile.tile_path)
                pixels[
                    tile.region.y0 : tile.region.y1,
                    tile.region.x0 : tile.region.x1,
                ] = stored_pixels
                new_tiles.append(tile)

        tile_records = {tile.candidate_id: tile.to_dict() for tile in new_tiles}
        published_signature = self._partition_signature(
            target, current_version + 1, tile_records
        )
        self._write_published(
            target,
            current_version + 1,
            pixels,
            published_signature,
            tile_records,
            current_signatures,
        )
        for tile in affected:
            tile.valid = False
            tile.reason = (
                f"superseded by recovery version {current_version + 1}"
            )
        self.adopted.extend(new_tiles)
        self.versions[target.key()] = current_version + 1
        self.baselines[target.key()] = {
            "signature": current_signatures[target.node_id],
            "version": current_version + 1,
            "node_signatures": dict(sorted(current_signatures.items())),
        }
        report["new_version"] = current_version + 1
        return report
