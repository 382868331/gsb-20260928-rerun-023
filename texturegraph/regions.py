"""Region-scoped partial recomputation and publication.

A *region* is a labelled rectangular (x, y, width, height) piece of a
final image.  Instead of recomputing the whole scheme after a local edit,
the scheme publishes one tile per region; unchanged tiles are reused,
edited tiles are recomputed with their *footprint* (the region dilated
through downstream neighborhood operations such as blur) and composited
back so filters crossing the boundary never leave a seam.

Every release carries a manifest tracing each region to the input version
and parameter choice used to render it.  Tiles staged by one candidate
plan can never be mixed into a release assembled from another candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image as PilImage

from .engine import (
    EvaluationCache,
    TextureEngine,
    image_digest,
)
from .errors import GraphValidationError, RenderCancelled
from .graph import Graph
from .variants import Variant, apply_variant

REGION_MANIFEST_NAME = "release.json"


def neighborhood_radius(node_type: str, params: dict) -> int:
    """Pixel radius of cross-pixel dependencies introduced by a node."""
    if node_type == "blur_mask":
        return max(0, int(params.get("radius_px", 1)))
    return 0


@dataclass(frozen=True)
class Region:
    """A rectangular piece of a final image.

    ``input_version`` identifies the input data/version the region is
    authored against.  Two regions with the same id, size and name but
    different input versions are provably different sources.
    """

    region_id: str
    x: int
    y: int
    width: int
    height: int
    input_version: str = "base"

    def to_dict(self) -> dict:
        return {
            "id": self.region_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "input_version": self.input_version,
        }

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def end_x(self) -> int:
        return self.x + self.width

    @property
    def end_y(self) -> int:
        return self.y + self.height

    def intersects(self, other: "Rect") -> bool:
        return not (
            other.x >= self.end_x
            or other.end_x <= self.x
            or other.y >= self.end_y
            or other.end_y <= self.y
        )

    def shares_pixels(self, other: "Rect") -> bool:
        overlap_width = min(self.end_x, other.end_x) - max(self.x, other.x)
        overlap_height = min(self.end_y, other.end_y) - max(self.y, other.y)
        return overlap_width > 0 and overlap_height > 0

    def union(self, other: "Rect") -> "Rect":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        end_x = max(self.end_x, other.end_x)
        end_y = max(self.end_y, other.end_y)
        return Rect(x, y, end_x - x, end_y - y)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def region_rect(region: Region) -> Rect:
    return Rect(region.x, region.y, region.width, region.height)


def clip_rect(rect: Rect, width: int, height: int) -> Rect:
    x = max(0, rect.x)
    y = max(0, rect.y)
    end_x = min(width, rect.end_x)
    end_y = min(height, rect.end_y)
    return Rect(x, y, max(0, end_x - x), max(0, end_y - y))


@dataclass(frozen=True)
class RegionCandidate:
    """A proposed local modification.

    The candidate layers a variant (parameter/input overrides) on top of
    the published base graph, but only inside one region.  ``candidate_id``
    distinguishes two proposals that target the same rectangle; the
    ``input_version`` records which input the edit was authored against.
    """

    candidate_id: str
    region: Region
    variant: Variant
    label: str = ""

    @property
    def choice_summary(self) -> str:
        return self.label or self.candidate_id


@dataclass(frozen=True)
class RegionPlan:
    """Preview of what a candidate changes.

    - ``recompute``: rectangle that must be rendered fresh, including the
      region itself and the neighborhood halo through filters;
    - ``reused``: rectangle of pixels that stay byte-identical to the
      published release;
    - ``full_recompute_required`` is set when the edit leaks outside every
      declared region (e.g. a global parameter changes the whole image);
    - ``overlaps`` lists other candidate ids whose footprint intersects
      this one, so the user must choose or repartition before publishing.
    """

    candidate: RegionCandidate
    recompute: Rect
    reused: Rect | None
    full_recompute_required: bool
    changed_node_ids: tuple[str, ...]
    overlaps: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.candidate_id,
            "region": self.candidate.region.to_dict(),
            "recompute": self.recompute.to_dict(),
            "reused": self.reused.to_dict() if self.reused else None,
            "full_recompute_required": self.full_recompute_required,
            "changed_nodes": list(self.changed_node_ids),
            "overlaps": list(self.overlaps),
            "reason": self.reason,
        }


LOCAL_SPATIAL_TYPES = frozenset(
    {
        "circle_mask",
        "rectangle_mask",
        "transform_coordinates",
    }
)


def _outgoing(graph: Graph) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for connection in graph.connections:
        result.setdefault(connection.source_node, []).append(connection.target_node)
    return result


def _incoming(graph: Graph) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for connection in graph.connections:
        result.setdefault(connection.target_node, []).append(connection.source_node)
    return result


def changed_nodes(graph: Graph, candidate: RegionCandidate) -> tuple[str, ...]:
    """Nodes whose effective parameters change under the candidate."""
    changed: list[str] = []
    for (node_id, param_name), value in candidate.variant.overrides.items():
        if node_id not in graph.nodes:
            raise GraphValidationError(
                f"candidate {candidate.candidate_id} overrides unknown node {node_id}"
            )
        current = graph.nodes[node_id].params.get(param_name)
        if current != value:
            changed.append(node_id)
    return tuple(sorted(set(changed)))


def _downstream_with_halo(
    graph: Graph,
    output_node: str,
    edited: Iterable[str],
    start: Rect,
    width: int,
    height: int,
) -> Rect:
    """Dilate the edited rectangle through neighborhood nodes downstream.

    Pointwise nodes keep the footprint; every neighborhood node on the
    path from an edited node to the output adds its pixel radius.  Taking
    the maximum path radius is conservative and guarantees blur halos are
    included, eliminating seams.
    """
    edges = _outgoing(graph)
    order = graph.topological_order(output_node)
    # Longest accumulated neighborhood radius from any edited node.
    radius_to: dict[str, int] = {node: 0 for node in order}
    reachable_from_edit: set[str] = set(edited)
    pending = [node for node in order if node in set(edited)]
    while pending:
        current = pending.pop()
        for target in edges.get(current, []):
            if target not in radius_to:
                continue
            target_node = graph.nodes[target]
            added = neighborhood_radius(
                target_node.type,
                graph.params_at(target, 0.0),
            )
            candidate_radius = radius_to[current] + added
            if target not in reachable_from_edit or candidate_radius > radius_to[target]:
                reachable_from_edit.add(target)
                radius_to[target] = max(radius_to[target], candidate_radius)
                pending.append(target)
    radius = max(
        (radius_to.get(node, 0) for node in reachable_from_edit),
        default=0,
    )
    dilated = Rect(
        start.x - radius,
        start.y - radius,
        start.width + 2 * radius,
        start.height + 2 * radius,
    )
    return clip_rect(dilated, width, height)


def is_global_edit(graph: Graph, node_id: str) -> bool:
    """Whether an edit to this node provably affects all image pixels.

    Spatial nodes (coordinate transforms, shape masks) only change pixels
    in the authored region; everything else (e.g. a solid color feeding a
    blend everywhere) is treated as global and forces full recomputation.
    """
    return graph.nodes[node_id].type not in LOCAL_SPATIAL_TYPES


def build_plan(
    graph: Graph,
    output_node: str,
    candidate: RegionCandidate,
    width: int,
    height: int,
) -> RegionPlan:
    """Compute the preview plan for one candidate against the base graph."""
    region = candidate.region
    rect = clip_rect(region_rect(region), width, height)
    if rect.width != region.width or rect.height != region.height:
        raise GraphValidationError(
            f"region {region.region_id} is outside the {width}x{height} image"
        )
    edited = changed_nodes(graph, candidate)
    full_canvas = Rect(0, 0, width, height)
    global_nodes = [node for node in edited if is_global_edit(graph, node)]
    if global_nodes:
        return RegionPlan(
            candidate=candidate,
            recompute=full_canvas,
            reused=None,
            full_recompute_required=True,
            changed_node_ids=edited,
            reason=(
                "global parameter/input change on node(s) "
                f"{', '.join(global_nodes)} affects pixels outside the region; "
                "full recompute required"
            ),
        )
    recompute = _downstream_with_halo(
        graph, output_node, edited, rect, width, height
    )
    return RegionPlan(
        candidate=candidate,
        recompute=recompute,
        reused=rect,
        full_recompute_required=False,
        changed_node_ids=edited,
        reason=(
            f"recompute {recompute.width}x{recompute.height} at "
            f"({recompute.x},{recompute.y}); other regions are reused"
        ),
    )


def preview_plans(
    graph: Graph,
    output_node: str,
    candidates: Iterable[RegionCandidate],
    width: int,
    height: int,
) -> list[RegionPlan]:
    """Preview every candidate and flag overlaps between their footprints."""
    plans = [
        build_plan(graph, output_node, candidate, width, height)
        for candidate in candidates
    ]
    for index, plan in enumerate(plans):
        overlaps = [
            other.candidate.candidate_id
            for other_index, other in enumerate(plans)
            if other_index != index and plan.recompute.shares_pixels(other.recompute)
        ]
        if overlaps:
            object.__setattr__(plan, "overlaps", tuple(sorted(overlaps)))
    return plans


@dataclass
class RegionTileRecord:
    region: Region
    path: str
    digest: str
    signature: str
    source_candidate: str
    choice: str
    variant_id: str
    contributors: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "region": self.region.to_dict(),
            "path": self.path,
            "content_sha256": self.digest,
            "signature": self.signature,
            "source_candidate": self.source_candidate,
            "choice": self.choice,
            "variant": self.variant_id,
            "contributors": list(self.contributors),
        }


@dataclass
class RegionRelease:
    """One formally published image assembled from region tiles.

    The manifest is the source-of-truth provenance: for every region it
    records the input version, the candidate/choice adopted and the full
    graph dependency signature of the tile pixels.  Old releases are never
    deleted, so a later input change leaves the previous release usable
    for every region that did not depend on the changed input.
    """

    release_id: str
    output_node: str
    path: str
    width: int
    height: int
    time: float
    base_signature: str
    tiles: dict[str, RegionTileRecord] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "release_id": self.release_id,
            "output_node": self.output_node,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "time": self.time,
            "base_signature": self.base_signature,
            "content_sha256": self.digest,
            "regions": {
                region_id: tile.to_dict()
                for region_id, tile in sorted(self.tiles.items())
            },
        }


@dataclass(frozen=True)
class RegionStatus:
    region_id: str
    status: str  # current | stale | damaged | missing
    release_id: str
    reason: str
    current_signature: str = ""
    tile_signature: str = ""
    input_version: str = ""
    choice: str = ""


@dataclass(frozen=True)
class SharedDamageReport:
    damaged_shared_signature: str
    affected: tuple[tuple[str, str, str], ...]  # (release, region, choice)
    unaffected_releases: tuple[str, ...]


def _load_pixels(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    try:
        with PilImage.open(path) as image:
            image.load()
            return np.array(image.convert("RGBA"), dtype=np.uint8)
    except Exception:
        return None


def _write_tile_png(path: str, pixels: np.ndarray, signature: str, label: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        image = PilImage.fromarray(pixels, mode="RGBA")
        from PIL.PngImagePlugin import PngInfo

        metadata = PngInfo()
        metadata.add_text("TextureGraph-Signature", signature)
        metadata.add_text("TextureGraph-Content-SHA256", image_digest(pixels))
        metadata.add_text("TextureGraph-Label", label)
        image.save(temporary, format="PNG", compress_level=6, pnginfo=metadata)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_graph(base_graph: Graph, candidate: RegionCandidate) -> Graph:
    graph = copy.deepcopy(base_graph)
    apply_variant(graph, candidate.variant)
    return graph


def _pixels_from_engine(
    engine: TextureEngine,
    output_node: str,
    width: int,
    height: int,
    time: float,
    cache: EvaluationCache,
    is_cancelled: Callable[[], bool],
) -> np.ndarray:
    from .engine import RenderContext

    context = RenderContext(is_cancelled=is_cancelled, time=time)
    value = engine.evaluate(output_node, context, width, height, time)
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


class RegionPublisher:
    """Publishes region-scoped partial updates of one output image."""

    def __init__(
        self,
        graph: Graph,
        output_node: str,
        width: int,
        height: int,
        root: str,
        time: float = 0.0,
        cache: EvaluationCache | None = None,
    ) -> None:
        graph.validate_references()
        if output_node not in graph.nodes:
            raise GraphValidationError(f"unknown output node: {output_node}")
        self.graph = graph
        self.output_node = output_node
        self.width = width
        self.height = height
        self.time = time
        self.root = os.path.abspath(root)
        self.cache = cache or EvaluationCache()
        os.makedirs(os.path.join(self.root, "releases"), exist_ok=True)

    # ----- paths / manifests ---------------------------------------------

    def release_dir(self, release_id: str) -> str:
        return os.path.join(self.root, "releases", release_id)

    def release_image_path(self, release_id: str) -> str:
        return os.path.join(self.release_dir(release_id), "image.png")

    def release_manifest_path(self, release_id: str) -> str:
        return os.path.join(self.release_dir(release_id), REGION_MANIFEST_NAME)

    def tile_path(self, release_id: str, region_id: str) -> str:
        return os.path.join(self.release_dir(release_id), "tiles", f"{region_id}.png")

    def _stage_path(self, plan_hash: str, region_id: str) -> str:
        return os.path.join(
            self.root, "staging", plan_hash, f"{region_id}.png"
        )

    def save_release(self, release: RegionRelease) -> None:
        os.makedirs(self.release_dir(release.release_id), exist_ok=True)
        path = self.release_manifest_path(release.release_id)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(release.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def load_release(self, release_id: str) -> RegionRelease:
        path = self.release_manifest_path(release_id)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        tiles = {}
        for region_id, item in data["regions"].items():
            region_data = item["region"]
            tiles[region_id] = RegionTileRecord(
                region=Region(
                    region_id=region_data["id"],
                    x=region_data["x"],
                    y=region_data["y"],
                    width=region_data["width"],
                    height=region_data["height"],
                    input_version=region_data.get("input_version", "base"),
                ),
                path=item["path"],
                digest=item["content_sha256"],
                signature=item["signature"],
                source_candidate=item["source_candidate"],
                choice=item["choice"],
                variant_id=item["variant"],
                contributors=tuple(item.get("contributors", [])),
            )
        return RegionRelease(
            release_id=data["release_id"],
            output_node=data["output_node"],
            path=data["path"],
            width=data["width"],
            height=data["height"],
            time=float(data["time"]),
            base_signature=data["base_signature"],
            tiles=tiles,
            digest=data.get("content_sha256", ""),
        )

    def list_releases(self) -> list[str]:
        directory = os.path.join(self.root, "releases")
        if not os.path.isdir(directory):
            return []
        return sorted(
            name
            for name in os.listdir(directory)
            if os.path.exists(
                os.path.join(directory, name, REGION_MANIFEST_NAME)
            )
        )

    def latest_release(self) -> RegionRelease | None:
        releases = self.list_releases()
        return self.load_release(releases[-1]) if releases else None

    def base_signature(self, graph: Graph | None = None) -> str:
        engine = TextureEngine(graph or self.graph, self.cache)
        return engine.signature_for(
            self.output_node, self.width, self.height, self.time
        )

    # ----- baseline (full) publication -----------------------------------

    def publish_baseline(
        self,
        regions: list[Region],
        release_id: str,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> RegionRelease:
        """Publish the current full scheme, split into documented regions."""
        self._validate_regions(regions)
        base = _pixels_from_engine(
            TextureEngine(self.graph, self.cache),
            self.output_node,
            self.width,
            self.height,
            self.time,
            self.cache,
            is_cancelled,
        )
        signature = self.base_signature()
        release = RegionRelease(
            release_id=release_id,
            output_node=self.output_node,
            path=self.release_image_path(release_id),
            width=self.width,
            height=self.height,
            time=self.time,
            base_signature=signature,
        )
        for region in regions:
            tile = _copy_tile(base, region_rect(region))
            record = self._write_release_tile(
                release_id, region, tile, signature,
                source_candidate="baseline",
                choice="published-base",
                variant_id="base",
            )
            release.tiles[region.region_id] = record
        _write_tile_png(release.path, base, signature, label=release_id)
        release.digest = image_digest(base)
        self.save_release(release)
        return release

    def _validate_regions(self, regions: list[Region]) -> None:
        ids = [region.region_id for region in regions]
        if len(ids) != len(set(ids)):
            raise GraphValidationError("region ids must be unique")
        coverage = np.zeros((self.height, self.width), dtype=bool)
        for region in regions:
            rect = clip_rect(region_rect(region), self.width, self.height)
            if rect.width <= 0 or rect.height <= 0:
                raise GraphValidationError(f"empty region: {region.region_id}")
            if rect.width != region.width or rect.height != region.height:
                raise GraphValidationError(
                    f"region {region.region_id} lies outside the image"
                )
            if coverage[rect.y:rect.end_y, rect.x:rect.end_x].any():
                raise GraphValidationError(
                    f"region {region.region_id} overlaps another region"
                )
            coverage[rect.y:rect.end_y, rect.x:rect.end_x] = True
        if not coverage.all():
            raise GraphValidationError(
                "regions must partition the whole image without gaps"
            )

    def _write_release_tile(
        self,
        release_id: str,
        region: Region,
        pixels: np.ndarray,
        signature: str,
        source_candidate: str,
        choice: str,
        variant_id: str,
    ) -> RegionTileRecord:
        path = self.tile_path(release_id, region.region_id)
        label = release_id + ":" + region.region_id
        _write_tile_png(path, pixels, signature, label=label)
        return RegionTileRecord(
            region=region,
            path=path,
            digest=image_digest(pixels),
            signature=signature,
            source_candidate=source_candidate,
            choice=choice,
            variant_id=variant_id,
        )

    # ----- partial update publication ------------------------------------

    def plan_update(
        self,
        candidates: list[RegionCandidate],
        base_release: RegionRelease | None = None,
    ) -> list[RegionPlan]:
        base_release = base_release or self.latest_release()
        if base_release is None:
            raise GraphValidationError(
                "publish a baseline release before planning region updates"
            )
        plans = preview_plans(
            self.graph, self.output_node, candidates, self.width, self.height
        )
        for plan in plans:
            self._check_region_known(plan.candidate.region, base_release)
        return plans

    def preview_update(
        self,
        candidates: list[RegionCandidate],
        base_release: RegionRelease | None = None,
    ) -> tuple[RegionRelease, list[RegionPlan]]:
        base_release = base_release or self.latest_release()
        if base_release is None:
            raise GraphValidationError("publish a baseline release before previewing updates")
        base_pixels = _load_pixels(base_release.path)
        if base_pixels is None:
            raise GraphValidationError("published release image is damaged")
        plans = self.plan_update(candidates, base_release)
        refined: list[RegionPlan] = []
        for plan in plans:
            candidate = plan.candidate
            if plan.full_recompute_required:
                refined.append(plan)
                continue
            full = self.render_full_comparison(candidate)
            differing = np.any(full != base_pixels, axis=2)
            rect = plan.recompute
            if differing.any():
                rect = clip_rect(rect.union(_mask_bounds(differing)), self.width, self.height)
            reason = (
                f"recompute {rect.width}x{rect.height} at ({rect.x},{rect.y}) "
                "(analytic halo expanded to observed changed pixels); "
                "every pixel outside is reused"
            )
            refined.append(RegionPlan(candidate=candidate, recompute=rect, reused=region_rect(candidate.region), full_recompute_required=False, changed_node_ids=plan.changed_node_ids, overlaps=plan.overlaps, reason=reason))
        for index, plan in enumerate(refined):
            overlaps = tuple(sorted(other.candidate.candidate_id for other_index, other in enumerate(refined) if other_index != index and plan.recompute.shares_pixels(other.recompute)))
            object.__setattr__(plan, "overlaps", overlaps)
        return base_release, refined

    def _check_region_known(
        self, region: Region, base_release: RegionRelease
    ) -> None:
        if region.region_id not in base_release.tiles:
            raise GraphValidationError(
                f"region {region.region_id} is not part of release "
                f"{base_release.release_id}"
            )
        existing = base_release.tiles[region.region_id].region
        if existing.bounds != region.bounds:
            raise GraphValidationError(
                f"region {region.region_id} geometry does not match the "
                "published region; same name with different geometry is a "
                "different source and must be re-partitioned"
            )

    def _validate_candidate_choice(
        self,
        plan: RegionPlan,
        chosen: dict[str, RegionCandidate],
    ) -> None:
        if plan.full_recompute_required:
            raise GraphValidationError(
                f"candidate {plan.candidate.candidate_id} requires a full "
                "recompute; use render_full_comparison instead of tiling"
            )
        if plan.overlaps:
            other = next(
                (candidate_id for candidate_id in plan.overlaps
                 if candidate_id in chosen
                 and chosen[candidate_id].region.region_id
                 != plan.candidate.region.region_id),
                None,
            )
            if other is not None:
                raise GraphValidationError(
                    f"candidates {plan.candidate.candidate_id} and {other} "
                    "cover overlapping pixels; choose one or repartition "
                    "their regions before publishing"
                )

    def publish_update(
        self,
        chosen: dict[str, RegionCandidate],
        new_release_id: str,
        base_release: RegionRelease | None = None,
        plans: list[RegionPlan] | None = None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> RegionRelease:
        """Publish a new release reusing unchanged region tiles.

        ``chosen`` maps region ids to the one candidate adopted for that
        region.  Candidates are pre-validated for overlap; tiles staged by
        a different candidate plan are never reused, so two half-finished
        candidates can never be merged into one formal release.
        """
        base_release = base_release or self.latest_release()
        if base_release is None:
            raise GraphValidationError("no published release to update")
        if os.path.exists(self.release_dir(new_release_id)):
            raise GraphValidationError(
                f"release {new_release_id} already exists; release ids are immutable"
            )
        if plans is None:
            plans = self.preview_update(list(chosen.values()), base_release)[1]
        owned_regions = [candidate.region.region_id for candidate in chosen.values()]
        if len(owned_regions) != len(set(owned_regions)):
            raise GraphValidationError(
                "two candidates claim the same region; choose exactly one per region"
            )
        plan_by_candidate = {
            plan.candidate.candidate_id: plan for plan in plans
        }
        for candidate in chosen.values():
            plan = plan_by_candidate[candidate.candidate_id]
            self._check_region_known(candidate.region, base_release)
            self._validate_candidate_choice(plan, chosen)

        plan_hash = self._plan_hash(base_release, chosen)
        canvas = _load_pixels(base_release.path)
        if canvas is None:
            raise GraphValidationError(
                f"published image {base_release.path} is unreadable; "
                "run inspect_releases to list damaged regions"
            )
        canvas = canvas.copy()
        release = RegionRelease(
            release_id=new_release_id,
            output_node=self.output_node,
            path=self.release_image_path(new_release_id),
            width=self.width,
            height=self.height,
            time=self.time,
            base_signature=self.base_signature(),
        )

        changed_regions = {
            candidate.region.region_id: (candidate, plan_by_candidate[candidate.candidate_id])
            for candidate in chosen.values()
        }

        rendered = self._render_candidate_tiles(
            chosen=chosen,
            plan_by_candidate=plan_by_candidate,
            base_release=base_release,
            canvas=canvas,
            plan_hash=plan_hash,
            is_cancelled=is_cancelled,
        )
        footprints = []
        for region_id, (_candidate, _tile, _signature, rect) in rendered.items():
            for other_id, other_rect in footprints:
                if rect.shares_pixels(other_rect):
                    raise GraphValidationError(
                        f"regions {region_id} and {other_id} share halo "
                        "pixels; repartition them or pick one candidate"
                    )
            footprints.append((region_id, rect))

        for region_id in sorted(rendered):
            _candidate, tile, _signature, rect = rendered[region_id]
            _paste_tile(canvas, tile, rect)

        for region_id, old_tile in base_release.tiles.items():
            region = old_tile.region
            tile_pixels = _copy_tile(canvas, region_rect(region))
            contributors = self._tile_contributors(region, rendered)
            if region_id in changed_regions:
                candidate, _plan = changed_regions[region_id]
                region = candidate.region
                tile_pixels = _copy_tile(canvas, region_rect(region))
                contributors = self._tile_contributors(region, rendered)
                signature = rendered[region_id][2]
                source_candidate = candidate.candidate_id
                choice = candidate.choice_summary
                variant_id = candidate.variant.id
            else:
                signature = _tile_provenance_signature(
                    old_tile.signature, contributors
                )
                source_candidate = old_tile.source_candidate
                choice = old_tile.choice
                variant_id = old_tile.variant_id
            record = self._write_release_tile(
                new_release_id, region, tile_pixels, signature,
                source_candidate=source_candidate,
                choice=choice,
                variant_id=variant_id,
            )
            object.__setattr__(record, "contributors", tuple(contributors))
            release.tiles[region_id] = record

        self._verify_no_seam(canvas, changed_regions, rendered)
        _write_tile_png(
            release.path, canvas, release.base_signature, label=new_release_id
        )
        release.digest = image_digest(canvas)
        self.save_release(release)
        self._clear_staging(plan_hash)
        return release

    # ----- rendering helpers ---------------------------------------------

    def _tile_contributors(
        self,
        region: Region,
        rendered,
    ):
        own = region_rect(region)
        contributors = []
        for other_id, (candidate, _tile, signature, rect) in rendered.items():
            if other_id == region.region_id:
                continue
            x0 = max(rect.x, own.x)
            y0 = max(rect.y, own.y)
            x1 = min(rect.end_x, own.end_x)
            y1 = min(rect.end_y, own.end_y)
            if x1 > x0 and y1 > y0:
                contributors.append(
                    {
                        "candidate": candidate.candidate_id,
                        "region": other_id,
                        "rect": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
                        "signature": signature,
                        "choice": candidate.choice_summary,
                        "input_version": candidate.region.input_version,
                    }
                )
        return contributors

    def _plan_hash(
        self,
        base_release: RegionRelease,
        chosen: dict[str, RegionCandidate],
    ) -> str:
        payload = {
            "base_release": base_release.release_id,
            "base_digest": base_release.digest,
            "output": self.output_node,
            "size": [self.width, self.height],
            "time": self.time,
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "region": candidate.region.to_dict(),
                    "variant": candidate.variant.id,
                    "overrides": _json_overrides(candidate.variant.overrides),
                }
                for candidate in sorted(
                    chosen.values(), key=lambda item: item.candidate_id
                )
            ],
        }
        return _stable_hash(payload)[:16]

    def _render_candidate_tiles(
        self,
        chosen: dict[str, RegionCandidate],
        plan_by_candidate: dict[str, RegionPlan],
        base_release: RegionRelease,
        canvas: np.ndarray,
        plan_hash: str,
        is_cancelled: Callable[[], bool],
    ) -> dict[str, tuple[RegionCandidate, np.ndarray, str, Rect]]:
        rendered: dict[str, tuple[RegionCandidate, np.ndarray, str, Rect]] = {}
        for candidate in chosen.values():
            is_cancelled()
            region = candidate.region
            plan = plan_by_candidate[candidate.candidate_id]
            rect = plan.recompute
            stage_path = self._stage_path(plan_hash, region.region_id)
            candidate_graph = _candidate_graph(self.graph, candidate)
            engine = TextureEngine(candidate_graph, self.cache)
            signature = engine.signature_for(
                self.output_node, self.width, self.height, self.time
            )
            staged = _load_valid_stage(stage_path, signature)
            if staged is not None:
                full = staged
            else:
                full = _pixels_from_engine(
                    engine,
                    self.output_node,
                    self.width,
                    self.height,
                    self.time,
                    self.cache,
                    is_cancelled,
                )
                os.makedirs(os.path.dirname(stage_path), exist_ok=True)
                _write_tile_png(
                    stage_path, full, signature,
                    label=f"stage:{plan_hash}:{region.region_id}",
                )
                is_cancelled()
            # Empirical seam guard: if pixels outside the analytic
            # footprint changed, the footprint model missed a dependency
            # and we must expand rather than publish a seam.
            changed_mask = _changed_outside(full, canvas, rect)
            if changed_mask.any():
                grown = _mask_bounds(changed_mask)
                rect = clip_rect(
                    rect.union(grown), self.width, self.height
                )
            tile = _copy_tile(full, rect)
            rendered[region.region_id] = (candidate, tile, signature, rect)
        return rendered

    def _verify_no_seam(
        self,
        canvas: np.ndarray,
        changed_regions: dict[str, tuple[RegionCandidate, RegionPlan]],
        rendered: dict[str, tuple[RegionCandidate, np.ndarray, str, Rect]],
    ) -> None:
        for region_id, (candidate, _plan) in changed_regions.items():
            candidate_graph = _candidate_graph(self.graph, candidate)
            engine = TextureEngine(candidate_graph, self.cache)
            expected = _pixels_from_engine(
                engine,
                self.output_node,
                self.width,
                self.height,
                self.time,
                self.cache,
                lambda: False,
            )
            _cand, tile, signature, rect = rendered[region_id]
            region = candidate.region
            region_rectangle = region_rect(region)
            if not np.array_equal(
                canvas[
                    region_rectangle.y:region_rectangle.end_y,
                    region_rectangle.x:region_rectangle.end_x,
                ],
                expected[
                    region_rectangle.y:region_rectangle.end_y,
                    region_rectangle.x:region_rectangle.end_x,
                ],
            ):
                raise GraphValidationError(
                    f"assembled region {region_id} does not match its "
                    "candidate render; refusing to publish"
                )

    def _clear_staging(self, plan_hash: str) -> None:
        import shutil

        path = os.path.join(self.root, "staging", plan_hash)
        shutil.rmtree(path, ignore_errors=True)

    # ----- inspection: stale input, damaged shares, full comparison ------

    def current_tile_signature(
        self,
        region: Region,
        candidate: RegionCandidate | None = None,
    ) -> str:
        graph = self.graph if candidate is None else _candidate_graph(self.graph, candidate)
        engine = TextureEngine(graph, self.cache)
        return engine.signature_for(
            self.output_node, self.width, self.height, self.time
        )

    def inspect_release(
        self,
        release: RegionRelease | None = None,
        graph: Graph | None = None,
    ) -> list[RegionStatus]:
        """Per-region status against current (or supplied) graph inputs.

        current: tile signature matches and pixels are intact;
        stale: inputs/parameters changed, old published version kept;
        damaged: tile pixels or digest no longer verify; other regions of
        the same release remain usable.
        """
        release = release or self.latest_release()
        if release is None:
            return []
        statuses: list[RegionStatus] = []
        for region_id, tile in sorted(release.tiles.items()):
            pixels = _load_pixels(tile.path)
            if pixels is None or image_digest(pixels) != tile.digest:
                statuses.append(
                    RegionStatus(
                        region_id=region_id,
                        status="damaged",
                        release_id=release.release_id,
                        reason=(
                            "tile file missing/corrupt or pixel digest "
                            "mismatch; dependent outputs must be recomputed"
                        ),
                        tile_signature=tile.signature,
                        input_version=tile.region.input_version,
                        choice=tile.choice,
                    )
                )
                continue
            current_graph = graph or self.graph
            current_signature = self.base_signature(current_graph)
            if tile.signature == current_signature:
                status, reason = "current", (
                    "inputs and parameters still match; published region is usable"
                )
            else:
                status, reason = "stale", (
                    "inputs or parameters changed after publication; old "
                    "version retained, region needs recompute"
                )
            statuses.append(
                RegionStatus(
                    region_id=region_id,
                    status=status,
                    release_id=release.release_id,
                    reason=reason,
                    current_signature=current_signature,
                    tile_signature=tile.signature,
                    input_version=tile.region.input_version,
                    choice=tile.choice,
                )
            )
        return statuses

    def render_full_comparison(
        self,
        candidate: RegionCandidate,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> np.ndarray:
        """Render a candidate over the whole image for side-by-side diff."""
        graph = _candidate_graph(self.graph, candidate)
        engine = TextureEngine(graph, self.cache)
        return _pixels_from_engine(
            engine,
            self.output_node,
            self.width,
            self.height,
            self.time,
            self.cache,
            is_cancelled,
        )

    def compare_candidate_impacts(
        self,
        candidates: list[RegionCandidate],
        base_release: RegionRelease | None = None,
    ) -> dict[str, dict]:
        """Compare each candidate diff against the same published release."""
        base_release = base_release or self.latest_release()
        if base_release is None:
            raise GraphValidationError("need a published release to compare against")
        base_pixels = _load_pixels(base_release.path)
        if base_pixels is None:
            raise GraphValidationError("base release image is damaged")
        _base, plans = self.preview_update(candidates, base_release)
        report: dict[str, dict] = {}
        for candidate, plan in zip(candidates, plans):
            full = self.render_full_comparison(candidate)
            differing = np.any(full != base_pixels, axis=2)
            ys, xs = np.where(differing)
            changed_bounds = None
            if xs.size:
                changed_bounds = Rect(
                    int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                )
                bounds_dict = changed_bounds.to_dict()
            else:
                bounds_dict = None
            within = bool(
                bounds_dict is None
                or _rect_contains(plan.recompute, changed_bounds)
            )
            report[candidate.candidate_id] = {
                "plan": plan.to_dict(),
                "changed_pixels": int(differing.sum()),
                "changed_bounds": bounds_dict,
                "within_recompute_footprint": within,
            }
        return report


# ----- module level helpers ---------------------------------------------


def _json_overrides(overrides: dict[tuple[str, str], object]) -> dict[str, object]:
    return {
        node_id + "." + param: value
        for (node_id, param), value in sorted(overrides.items())
    }


def _load_valid_stage(path: str, signature: str) -> np.ndarray | None:
    pixels = _load_pixels(path)
    if pixels is None:
        return None
    try:
        with PilImage.open(path) as image:
            image.load()
            embedded = image.info.get("TextureGraph-Signature")
            embedded_digest = image.info.get("TextureGraph-Content-SHA256")
    except Exception:
        return None
    if embedded != signature or embedded_digest != image_digest(pixels):
        return None
    return pixels


def _changed_outside(
    full: np.ndarray, canvas: np.ndarray, rect: Rect
) -> np.ndarray:
    differing = np.any(full != canvas, axis=2)
    mask = np.zeros_like(differing)
    mask[rect.y:rect.end_y, rect.x:rect.end_x] = True
    return differing & ~mask


def _mask_bounds(mask: np.ndarray) -> Rect:
    ys, xs = np.where(mask)
    return Rect(
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def _rect_contains(outer: Rect, inner: Rect) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.end_x <= outer.end_x
        and inner.end_y <= outer.end_y
    )


def _copy_tile(pixels: np.ndarray, rect: Rect) -> np.ndarray:
    return pixels[rect.y:rect.end_y, rect.x:rect.end_x].copy()


def _paste_tile(canvas: np.ndarray, pixels: np.ndarray, rect: Rect) -> None:
    canvas[rect.y:rect.end_y, rect.x:rect.end_x] = pixels


def _tile_provenance_signature(
    owner_signature: str, contributors: list[dict]
) -> str:
    """Stable signature of an unchanged tile plus intruding candidate halos."""
    if not contributors:
        return owner_signature
    return _stable_hash(
        {
            "owner": owner_signature,
            "contributors": [
                {
                    "candidate": item["candidate"],
                    "rect": item["rect"],
                    "signature": item["signature"],
                }
                for item in sorted(contributors, key=lambda item: item["candidate"])
            ],
        }
    )


def _paste_tile(canvas: np.ndarray, pixels: np.ndarray, rect: Rect) -> None:
    canvas[rect.y:rect.end_y, rect.x:rect.end_x] = pixels


def inspect_all_releases(publisher: RegionPublisher) -> dict[str, SharedDamageReport | list[RegionStatus]]:
    """Inspect every retained release.

    When a shared result is damaged, list exactly which regions (and the
    choice/provenance behind them) are affected; releases whose regions do
    not depend on the damaged signature stay usable and are listed under
    ``unaffected_releases``.
    """
    reports: dict[str, object] = {}
    damaged_signatures: set[str] = set()
    damaged_membership: dict[str, list[tuple[str, str, str]]] = {}
    per_release: dict[str, list[RegionStatus]] = {}
    for release_id in publisher.list_releases():
        release = publisher.load_release(release_id)
        statuses = publisher.inspect_release(release)
        per_release[release_id] = statuses
        for status in statuses:
            if status.status == "damaged":
                damaged_signatures.add(status.tile_signature)
                damaged_membership.setdefault(status.tile_signature, []).append(
                    (release_id, status.region_id, status.choice)
                )
    for signature in sorted(damaged_signatures):
        affected = tuple(sorted(damaged_membership[signature]))
        affected_releases = {item[0] for item in affected}
        unaffected = tuple(
            release_id
            for release_id in publisher.list_releases()
            if release_id not in affected_releases
        )
        reports[signature] = SharedDamageReport(
            damaged_shared_signature=signature,
            affected=affected,
            unaffected_releases=unaffected,
        )
    reports["releases"] = per_release
    return reports
