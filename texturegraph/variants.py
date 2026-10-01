"""Named parameter schemes (variants) rendered and compared as one batch.

A :class:`Variant` is a named set of parameter overrides on top of a shared
base graph.  A batch renders several variants in one submission:

- nodes untouched by any override share one :class:`EvaluationCache`, so a
  shared mask or transform is computed once for the whole batch;
- each published PNG carries the variant id in its dependency signature, so
  two schemes can never claim each other's output even when their effective
  parameters, node names and sizes are identical;
- cancelling mid-batch keeps the variants that already published; on resume
  every variant's outputs are validated individually against current inputs;
- :func:`compare_variants` reports, per variant and output, whether the file
  on disk is the result of the current inputs or a kept older version.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .engine import (
    EvaluationCache,
    FrameOutput,
    FrameRequest,
    FrameResult,
    TextureEngine,
    published_png_is_valid,
)
from .errors import GraphValidationError
from .graph import Connection, Graph, Node

STATUS_CURRENT = "current"
STATUS_STALE = "stale"
STATUS_MISSING = "missing"


@dataclass(frozen=True)
class Variant:
    """One named parameter scheme: overrides on top of the base graph."""

    id: str
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    label: str | None = None

    def display_name(self) -> str:
        return self.label or self.id


def apply_variant(graph: Graph, variant: Variant) -> Graph:
    """Return a copy of ``graph`` with the variant's overrides applied."""
    for node_id in variant.overrides:
        if node_id not in graph.nodes:
            raise GraphValidationError(
                f"variant {variant.id!r} overrides unknown node: {node_id}"
            )
    derived = Graph(width=graph.width, height=graph.height)
    for node_id, node in graph.nodes.items():
        params = dict(node.params)
        params.update(variant.overrides.get(node_id, {}))
        derived.nodes[node_id] = Node(node_id, node.type, params)
    derived.connections = [
        Connection(
            connection.source_node,
            connection.source_port,
            connection.target_node,
            connection.target_port,
        )
        for connection in graph.connections
    ]
    derived.tracks = dict(graph.tracks)
    derived.validate_references()
    return derived


@dataclass(frozen=True)
class VariantOutput:
    """One output target rendered for every variant in a batch.

    ``path`` may contain ``{variant}``, replaced by the variant id.  When a
    batch contains more than one variant, every output path must stay
    distinct per variant so same-named, same-sized results never replace
    each other on disk.
    """

    node_id: str
    path: str
    width: int
    height: int
    time: float = 0.0

    def resolve_path(self, variant_id: str) -> str:
        return self.path.format(variant=variant_id)


@dataclass
class VariantBatchResult:
    variant_id: str
    label: str
    frames: list[FrameResult]
    outputs: dict[str, str]


def select_variants(
    variants: Iterable[Variant], only: Iterable[str] | None
) -> list[Variant]:
    variants = list(variants)
    if only is None:
        return variants
    wanted = list(only)
    known = {variant.id for variant in variants}
    unknown = [variant_id for variant_id in wanted if variant_id not in known]
    if unknown:
        raise GraphValidationError(f"unknown variant ids: {', '.join(unknown)}")
    return [variant for variant in variants if variant.id in set(wanted)]


def render_variant_batch(
    graph: Graph,
    variants: list[Variant],
    outputs: list[VariantOutput],
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
    only: Iterable[str] | None = None,
) -> dict[str, VariantBatchResult]:
    """Render several parameter schemes in one batch.

    All variants share one evaluation cache, so inputs and intermediate
    results untouched by overrides are computed once.  ``only`` restricts
    the run to the selected variant ids; the rest keep whatever they have
    already published and can be handled by a later call.
    """
    selected = select_variants(variants, only)
    cache = cache or EvaluationCache()
    _check_distinct_paths(selected, outputs)

    results: dict[str, VariantBatchResult] = {}
    for variant in selected:
        derived = apply_variant(graph, variant)
        engine = TextureEngine(derived, cache, variant_id=variant.id)
        frames = _frames_for_variant(variant, outputs)
        frame_results = engine.render_frames(
            frames, is_cancelled=is_cancelled, resume=resume
        )
        published: dict[str, str] = {}
        for frame_result in frame_results:
            published.update(frame_result.outputs)
        results[variant.id] = VariantBatchResult(
            variant_id=variant.id,
            label=variant.display_name(),
            frames=frame_results,
            outputs=published,
        )
    return results


def _frames_for_variant(
    variant: Variant, outputs: list[VariantOutput]
) -> list[FrameRequest]:
    by_time: dict[float, list[FrameOutput]] = {}
    order: list[float] = []
    for output in outputs:
        time = float(output.time)
        if time not in by_time:
            by_time[time] = []
            order.append(time)
        by_time[time].append(
            FrameOutput(
                output.node_id,
                output.resolve_path(variant.id),
                output.width,
                output.height,
            )
        )
    return [FrameRequest(time, tuple(by_time[time])) for time in order]


def _check_distinct_paths(
    variants: list[Variant], outputs: list[VariantOutput]
) -> None:
    seen: dict[str, str] = {}
    for variant in variants:
        for output in outputs:
            path = os.path.abspath(output.resolve_path(variant.id))
            owner = seen.get(path)
            if owner is not None and owner != variant.id:
                raise GraphValidationError(
                    f"variants {owner!r} and {variant.id!r} would both write "
                    f"{path}: give each scheme a distinct output path "
                    "(for example with a {variant} placeholder)"
                )
            seen[path] = variant.id


@dataclass(frozen=True)
class OutputComparison:
    """Whether one published output still matches the current inputs."""

    variant_id: str
    label: str
    node_id: str
    path: str
    status: str
    reason: str


def compare_variants(
    graph: Graph,
    variants: list[Variant],
    outputs: list[VariantOutput],
) -> list[OutputComparison]:
    """Classify every variant output as current, stale or missing.

    ``current``  - the PNG on disk matches the current inputs exactly;
    ``stale``    - a file exists but was published from older inputs and is
                   kept, untouched, as a previous version;
    ``missing``  - nothing has been published for this variant output yet.
    """
    comparisons: list[OutputComparison] = []
    for variant in variants:
        derived = apply_variant(graph, variant)
        engine = TextureEngine(derived, EvaluationCache(), variant_id=variant.id)
        for output in outputs:
            path = output.resolve_path(variant.id)
            frame_output = FrameOutput(
                output.node_id, path, output.width, output.height
            )
            if not os.path.exists(path):
                comparisons.append(
                    OutputComparison(
                        variant.id,
                        variant.display_name(),
                        output.node_id,
                        path,
                        STATUS_MISSING,
                        "no published output for this scheme yet",
                    )
                )
                continue
            expected = engine.signature_for(
                output.node_id, output.width, output.height, float(output.time)
            )
            if published_png_is_valid(frame_output, expected):
                comparisons.append(
                    OutputComparison(
                        variant.id,
                        variant.display_name(),
                        output.node_id,
                        path,
                        STATUS_CURRENT,
                        "published from the current inputs",
                    )
                )
            else:
                comparisons.append(
                    OutputComparison(
                        variant.id,
                        variant.display_name(),
                        output.node_id,
                        path,
                        STATUS_STALE,
                        "kept previous version: inputs changed or file is damaged",
                    )
                )
    return comparisons


def affected_variants(
    comparisons: list[OutputComparison],
) -> dict[str, list[OutputComparison]]:
    """Group the stale/missing outputs by variant id.

    After a shared input changes, this lists exactly which schemes and
    outputs are affected; variants absent from the result keep their
    published versions and need no work.
    """
    affected: dict[str, list[OutputComparison]] = {}
    for comparison in comparisons:
        if comparison.status == STATUS_CURRENT:
            continue
        affected.setdefault(comparison.variant_id, []).append(comparison)
    return affected


def format_comparison(comparisons: list[OutputComparison]) -> str:
    """Render the comparison page: per scheme, which outputs are current
    results and which are kept older versions."""
    lines = ["comparison of parameter schemes:"]
    by_variant: dict[str, list[OutputComparison]] = {}
    order: list[str] = []
    labels: dict[str, str] = {}
    for comparison in comparisons:
        if comparison.variant_id not in by_variant:
            by_variant[comparison.variant_id] = []
            order.append(comparison.variant_id)
            labels[comparison.variant_id] = comparison.label
        by_variant[comparison.variant_id].append(comparison)
    for variant_id in order:
        entries = by_variant[variant_id]
        current = sum(1 for entry in entries if entry.status == STATUS_CURRENT)
        lines.append(
            f"  scheme {labels[variant_id]} ({variant_id}): "
            f"{current}/{len(entries)} outputs current"
        )
        for entry in entries:
            marker = {
                STATUS_CURRENT: "current",
                STATUS_STALE: "OLD VERSION kept",
                STATUS_MISSING: "missing",
            }[entry.status]
            lines.append(f"    {entry.node_id}: {marker} - {entry.path}")
    affected = affected_variants(comparisons)
    if affected:
        lines.append("  affected schemes: " + ", ".join(sorted(affected)))
    else:
        lines.append("  all schemes are up to date")
    return "\n".join(lines)
