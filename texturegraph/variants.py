from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

from .engine import (
    EvaluationCache,
    FrameOutput,
    FrameRequest,
    FrameResult,
    TextureEngine,
    published_png_is_valid,
)
from .errors import GraphValidationError
from .graph import Graph
from .project import RenderJob, load_project, resolve_output_path


@dataclass(frozen=True)
class Variant:
    """A named parameter scheme layered on top of the base graph.

    Overrides map (node_id, param_name) to a replacement value.  Variants
    share the graph topology, so nodes they do not touch keep identical
    signatures and share cached intermediate results across variants.
    """

    id: str
    overrides: dict[tuple[str, str], object] = field(default_factory=dict)


def variant_from_dict(document: dict) -> Variant:
    overrides: dict[tuple[str, str], object] = {}
    for key, value in document.get("overrides", {}).items():
        node_id, separator, param_name = str(key).partition(".")
        if not separator:
            raise ValueError(f"variant override must be 'node.param': {key}")
        overrides[(node_id, param_name)] = value
    return Variant(id=str(document["id"]), overrides=overrides)


def apply_variant(graph: Graph, variant: Variant) -> None:
    for (node_id, param_name), value in variant.overrides.items():
        if node_id not in graph.nodes:
            raise GraphValidationError(
                f"variant {variant.id} overrides unknown node: {node_id}"
            )
        graph.set_param(node_id, param_name, value)


def load_variants(project_path: str) -> list[Variant]:
    _, _, data = load_project(project_path)
    variants = [variant_from_dict(item) for item in data.get("variants", [])]
    seen: set[str] = set()
    for variant in variants:
        if variant.id in seen:
            raise ValueError(f"duplicate variant id: {variant.id}")
        seen.add(variant.id)
    return variants


def select_variants(
    variants: list[Variant], only: Iterable[str] | None
) -> list[Variant]:
    if only is None:
        return list(variants)
    wanted = list(only)
    by_id = {variant.id: variant for variant in variants}
    unknown = [variant_id for variant_id in wanted if variant_id not in by_id]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}")
    return [by_id[variant_id] for variant_id in wanted]


def variant_output_path(job_path: str, variant_id: str, project_directory: str) -> str:
    """Scope every variant output under variants/<id>/ so same-named,
    same-sized outputs from different variants never overwrite each other.
    """
    path_obj = Path(job_path)
    relative = (
        path_obj
        if not path_obj.is_absolute()
        else Path(path_obj.name)
    )
    return str(
        (Path(project_directory) / "variants" / variant_id / relative).resolve()
    )


def _variant_frames(
    jobs: list[RenderJob],
    variant: Variant,
    project_directory: str,
) -> list[FrameRequest]:
    frames_by_time: dict[float, dict[str, FrameOutput]] = {}
    order: list[float] = []
    for job in jobs:
        variant_job = replace(
            job,
            path=variant_output_path(job.path, variant.id, project_directory),
        )
        for request in variant_job.frames(project_directory):
            time = float(request.time)
            if time not in frames_by_time:
                frames_by_time[time] = {}
                order.append(time)
            for output in request.outputs:
                labelled = FrameOutput(
                    output.node_id,
                    output.path,
                    output.width,
                    output.height,
                    label=variant.id,
                )
                frames_by_time[time][labelled.path] = labelled
    return [
        FrameRequest(time, tuple(frames_by_time[time].values()))
        for time in order
    ]


def _variant_engines(
    project_path: str,
    variants: list[Variant],
    cache: EvaluationCache,
) -> tuple[str, list[RenderJob], dict[str, TextureEngine]]:
    project_path = str(Path(project_path))
    project_directory = os.path.dirname(os.path.abspath(project_path))
    graph, _, data = load_project(project_path)
    jobs: list[RenderJob] = data.get("jobs", [])
    engines: dict[str, TextureEngine] = {}
    for variant in variants:
        variant_graph = copy.deepcopy(graph)
        apply_variant(variant_graph, variant)
        engines[variant.id] = TextureEngine(variant_graph, cache)
    return project_directory, jobs, engines


def render_variant_batch(
    project_path: str,
    only: Iterable[str] | None = None,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
) -> dict[str, list[FrameResult]]:
    """Render every selected variant in one batch.

    All variants share one EvaluationCache, so shared inputs and
    intermediate results are computed once even across variants.  Each
    variant publishes under its own directory with its id embedded in the
    PNG metadata, so outputs stay attributable to their scheme.  A cancel
    or close between variants keeps already published variants intact;
    rerunning (resume) revalidates each variant's outputs independently.
    """
    variants = select_variants(load_variants(project_path), only)
    shared_cache = cache or EvaluationCache()
    project_directory, jobs, engines = _variant_engines(
        project_path, variants, shared_cache
    )
    results: dict[str, list[FrameResult]] = {}
    for variant in variants:
        frames = _variant_frames(jobs, variant, project_directory)
        results[variant.id] = engines[variant.id].render_frames(
            frames, is_cancelled=is_cancelled, resume=resume
        )
    return results


@dataclass(frozen=True)
class VariantOutputStatus:
    variant_id: str
    node_id: str
    path: str
    status: str  # "current" | "stale" | "missing"
    reason: str


def compare_variants(
    project_path: str,
    only: Iterable[str] | None = None,
) -> dict[str, list[VariantOutputStatus]]:
    """Compare published outputs against the current inputs, per variant.

    "current" outputs match the current graph parameters and are still
    usable; "stale" outputs are old versions kept on disk (e.g. after a
    shared input changed or the file was corrupted); "missing" outputs
    were never published.  Nothing is rendered or deleted here.
    """
    variants = select_variants(load_variants(project_path), only)
    cache = EvaluationCache()
    project_directory, jobs, engines = _variant_engines(
        project_path, variants, cache
    )
    report: dict[str, list[VariantOutputStatus]] = {}
    for variant in variants:
        engine = engines[variant.id]
        statuses: list[VariantOutputStatus] = []
        for frame in _variant_frames(jobs, variant, project_directory):
            for output in frame.outputs:
                signature = engine.signature_for(
                    output.node_id, output.width, output.height, frame.time
                )
                if published_png_is_valid(output, signature):
                    statuses.append(
                        VariantOutputStatus(
                            variant.id,
                            output.node_id,
                            output.path,
                            "current",
                            "matches current inputs; published version is usable",
                        )
                    )
                elif os.path.exists(output.path):
                    statuses.append(
                        VariantOutputStatus(
                            variant.id,
                            output.node_id,
                            output.path,
                            "stale",
                            "old version on disk; inputs changed or file "
                            "damaged, needs recompute for this variant",
                        )
                    )
                else:
                    statuses.append(
                        VariantOutputStatus(
                            variant.id,
                            output.node_id,
                            output.path,
                            "missing",
                            "no published output for this variant yet",
                        )
                    )
        report[variant.id] = statuses
    return report
