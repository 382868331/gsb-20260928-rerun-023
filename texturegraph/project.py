from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .animation import track_from_dict, tracks_to_list
from .engine import (
    EvaluationCache,
    FrameOutput,
    FrameRequest,
    TextureEngine,
)
from .graph import Connection, Graph
from .variants import Variant, apply_variant

PROJECT_VERSION = 3


@dataclass(frozen=True)
class RenderJob:
    """A declared output target.

    Kinds:
      static   - one image at one time (preview or high-resolution still)
      sequence - one PNG per frame across a time range
    """

    id: str
    kind: str
    node: str
    path: str
    width: int
    height: int
    time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    frame_count: int = 1
    # When the project declares variants, this job runs once per listed
    # variant id (None = every variant).  The path may contain {variant}.
    variants: tuple[str, ...] | None = None

    def frames(
        self, project_directory: str, variant_id: str | None = None
    ) -> list[FrameRequest]:
        path = self.path if variant_id is None else self.path.format(variant=variant_id)
        base = resolve_output_path(path, project_directory)
        if self.kind == "static":
            output = FrameOutput(self.node, base, self.width, self.height)
            return [FrameRequest(float(self.time), (output,))]
        if self.kind == "sequence":
            if self.frame_count < 1:
                raise ValueError(f"sequence {self.id} needs frame_count >= 1")
            requests: list[FrameRequest] = []
            for index in range(self.frame_count):
                if self.frame_count == 1:
                    time = float(self.start_time)
                else:
                    fraction = index / (self.frame_count - 1)
                    time = (
                        float(self.start_time)
                        + (float(self.end_time) - float(self.start_time)) * fraction
                    )
                path = sequence_frame_path(base, index, self.frame_count)
                output = FrameOutput(self.node, path, self.width, self.height)
                requests.append(FrameRequest(time, (output,)))
            return requests
        raise ValueError(f"unknown render job kind: {self.kind}")


def sequence_frame_path(pattern_path: str, index: int, frame_count: int) -> str:
    stem, extension = os.path.splitext(pattern_path)
    extension = extension or ".png"
    digits = 4
    return f"{stem}.{index:0{digits}d}{extension}"


def graph_to_dict(
    graph: Graph,
    outputs: dict[str, str] | None = None,
    jobs: list[RenderJob] | None = None,
    variants: list[Variant] | None = None,
) -> dict:
    document = {
        "version": PROJECT_VERSION,
        "width": graph.width,
        "height": graph.height,
        "nodes": [
            {"id": node.id, "type": node.type, "params": node.params}
            for node in sorted(graph.nodes.values(), key=lambda item: item.id)
        ],
        "connections": [
            {
                "source_node": connection.source_node,
                "source_port": connection.source_port,
                "target_node": connection.target_node,
                "target_port": connection.target_port,
            }
            for connection in graph.connections
        ],
        "tracks": tracks_to_list(graph.tracks.values()),
        "outputs": dict(sorted((outputs or {}).items())),
        "jobs": [job_to_dict(job) for job in sorted(jobs or [], key=lambda item: item.id)],
        "variants": [variant_to_dict(v) for v in sorted(variants or [], key=lambda item: item.id)],
    }
    return document


def variant_to_dict(variant: Variant) -> dict:
    document = {"id": variant.id, "overrides": variant.overrides}
    if variant.label is not None:
        document["label"] = variant.label
    return document


def variant_from_dict(document: dict) -> Variant:
    overrides = {
        str(node_id): dict(params)
        for node_id, params in document.get("overrides", {}).items()
    }
    return Variant(
        id=document["id"],
        overrides=overrides,
        label=document.get("label"),
    )


def job_to_dict(job: RenderJob) -> dict:
    document = {
        "id": job.id,
        "kind": job.kind,
        "node": job.node,
        "path": job.path,
        "width": job.width,
        "height": job.height,
    }
    if job.kind == "static":
        document["time"] = float(job.time)
    else:
        document["start_time"] = float(job.start_time)
        document["end_time"] = float(job.end_time)
        document["frame_count"] = int(job.frame_count)
    if job.variants is not None:
        document["variants"] = list(job.variants)
    return document


def job_from_dict(document: dict, graph: Graph) -> RenderJob:
    node_id = document["node"]
    if node_id not in graph.nodes:
        raise ValueError(f"job references unknown node: {node_id}")
    width = int(document["width"])
    height = int(document["height"])
    if width <= 0 or height <= 0:
        raise ValueError("job width and height must be positive")
    kind = document.get("kind", "static")
    variants = document.get("variants")
    variants = tuple(variants) if variants is not None else None
    if kind == "static":
        return RenderJob(
            id=document["id"],
            kind=kind,
            node=node_id,
            path=document["path"],
            width=width,
            height=height,
            time=float(document.get("time", 0.0)),
            variants=variants,
        )
    if kind == "sequence":
        return RenderJob(
            id=document["id"],
            kind=kind,
            node=node_id,
            path=document["path"],
            width=width,
            height=height,
            start_time=float(document["start_time"]),
            end_time=float(document["end_time"]),
            frame_count=int(document["frame_count"]),
            variants=variants,
        )
    raise ValueError(f"unknown render job kind: {kind}")


def graph_from_dict(
    data: dict,
) -> tuple[Graph, dict[str, str], list[RenderJob], list[Variant]]:
    graph = Graph(width=int(data["width"]), height=int(data["height"]))
    if graph.width <= 0 or graph.height <= 0:
        raise ValueError("width and height must be positive")
    for item in data.get("nodes", []):
        graph.add_node(item["id"], item["type"], item.get("params", {}))
    for item in data.get("connections", []):
        graph.connections.append(
            Connection(
                item["source_node"],
                item.get("source_port", "value"),
                item["target_node"],
                item["target_port"],
            )
        )
    for item in data.get("tracks", []):
        graph.add_track(track_from_dict(item))
    outputs = dict(data.get("outputs", {}))
    jobs = [job_from_dict(item, graph) for item in data.get("jobs", [])]
    variants = [variant_from_dict(item) for item in data.get("variants", [])]
    graph.validate_references()
    return graph, outputs, jobs, variants


def save_project(
    path: str,
    graph: Graph,
    outputs: dict[str, str] | None = None,
    jobs: list[RenderJob] | None = None,
    variants: list[Variant] | None = None,
) -> None:
    document = graph_to_dict(graph, outputs, jobs, variants)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_project(path: str) -> tuple[Graph, dict[str, str], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    graph, outputs, jobs, variants = graph_from_dict(data)
    data = dict(data)
    data["jobs"] = jobs
    data["variants"] = variants
    return graph, outputs, data


def render_project(
    project_path: str,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
    only_variants: list[str] | None = None,
) -> dict[str, str]:
    rendered, _ = render_project_detailed(
        project_path,
        is_cancelled=is_cancelled,
        cache=cache,
        resume=resume,
        only_variants=only_variants,
    )
    return rendered


def render_project_detailed(
    project_path: str,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
    only_variants: list[str] | None = None,
) -> tuple[dict[str, str], list]:
    project_path = str(Path(project_path))
    project_directory = os.path.dirname(os.path.abspath(project_path))
    graph, outputs, data = load_project(project_path)
    jobs: list[RenderJob] = data.get("jobs", [])
    variants: list[Variant] = data.get("variants", [])
    cache = cache or EvaluationCache()

    if data.get("version", 1) < 2 and not jobs and outputs:
        jobs = [
            RenderJob(
                id=node_id,
                kind="static",
                node=node_id,
                path=path,
                width=graph.width,
                height=graph.height,
                time=0.0,
            )
            for node_id, path in outputs.items()
        ]

    if variants:
        return _render_project_variants(
            graph,
            jobs,
            variants,
            project_directory,
            is_cancelled=is_cancelled,
            cache=cache,
            resume=resume,
            only_variants=only_variants,
        )

    engine = TextureEngine(graph, cache)

    # Merge job requests by time (paths make outputs unique), but only at the
    # exact same time.  Sequence frames at distinct times stay separate, so a
    # multi-frame job cannot collapse onto another job timestamp.
    frames_by_time: dict[float, dict[str, FrameOutput]] = {}
    order: list[float] = []
    for job in jobs:
        for request in job.frames(project_directory):
            time = float(request.time)
            if time not in frames_by_time:
                frames_by_time[time] = {}
                order.append(time)
            for output in request.outputs:
                frames_by_time[time][output.path] = output

    frames = [
        FrameRequest(time, tuple(frames_by_time[time].values()))
        for time in order
    ]
    results = engine.render_frames(frames, is_cancelled=is_cancelled, resume=resume)
    rendered: dict[str, str] = {}
    for result in results:
        rendered.update(result.outputs)
    return rendered, results


def _render_project_variants(
    graph: Graph,
    jobs: list[RenderJob],
    variants: list[Variant],
    project_directory: str,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
    only_variants: list[str] | None = None,
) -> tuple[dict[str, str], list]:
    """Render every job for each selected parameter scheme.

    One shared evaluation cache means inputs and intermediate results that
    no override touches are computed once for the whole batch; each scheme
    still publishes PNGs stamped with its own variant id.
    """
    import dataclasses

    from .variants import select_variants

    selected = select_variants(variants, only_variants)
    cache = cache or EvaluationCache()

    # Every (job, variant) pair must resolve to a distinct path so schemes
    # cannot replace each other's published files.
    seen_paths: dict[str, str] = {}
    for job in jobs:
        job_variants = select_variants(variants, job.variants)
        for variant in job_variants:
            if variant not in selected:
                continue
            for request in job.frames(project_directory, variant.id):
                for output in request.outputs:
                    path = os.path.abspath(output.path)
                    owner = seen_paths.get(path)
                    if owner is not None and owner != variant.id:
                        raise ValueError(
                            f"job {job.id!r} path collides between schemes "
                            f"{owner!r} and {variant.id!r}: {path}"
                        )
                    seen_paths[path] = variant.id

    rendered: dict[str, str] = {}
    results: list = []
    for variant in selected:
        derived = apply_variant(graph, variant)
        engine = TextureEngine(derived, cache, variant_id=variant.id)
        frames_by_time: dict[float, dict[str, FrameOutput]] = {}
        order: list[float] = []
        for job in jobs:
            job_variants = select_variants(variants, job.variants)
            if variant not in job_variants:
                continue
            for request in job.frames(project_directory, variant.id):
                time = float(request.time)
                if time not in frames_by_time:
                    frames_by_time[time] = {}
                    order.append(time)
                for output in request.outputs:
                    frames_by_time[time][output.path] = output
        frames = [
            FrameRequest(time, tuple(frames_by_time[time].values()))
            for time in order
        ]
        frame_results = engine.render_frames(
            frames, is_cancelled=is_cancelled, resume=resume
        )
        for result in frame_results:
            result = dataclasses.replace(result, variant=variant.id)
            results.append(result)
            rendered.update(result.outputs)
    return rendered, results


def resolve_output_path(path: str, project_directory: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(project_directory) / path_obj).resolve())
