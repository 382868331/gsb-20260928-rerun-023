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

PROJECT_VERSION = 2


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

    def frames(self, project_directory: str) -> list[FrameRequest]:
        base = resolve_output_path(self.path, project_directory)
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
    }
    return document


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
    if kind == "static":
        return RenderJob(
            id=document["id"],
            kind=kind,
            node=node_id,
            path=document["path"],
            width=width,
            height=height,
            time=float(document.get("time", 0.0)),
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
        )
    raise ValueError(f"unknown render job kind: {kind}")


def graph_from_dict(data: dict) -> tuple[Graph, dict[str, str], list[RenderJob]]:
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
    graph.validate_references()
    return graph, outputs, jobs


def save_project(
    path: str,
    graph: Graph,
    outputs: dict[str, str] | None = None,
    jobs: list[RenderJob] | None = None,
) -> None:
    document = graph_to_dict(graph, outputs, jobs)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_project(path: str) -> tuple[Graph, dict[str, str], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    graph, outputs, jobs = graph_from_dict(data)
    data = dict(data)
    data["jobs"] = jobs
    return graph, outputs, data


def render_project(
    project_path: str,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
) -> dict[str, str]:
    rendered, _ = render_project_detailed(
        project_path,
        is_cancelled=is_cancelled,
        cache=cache,
        resume=resume,
    )
    return rendered


def render_project_detailed(
    project_path: str,
    is_cancelled: Callable[[], bool] = lambda: False,
    cache: EvaluationCache | None = None,
    resume: bool = True,
) -> tuple[dict[str, str], list]:
    project_path = str(Path(project_path))
    project_directory = os.path.dirname(os.path.abspath(project_path))
    graph, outputs, data = load_project(project_path)
    engine = TextureEngine(graph, cache)
    jobs: list[RenderJob] = data.get("jobs", [])

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


def resolve_output_path(path: str, project_directory: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(project_directory) / path_obj).resolve())
