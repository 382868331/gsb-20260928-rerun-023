from __future__ import annotations

import json
import os
from pathlib import Path

from .engine import EvaluationCache, RenderTarget, TextureEngine
from .graph import Connection, Graph

PROJECT_VERSION = 2


def graph_to_dict(
    graph: Graph,
    outputs: dict[str, str] | list[RenderTarget] | None = None,
) -> dict:
    return {
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
        "tracks": [
            {
                "node_id": track.node_id,
                "param": track.param,
                "interpolation": track.interpolation,
                "keyframes": [
                    {"frame": keyframe.frame, "value": keyframe.value}
                    for keyframe in track.keyframes
                ],
            }
            for track in sorted(
                graph.tracks.values(), key=lambda item: (item.node_id, item.param)
            )
        ],
        "outputs": serialize_outputs(outputs or {}),
    }


def serialize_outputs(outputs: dict[str, str] | list[RenderTarget]) -> list[dict]:
    if isinstance(outputs, dict):
        return [
            {"node_id": node_id, "path": path, "frame": 0}
            for node_id, path in sorted(outputs.items())
        ]
    return [
        {
            "node_id": target.output_node,
            "path": target.path,
            "frame": target.frame,
            **(
                {"width": target.width, "height": target.height}
                if target.width is not None or target.height is not None
                else {}
            ),
        }
        for target in outputs
    ]


def graph_from_dict(data: dict) -> tuple[Graph, dict[str, str] | list[RenderTarget]]:
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
        interpolation = item.get("interpolation", "linear")
        for keyframe in item.get("keyframes", []):
            graph.add_keyframe(
                item["node_id"],
                item["param"],
                int(keyframe["frame"]),
                keyframe["value"],
                interpolation=interpolation,
            )
    outputs = parse_outputs(data.get("outputs", {}))
    graph.validate_references()
    return graph, outputs


def parse_outputs(data: dict | list) -> list[RenderTarget]:
    if isinstance(data, dict):
        return [RenderTarget(node_id, path) for node_id, path in sorted(data.items())]
    targets = []
    for item in data:
        targets.append(
            RenderTarget(
                output_node=item["node_id"],
                path=item["path"],
                frame=int(item.get("frame", 0)),
                width=item.get("width"),
                height=item.get("height"),
            )
        )
    return targets


def save_project(
    path: str,
    graph: Graph,
    outputs: dict[str, str] | list[RenderTarget] | None = None,
) -> None:
    document = graph_to_dict(graph, outputs)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_project(path: str) -> tuple[Graph, dict[str, str], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    graph, outputs = graph_from_dict(data)
    return graph, outputs, data


def render_project(
    project_path: str,
    is_cancelled=lambda: False,
    cache: EvaluationCache | None = None,
) -> dict[str, str]:
    project_path = str(Path(project_path))
    project_directory = os.path.dirname(os.path.abspath(project_path))
    graph, outputs, _ = load_project(project_path)
    engine = TextureEngine(graph, cache)
    if isinstance(outputs, dict):
        targets = [
            RenderTarget(node_id, path) for node_id, path in outputs.items()
        ]
    else:
        targets = outputs
    resolved = [
        RenderTarget(
            output_node=target.output_node,
            path=resolve_output_path(target.path, project_directory),
            frame=target.frame,
            width=target.width,
            height=target.height,
        )
        for target in targets
    ]
    engine.render_batch(resolved, is_cancelled=is_cancelled)
    simple_outputs = len({target.output_node for target in resolved}) == len(resolved)
    if simple_outputs and all(
        target.frame == 0 and target.width is None and target.height is None
        for target in resolved
    ):
        return {target.output_node: target.path for target in resolved}
    return {target_key(target): target.path for target in resolved}


def target_key(target: RenderTarget) -> str:
    width = target.width or 0
    height = target.height or 0
    return f"{target.output_node}:f{target.frame}:{width}x{height}:{target.path}"


def resolve_output_path(path: str, project_directory: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(project_directory) / path_obj).resolve())
