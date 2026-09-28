from __future__ import annotations

import json
import os
from pathlib import Path

from .engine import EvaluationCache, TextureEngine
from .graph import Connection, Graph

PROJECT_VERSION = 1


def graph_to_dict(graph: Graph, outputs: dict[str, str] | None = None) -> dict:
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
        "outputs": dict(sorted((outputs or {}).items())),
    }


def graph_from_dict(data: dict) -> tuple[Graph, dict[str, str]]:
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
    outputs = dict(data.get("outputs", {}))
    graph.validate_references()
    return graph, outputs


def save_project(path: str, graph: Graph, outputs: dict[str, str] | None = None) -> None:
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
    resolved = {
        node_id: resolve_output_path(path, project_directory)
        for node_id, path in outputs.items()
    }
    engine.render_batch(list(outputs), resolved, is_cancelled=is_cancelled)
    return resolved


def resolve_output_path(path: str, project_directory: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(project_directory) / path_obj).resolve())
