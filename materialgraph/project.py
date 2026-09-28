from __future__ import annotations

import json
from pathlib import Path

from .graph import Graph

PROJECT_VERSION = 1


def graph_to_dict(graph: Graph, outputs: dict[str, str] | None = None):
    graph.validate()
    return {
        "format": "materialgraph",
        "version": PROJECT_VERSION,
        "nodes": [
            {"id": node.id, "type": node.type_name, "params": node.params}
            for node in graph.nodes.values()
        ],
        "connections": [
            {
                "source_node": source_id,
                "source_port": source_port,
                "target_node": target_id,
                "target_port": target_port,
            }
            for (target_id, target_port), (source_id, source_port) in sorted(
                graph.connections.items()
            )
        ],
        "outputs": {key: str(value) for key, value in sorted((outputs or {}).items())},
    }


def save_project(path: str | Path, graph: Graph, outputs: dict[str, str] | None = None):
    path = Path(path)
    payload = graph_to_dict(graph, outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_project(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "materialgraph":
        raise ValueError("Not a materialgraph project file")
    if payload.get("version") != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {payload.get('version')}")
    graph = Graph()
    for node in payload["nodes"]:
        graph.add_node(node["type"], node["id"], node.get("params", {}))
    for connection in payload["connections"]:
        graph.connect(
            connection["source_node"],
            connection["target_node"],
            connection["target_port"],
            connection.get("source_port", "output"),
        )
    graph.validate()
    return graph, dict(payload.get("outputs", {}))
