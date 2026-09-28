from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from texturegraph.graph import Graph
from texturegraph.project import render_project, save_project


def build_graph() -> tuple[Graph, dict[str, str]]:
    graph = Graph(width=256, height=256)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node(
        "circle_transform",
        "transform_coordinates",
        {"scale_x": 1.25, "scale_y": 0.75, "rotation_degrees": 20},
    )
    graph.add_node("shared_mask", "circle_mask", {"radius": 0.45, "softness": 0.03})

    graph.add_node("red", "solid_color", {"color": [0.85, 0.12, 0.10, 1.0]})
    graph.add_node("dark_blue", "solid_color", {"color": [0.04, 0.08, 0.20, 1.0]})
    graph.add_node("branch_a", "blend_colors")

    graph.add_node("gold", "solid_color", {"color": [0.95, 0.72, 0.18, 1.0]})
    graph.add_node("dark_green", "solid_color", {"color": [0.03, 0.13, 0.08, 1.0]})
    graph.add_node("branch_b", "blend_colors")

    graph.connect("uv", "circle_transform", "coordinates")
    graph.connect("circle_transform", "shared_mask", "coordinates")

    graph.connect("red", "branch_a", "foreground")
    graph.connect("dark_blue", "branch_a", "background")
    graph.connect("shared_mask", "branch_a", "mask")

    graph.connect("gold", "branch_b", "foreground")
    graph.connect("dark_green", "branch_b", "background")
    graph.connect("shared_mask", "branch_b", "mask")

    outputs = {
        "branch_a": "outputs/red-circle.png",
        "branch_b": "outputs/gold-circle.png",
    }
    return graph, outputs


def main() -> None:
    root = ROOT / "examples"
    project_path = root / "shared-mask-project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs)
    rendered = render_project(str(project_path))
    for path in rendered.values():
        print(path)


if __name__ == "__main__":
    main()
