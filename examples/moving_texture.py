from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from texturegraph import RenderTarget
from texturegraph.graph import Graph
from texturegraph.project import render_project, save_project


def build_graph() -> tuple[Graph, list[RenderTarget]]:
    graph = Graph(width=256, height=256)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node("transform", "transform_coordinates")
    graph.add_node("mask", "circle_mask", {"radius": 0.32, "softness": 0.02})
    graph.add_node("fill", "solid_color", {"color": [0.15, 0.65, 0.95, 1.0]})
    graph.add_node("background", "solid_color", {"color": [0.02, 0.03, 0.08, 1.0]})
    graph.add_node("result", "blend_colors")

    graph.connect("uv", "transform", "coordinates")
    graph.connect("transform", "mask", "coordinates")
    graph.connect("fill", "result", "foreground")
    graph.connect("background", "result", "background")
    graph.connect("mask", "result", "mask")

    graph.add_keyframe("transform", "offset_x", 0, -0.7)
    graph.add_keyframe("transform", "offset_x", 3, 0.7)

    targets = [
        RenderTarget("result", "outputs/moving-preview.png", frame=0),
        *[
            RenderTarget("result", f"outputs/moving-frame-{frame}.png", frame=frame)
            for frame in range(4)
        ],
        RenderTarget("result", "outputs/moving-hd.png", frame=3, width=512, height=512),
    ]
    return graph, targets


def main() -> None:
    root = ROOT / "examples"
    project_path = root / "moving-texture-project.json"
    graph, targets = build_graph()
    save_project(str(project_path), graph, targets)
    rendered = render_project(str(project_path))
    for path in rendered.values():
        print(path)


if __name__ == "__main__":
    main()
