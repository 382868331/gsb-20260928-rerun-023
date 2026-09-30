from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from texturegraph.animation import Keyframe, ParameterTrack
from texturegraph.graph import Graph
from texturegraph.project import RenderJob, render_project, save_project

FRAME_COUNT = 5


def build_graph() -> tuple[Graph, dict[str, str]]:
    graph = Graph(width=64, height=64)
    graph.add_node("uv", "uv_coordinates")
    # The mask moves over time: both branches share this moving mask.
    graph.add_node(
        "circle_transform",
        "transform_coordinates",
        {"scale_x": 1.25, "scale_y": 0.75, "rotation_degrees": 20, "offset_x": -0.5},
    )
    graph.add_node("shared_mask", "circle_mask", {"radius": 0.45, "softness": 0.03})
    graph.add_track(
        ParameterTrack(
            "circle_transform",
            "offset_x",
            (Keyframe(0.0, -0.5), Keyframe(1.0, 0.5)),
        )
    )

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


def build_jobs(graph: Graph) -> list[RenderJob]:
    return [
        # Fast static preview of both branches at the first instant.
        RenderJob(
            id="preview_a",
            kind="static",
            node="branch_a",
            path="outputs/preview/branch_a.png",
            width=graph.width,
            height=graph.height,
            time=0.0,
        ),
        RenderJob(
            id="preview_b",
            kind="static",
            node="branch_b",
            path="outputs/preview/branch_b.png",
            width=graph.width,
            height=graph.height,
            time=0.0,
        ),
        # Continuous frames for the moving shared mask.
        RenderJob(
            id="sequence_a",
            kind="sequence",
            node="branch_a",
            path="outputs/frames/branch_a.png",
            width=graph.width,
            height=graph.height,
            start_time=0.0,
            end_time=1.0,
            frame_count=FRAME_COUNT,
        ),
        RenderJob(
            id="sequence_b",
            kind="sequence",
            node="branch_b",
            path="outputs/frames/branch_b.png",
            width=graph.width,
            height=graph.height,
            start_time=0.0,
            end_time=1.0,
            frame_count=FRAME_COUNT,
        ),
        # A single high-resolution still on only one branch.
        RenderJob(
            id="hires_a",
            kind="static",
            node="branch_a",
            path="outputs/hires/branch_a.png",
            width=512,
            height=512,
            time=0.5,
        ),
    ]


def main() -> None:
    root = ROOT / "examples"
    project_path = root / "shared-mask-project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs, build_jobs(graph))
    rendered = render_project(str(project_path))
    for path in sorted(rendered.values()):
        print(path)


if __name__ == "__main__":
    main()
