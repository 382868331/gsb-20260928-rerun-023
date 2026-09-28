from __future__ import annotations

from pathlib import Path

from ..graph import Graph
from ..project import save_project
from ..render import render_batch


def build_shared_mask_example(width=32, height=32):
    graph = Graph()
    graph.add_node("coordinates", "base_coords", {"width": width, "height": height})
    graph.add_node(
        "transform_coordinates",
        "moved_coords",
        {"offset_x": 0.08, "offset_y": -0.06, "rotation_degrees": 18.0, "scale_x": 1.15, "scale_y": 0.9},
    )
    graph.connect("base_coords", "moved_coords", "coordinates")
    graph.add_node("circle_mask", "shared_circle", {"radius": 0.28, "softness": 0.04})
    graph.connect("moved_coords", "shared_circle", "coordinates")

    graph.add_node("constant_color", "bg_warm", {"rgba": [35, 28, 22, 255]})
    graph.add_node("constant_color", "fg_red", {"rgba": [224, 74, 52, 255]})
    graph.add_node("mix_color", "red_branch")
    graph.connect("shared_circle", "red_branch", "mask")
    graph.connect("bg_warm", "red_branch", "background")
    graph.connect("fg_red", "red_branch", "foreground")
    graph.add_node("output", "red_output")
    graph.connect("red_branch", "red_output", "image")

    graph.add_node("constant_color", "bg_cool", {"rgba": [18, 30, 42, 255]})
    graph.add_node("constant_color", "fg_blue", {"rgba": [70, 160, 235, 255]})
    graph.add_node("mix_color", "blue_branch")
    graph.connect("shared_circle", "blue_branch", "mask")
    graph.connect("bg_cool", "blue_branch", "background")
    graph.connect("fg_blue", "blue_branch", "foreground")
    graph.add_node("output", "blue_output")
    graph.connect("blue_branch", "blue_output", "image")
    return graph


def write_shared_mask_example(directory: str | Path, width=32, height=32):
    directory = Path(directory)
    graph = build_shared_mask_example(width, height)
    output_paths = {
        "red_output": directory / "red.png",
        "blue_output": directory / "blue.png",
    }
    project_outputs = {
        "red_output": "red.png",
        "blue_output": "blue.png",
    }
    save_project(directory / "project.json", graph, project_outputs)
    render_batch(graph, output_paths)
    return graph, output_paths
