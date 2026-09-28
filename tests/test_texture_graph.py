from pathlib import Path

import pytest
from PIL import Image

from texturegraph import (
    CycleError,
    EvaluationCache,
    Graph,
    RenderCancelled,
    RenderContext,
    TextureEngine,
    load_project,
    render_project,
    save_project,
)
from examples.shared_mask import build_graph


def test_shared_mask_is_evaluated_once_and_cached_across_branches():
    graph, _ = build_graph()
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)

    first = RenderContext()
    engine.evaluate("branch_a", first)
    second = RenderContext()
    engine.evaluate("branch_b", second)

    assert first.evaluated.count("shared_mask") == 1
    assert "shared_mask" in second.cache_hits
    assert "circle_transform" in second.cache_hits
    assert second.evaluated == ["dark_green", "gold", "branch_b"]


def test_changing_one_color_only_invalidates_related_branch():
    graph, _ = build_graph()
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)

    first = RenderContext()
    engine.evaluate("branch_a", first)
    second = RenderContext()
    engine.evaluate("branch_b", second)

    graph.set_param("red", "color", [0.2, 0.2, 0.9, 1.0])
    changed_a = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_a", changed_a)

    changed_b = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_b", changed_b)

    assert changed_a.evaluated == ["red", "branch_a"]
    assert "shared_mask" in changed_a.cache_hits
    assert changed_b.evaluated == []
    assert "branch_b" in changed_b.cache_hits


def test_saved_project_renders_identical_images(tmp_path):
    project_path = tmp_path / "project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs)

    rendered_one = render_project(str(project_path))
    first_bytes = {
        node_id: Path(path).read_bytes() for node_id, path in rendered_one.items()
    }

    loaded_graph, loaded_outputs, _ = load_project(str(project_path))
    second_project = tmp_path / "copied-project.json"
    save_project(str(second_project), loaded_graph, loaded_outputs)
    rendered_two = render_project(str(second_project))

    assert first_bytes.keys() == rendered_two.keys()
    for node_id, path in rendered_two.items():
        assert Path(path).read_bytes() == first_bytes[node_id]


def test_cycle_is_reported_before_evaluation():
    graph = Graph(width=4, height=4)
    graph.add_node("mask_a", "combine_masks")
    graph.add_node("mask_b", "combine_masks")
    graph.connect("mask_a", "mask_b", "a")
    graph.connect("mask_b", "mask_a", "a")

    with pytest.raises(CycleError):
        TextureEngine(graph).evaluate("mask_a", RenderContext())


def test_cancelled_batch_preserves_existing_outputs(tmp_path):
    graph, _ = build_graph()
    output_a = tmp_path / "a.png"
    output_b = tmp_path / "b.png"
    engine = TextureEngine(graph)
    engine.render_batch(
        ["branch_a", "branch_b"],
        {"branch_a": str(output_a), "branch_b": str(output_b)},
    )
    valid_a = output_a.read_bytes()
    valid_b = output_b.read_bytes()

    graph.set_param("red", "color", [1.0, 1.0, 1.0, 1.0])
    calls = iter([False] * 5 + [True])

    with pytest.raises(RenderCancelled):
        TextureEngine(graph).render_batch(
            ["branch_a", "branch_b"],
            {"branch_a": str(output_a), "branch_b": str(output_b)},
            lambda: next(calls),
        )

    assert output_a.read_bytes() == valid_a
    assert output_b.read_bytes() == valid_b


def test_example_outputs_are_valid_rgba_pictures(tmp_path):
    project_path = tmp_path / "project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs)
    rendered = render_project(str(project_path))
    for path in rendered.values():
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size == (256, 256)
