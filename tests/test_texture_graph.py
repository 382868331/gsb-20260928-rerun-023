from pathlib import Path

import pytest
from PIL import Image

from texturegraph import (
    CycleError,
    EvaluationCache,
    Graph,
    RenderCancelled,
    RenderContext,
    RenderTarget,
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


def test_animated_mask_is_not_reused_across_frames():
    graph, _ = build_graph()
    graph.add_keyframe("shared_mask", "radius", 0, 0.2)
    graph.add_keyframe("shared_mask", "radius", 1, 0.8)
    cache = EvaluationCache()

    frame_zero = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_a", frame_zero, frame=0)
    frame_one = RenderContext()
    value_one = TextureEngine(graph, cache).evaluate("branch_a", frame_one, frame=1)

    assert "shared_mask" not in frame_one.cache_hits
    assert "uv" in frame_one.cache_hits
    assert "circle_transform" in frame_one.cache_hits
    assert frame_one.evaluated == ["shared_mask", "branch_a"]
    assert value_one.shape == (256, 256, 4)

    frame_zero_again = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_a", frame_zero_again, frame=0)
    assert frame_zero_again.evaluated == []
    assert "shared_mask" in frame_zero_again.cache_hits


def test_resolution_results_are_cached_separately_and_have_correct_size():
    graph, _ = build_graph()
    cache = EvaluationCache()
    context = RenderContext()

    small = TextureEngine(graph, cache).evaluate(
        "branch_a", context, width=16, height=8
    )
    high = TextureEngine(graph, cache).evaluate(
        "branch_a", RenderContext(), width=64, height=32
    )
    small_again_context = RenderContext()
    small_again = TextureEngine(graph, cache).evaluate(
        "branch_a", small_again_context, width=16, height=8
    )

    assert small.shape == (8, 16, 4)
    assert high.shape == (32, 64, 4)
    assert small_again.shape == (8, 16, 4)
    assert "shared_mask" in small_again_context.cache_hits


def test_modifying_one_animated_branch_only_invalidates_that_frame_and_branch():
    graph, _ = build_graph()
    graph.add_keyframe("red", "color", 0, [0.2, 0.2, 0.9, 1.0], "step")
    graph.add_keyframe("red", "color", 2, [0.9, 0.9, 0.1, 1.0], "step")
    cache = EvaluationCache()

    for frame in (0, 1, 2):
        TextureEngine(graph, cache).evaluate(
            "branch_a", RenderContext(), frame=frame
        )
        TextureEngine(graph, cache).evaluate(
            "branch_b", RenderContext(), frame=frame
        )

    graph.set_param("gold", "color", [0.1, 0.9, 0.8, 1.0])
    frame_one_b = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_b", frame_one_b, frame=1)
    frame_zero_a = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_a", frame_zero_a, frame=0)

    assert frame_one_b.evaluated == ["gold", "branch_b"]
    assert frame_zero_a.evaluated == []


def test_cancelled_multires_frame_batch_can_resume_without_skipping(tmp_path):
    graph, _ = build_graph()
    graph.add_keyframe("circle_transform", "offset_x", 0, -0.5)
    graph.add_keyframe("circle_transform", "offset_x", 2, 0.5)
    cache = EvaluationCache()

    targets = [
        RenderTarget("branch_a", str(tmp_path / "preview.png"), frame=0),
        RenderTarget("branch_a", str(tmp_path / "f0.png"), frame=0, width=32, height=16),
        RenderTarget("branch_a", str(tmp_path / "f1.png"), frame=1, width=32, height=16),
        RenderTarget("branch_b", str(tmp_path / "f2.png"), frame=2, width=32, height=16),
        RenderTarget("branch_b", str(tmp_path / "hd.png"), frame=2, width=64, height=64),
    ]
    cancel_after = iter([False] * 7 + [True])

    with pytest.raises(RenderCancelled):
        TextureEngine(graph, cache).render_batch(targets, is_cancelled=lambda: next(cancel_after))

    assert not any(Path(target.path).exists() for target in targets)

    resumed = []
    original_evaluate = TextureEngine.evaluate

    def tracked_evaluate(
        self,
        output_node,
        context=None,
        frame=0,
        width=None,
        height=None,
    ):
        resumed.append((output_node, frame, width, height))
        return original_evaluate(
            self,
            output_node,
            context,
            frame=frame,
            width=width,
            height=height,
        )

    TextureEngine.evaluate = tracked_evaluate
    try:
        TextureEngine(graph, cache).render_batch(targets)
    finally:
        TextureEngine.evaluate = original_evaluate

    sizes = {}
    for target in targets:
        assert Path(target.path).exists()
        with Image.open(target.path) as image:
            sizes[Path(target.path).name] = image.size
    assert sizes == {
        "preview.png": (256, 256),
        "f0.png": (32, 16),
        "f1.png": (32, 16),
        "f2.png": (32, 16),
        "hd.png": (64, 64),
    }
    assert [item[0] for item in resumed] == [target.output_node for target in targets]
    assert [item[1] for item in resumed] == [target.frame for target in targets]
    assert [item[2:] for item in resumed] == [
        (target.width, target.height) for target in targets
    ]
