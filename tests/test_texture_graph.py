from pathlib import Path

import pytest
from PIL import Image

from texturegraph import (
    CycleError,
    EvaluationCache,
    FrameOutput,
    FrameRequest,
    Graph,
    Keyframe,
    ParameterTrack,
    RenderCancelled,
    RenderContext,
    RenderJob,
    TextureEngine,
    load_project,
    render_project,
    save_project,
)
from texturegraph.project import sequence_frame_path
from examples.shared_mask import build_graph


def test_shared_mask_is_evaluated_once_and_cached_across_branches():
    graph, _ = build_graph()
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)

    first = RenderContext(time=0.0)
    engine.evaluate("branch_a", first, time=0.0)
    second = RenderContext(time=0.0)
    engine.evaluate("branch_b", second, time=0.0)

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
    TextureEngine(graph, cache).evaluate("branch_b", RenderContext())

    graph.set_param("red", "color", [0.2, 0.2, 0.9, 1.0])
    changed_a = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_a", changed_a)

    changed_b = RenderContext()
    TextureEngine(graph, cache).evaluate("branch_b", changed_b)

    assert changed_a.evaluated == ["red", "branch_a"]
    assert "shared_mask" in changed_a.cache_hits
    assert changed_b.evaluated == []
    assert "branch_b" in changed_b.cache_hits


def test_moving_mask_is_recomputed_per_frame_and_never_reused():
    graph, _ = build_graph()
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)

    frame_zero = RenderContext(time=0.0)
    engine.evaluate("branch_a", frame_zero, time=0.0)
    frame_one = RenderContext(time=0.5)
    engine.evaluate("branch_a", frame_one, time=0.5)

    # The animated transform, its mask and the dependent branch change;
    # coordinate roots are size-dependent constants and recomputed too.
    assert set(frame_one.evaluated) == {
        "circle_transform",
        "shared_mask",
        "branch_a",
    }
    # Static colors are still shared across times.
    assert "red" in frame_one.cache_hits
    assert "dark_blue" in frame_one.cache_hits

    frame_zero_again = RenderContext(time=0.0)
    engine.evaluate("branch_a", frame_zero_again, time=0.0)
    assert frame_zero_again.evaluated == []


def test_shared_mask_in_one_frame_is_evaluated_once_for_both_outputs(tmp_path):
    graph, _ = build_graph()
    outputs = tuple(
        FrameOutput(node, str(tmp_path / f"{node}.png"), 32, 32)
        for node in ("branch_a", "branch_b")
    )
    context = RenderContext(time=0.25)
    engine = TextureEngine(graph, EvaluationCache())
    scope = {}
    engine.evaluate("branch_a", context, 32, 32, 0.25, scope)
    engine.evaluate("branch_b", context, 32, 32, 0.25, scope)
    assert context.evaluated.count("shared_mask") == 1
    assert context.evaluated.count("circle_transform") == 1


def test_resume_recomputes_changed_branch_and_reuses_unaffected_output(tmp_path):
    graph, _ = build_graph()
    frames = [
        FrameRequest(
            0.0,
            (
                FrameOutput("branch_a", str(tmp_path / "a.png"), 32, 32),
                FrameOutput("branch_b", str(tmp_path / "b.png"), 32, 32),
            ),
        )
    ]
    TextureEngine(graph, EvaluationCache()).render_frames(frames, resume=False)
    original_b = (tmp_path / "b.png").read_bytes()

    graph.set_param("red", "color", [0.1, 0.9, 0.2, 1.0])
    results = TextureEngine(graph, EvaluationCache()).render_frames(frames)

    result = results[0]
    assert list(result.recomputed) == [str(tmp_path / "a.png")]
    assert list(result.reused) == [str(tmp_path / "b.png")]
    assert "上游依赖签名不匹配" in result.reasons[str(tmp_path / "a.png")]
    assert "像素指纹均匹配" in result.reasons[str(tmp_path / "b.png")]
    assert (tmp_path / "b.png").read_bytes() == original_b
    assert (tmp_path / "a.png").read_bytes() != original_b


def test_resume_recomputes_only_outputs_using_changed_shared_input(tmp_path):
    graph, _ = build_graph()
    frames = [
        FrameRequest(
            0.0,
            (
                FrameOutput("branch_a", str(tmp_path / "a.png"), 32, 32),
                FrameOutput("branch_b", str(tmp_path / "b.png"), 32, 32),
            ),
        )
    ]
    TextureEngine(graph, EvaluationCache()).render_frames(frames, resume=False)

    graph.set_param("shared_mask", "radius", 0.2)
    result = TextureEngine(graph, EvaluationCache()).render_frames(frames)[0]

    assert set(result.recomputed) == {
        str(tmp_path / "a.png"),
        str(tmp_path / "b.png"),
    }
    assert all("上游依赖签名不匹配" in reason for reason in result.reasons.values())


def test_resume_detects_tampered_pixels_even_when_size_matches(tmp_path):
    graph, _ = build_graph()
    output = FrameOutput("branch_a", str(tmp_path / "a.png"), 32, 32)
    frames = [FrameRequest(0.0, (output,))]
    TextureEngine(graph, EvaluationCache()).render_frames(frames, resume=False)

    with Image.open(output.path) as image:
        signature = image.text["texturegraph_node_signature"]
        digest = image.text["texturegraph_pixel_sha256"]
        changed = Image.new("RGBA", image.size)
        changed.paste(image.copy())
        changed.load()[0, 0] = (0, 255, 0, 255)
        changed.save(
            output.path,
            format="PNG",
            pnginfo=_png_metadata(signature, digest),
        )

    result = TextureEngine(graph, EvaluationCache()).render_frames(frames)[0]
    assert result.skipped is False
    assert output.path in result.recomputed
    assert "像素内容指纹不匹配" in result.reasons[output.path]


def test_same_filename_and_size_at_same_time_keeps_distinct_content(tmp_path):
    red_graph, _ = build_graph()
    green_graph, _ = build_graph()
    green_graph.set_param("red", "color", [0.1, 0.9, 0.2, 1.0])

    red_path = tmp_path / "red" / "branch.png"
    green_path = tmp_path / "green" / "branch.png"
    red_request = [
        FrameRequest(0.0, (FrameOutput("branch_a", str(red_path), 32, 32),))
    ]
    green_request = [
        FrameRequest(0.0, (FrameOutput("branch_a", str(green_path), 32, 32),))
    ]

    TextureEngine(red_graph, EvaluationCache()).render_frames(
        red_request, resume=False
    )
    TextureEngine(green_graph, EvaluationCache()).render_frames(
        green_request, resume=False
    )

    assert red_path.read_bytes() != green_path.read_bytes()
    assert all(
        result.skipped
        for result in TextureEngine(green_graph, EvaluationCache()).render_frames(
            green_request
        )
    )


def test_conflicting_outputs_at_same_path_and_time_are_rejected(tmp_path):
    graph, _ = build_graph()
    path = str(tmp_path / "branch.png")
    frame = FrameRequest(
        0.0,
        (
            FrameOutput("branch_a", path, 32, 32),
            FrameOutput("branch_b", path, 32, 32),
        ),
    )

    with pytest.raises(ValueError, match="same path"):
        TextureEngine(graph, EvaluationCache()).render_frames(
            [frame], resume=False
        )


def _png_metadata(signature: str, digest: str):
    from PIL.PngImagePlugin import PngInfo

    metadata = PngInfo()
    metadata.add_text("texturegraph_node_signature", signature)
    metadata.add_text("texturegraph_node_id", "branch_a")
    metadata.add_text("texturegraph_pixel_sha256", digest)
    return metadata


def test_same_frame_at_different_resolutions_never_share_values():
    graph, _ = build_graph()
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)

    preview = RenderContext(time=0.5)
    engine.evaluate("branch_a", preview, width=64, height=64, time=0.5)
    hires = RenderContext(time=0.5)
    engine.evaluate("branch_a", hires, width=256, height=256, time=0.5)

    # Every node is recomputed at the high resolution; the preview cache
    # cannot satisfy a different resolution.
    assert set(hires.evaluated) == {
        "uv",
        "circle_transform",
        "shared_mask",
        "red",
        "dark_blue",
        "branch_a",
    }
    assert hires.cache_hits == []

    hires_again = RenderContext(time=0.5)
    engine.evaluate("branch_a", hires_again, width=256, height=256, time=0.5)
    assert hires_again.evaluated == []


def test_animated_parameter_on_one_branch_only_invalidates_that_branch():
    graph, _ = build_graph()
    # Remove the shared moving-mask track so only one branch is animated.
    graph.tracks.clear()
    graph.add_track(
        ParameterTrack(
            "red",
            "color",
            (Keyframe(0.0, [0.85, 0.12, 0.10, 1.0]), Keyframe(1.0, [0.1, 0.9, 0.2, 1.0])),
        )
    )
    cache = EvaluationCache()
    engine = TextureEngine(graph, cache)
    engine.evaluate("branch_a", RenderContext(time=0.0), time=0.0)
    engine.evaluate("branch_b", RenderContext(time=0.0), time=0.0)

    later_a = RenderContext(time=1.0)
    engine.evaluate("branch_a", later_a, time=1.0)
    assert later_a.evaluated == ["red", "branch_a"]
    assert "shared_mask" in later_a.cache_hits

    later_b = RenderContext(time=1.0)
    engine.evaluate("branch_b", later_b, time=1.0)
    # branch_b depends on the (time-invariant here) shared mask and its own
    # colors; nothing on its path is animated at t=1 yet for its inputs.
    assert later_b.evaluated == []


def test_saved_project_renders_identical_images(tmp_path):
    project_path = tmp_path / "project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs)

    rendered_one = render_project(str(project_path), resume=False)
    first_bytes = {
        node_id: Path(path).read_bytes() for node_id, path in rendered_one.items()
    }

    loaded_graph, loaded_outputs, _ = load_project(str(project_path))
    second_project = tmp_path / "copied-project.json"
    save_project(str(second_project), loaded_graph, loaded_outputs)
    rendered_two = render_project(str(second_project), resume=False)

    assert set(first_bytes) == set(rendered_two)
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

    with pytest.raises(RenderCancelled):
        TextureEngine(graph).render_batch(
            ["branch_a", "branch_b"],
            {"branch_a": str(output_a), "branch_b": str(output_b)},
            iter([False] * 11 + [True]).__next__,
        )

    assert output_a.read_bytes() == valid_a
    assert output_b.read_bytes() == valid_b


def _moving_frame_batch(tmp_path, width: int = 32, hires_width: int = 64):
    graph, _ = build_graph()
    frames = []
    for time in (0.0, 1.0):
        outputs = tuple(
            FrameOutput(
                node,
                str(tmp_path / f"frame_{time:g}_{node}.png"),
                width,
                width,
            )
            for node in ("branch_a", "branch_b")
        )
        frames.append(FrameRequest(time, outputs))
    hires = FrameOutput(
        "branch_a", str(tmp_path / "hires_a.png"), hires_width, hires_width
    )
    frames.append(FrameRequest(0.5, (hires,)))
    return graph, frames
    return graph, frames


def test_frame_batch_can_cancel_and_resume_without_dropping_frames(tmp_path):
    graph, frames = _moving_frame_batch(tmp_path)
    cache = EvaluationCache()

    # Let frame zero finish and publish; cancel at frame one.
    checks = iter([False] * 21 + [True])

    with pytest.raises(RenderCancelled):
        TextureEngine(graph, cache).render_frames(
            frames,
            lambda: next(checks, True),
            resume=False,
        )

    assert (tmp_path / "frame_0_branch_a.png").exists()
    assert (tmp_path / "frame_0_branch_b.png").exists()
    assert not (tmp_path / "frame_1_branch_a.png").exists()
    assert not (tmp_path / "hires_a.png").exists()

    results = TextureEngine(graph, cache).render_frames(frames)
    statuses = {result.time: result.skipped for result in results}
    assert statuses == {0.0: True, 1.0: False, 0.5: False}

    for node in ("branch_a", "branch_b"):
        with Image.open(tmp_path / f"frame_0_{node}.png") as image:
            assert image.size == (32, 32)
    with Image.open(tmp_path / "hires_a.png") as image:
        assert image.size == (64, 64)

    # A second resume is a no-op: nothing is rendered again.
    idle = TextureEngine(graph, EvaluationCache()).render_frames(frames)
    assert all(result.skipped for result in idle)


def test_resume_checks_resolution_and_replaces_wrong_sized_png(tmp_path):
    graph, frames = _moving_frame_batch(tmp_path)
    TextureEngine(graph, EvaluationCache()).render_frames(frames)

    # Corrupt the resume marker by publishing a different resolution at one
    # path; resume must detect the size mismatch and render that output again.
    wrong = FrameOutput("branch_a", str(tmp_path / "frame_0_branch_a.png"), 16, 16)
    TextureEngine(graph, EvaluationCache()).render_frames(
        [FrameRequest(0.0, (wrong,))], resume=False
    )
    results = TextureEngine(graph, EvaluationCache()).render_frames(frames)
    assert results[0].skipped is False
    with Image.open(tmp_path / "frame_0_branch_a.png") as image:
        assert image.size == (32, 32)


def test_frames_actually_change_with_the_moving_mask(tmp_path):
    graph, _ = build_graph()
    engine = TextureEngine(graph, EvaluationCache())
    paths = [tmp_path / f"f{i}.png" for i in range(3)]
    frames = [
        FrameRequest(
            time / 2,
            (FrameOutput("branch_a", str(paths[time]), 32, 32),),
        )
        for time in range(3)
    ]
    engine.render_frames(frames, resume=False)
    contents = {path.read_bytes() for path in paths}
    assert len(contents) == 3


def test_full_project_jobs_render_preview_frames_and_hires(tmp_path):
    project_path = tmp_path / "project.json"
    graph, outputs = build_graph()
    jobs = [
        RenderJob(
            id="preview_a",
            kind="static",
            node="branch_a",
            path="out/preview_a.png",
            width=32,
            height=32,
            time=0.0,
        ),
        RenderJob(
            id="seq_a",
            kind="sequence",
            node="branch_a",
            path="out/seq_a.png",
            width=32,
            height=32,
            start_time=0.0,
            end_time=1.0,
            frame_count=3,
        ),
        RenderJob(
            id="hires_a",
            kind="static",
            node="branch_a",
            path="out/hires_a.png",
            width=128,
            height=128,
            time=0.5,
        ),
    ]
    save_project(str(project_path), graph, outputs, jobs)
    rendered = render_project(str(project_path), resume=False)

    with Image.open(tmp_path / "out/preview_a.png") as image:
        assert image.size == (32, 32)
    for index in range(3):
        frame_path = tmp_path / sequence_frame_path("out/seq_a.png", index, 3)
        assert frame_path.exists()
    with Image.open(tmp_path / "out/hires_a.png") as image:
        assert image.size == (128, 128)


def test_example_outputs_are_valid_rgba_pictures(tmp_path):
    project_path = tmp_path / "project.json"
    graph, outputs = build_graph()
    save_project(str(project_path), graph, outputs)
    rendered = render_project(str(project_path), resume=False)
    for path in rendered.values():
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.size == (64, 64)
