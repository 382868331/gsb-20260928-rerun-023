import numpy as np
import pytest
from PIL import Image

from texturegraph import (
    EvaluationCache,
    Graph,
    Region,
    RegionTarget,
    RegionWorkspace,
)


def _pattern_graph(width=64, height=64):
    """fg pattern -> blur -> blend over bg with a circular mask."""
    graph = Graph(width=width, height=height)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node("t", "transform_coordinates", {"offset_x": 0.1})
    graph.add_node("m", "circle_mask", {"radius": 0.4, "softness": 0.02})
    graph.add_node("pm", "rectangle_mask", {"half_width": 0.6, "half_height": 0.2})
    graph.add_node("pat_fg", "solid_color", {"color": [0.9, 0.8, 0.2, 1.0]})
    graph.add_node("pat_bg", "solid_color", {"color": [0.1, 0.2, 0.5, 1.0]})
    graph.add_node("pattern", "blend_colors")
    graph.add_node("blur", "box_blur", {"radius": 3})
    graph.add_node("bg", "solid_color", {"color": [0.03, 0.03, 0.03, 1.0]})
    graph.add_node("out", "blend_colors")
    graph.add_node("lone", "solid_color", {"color": [0.5, 0.5, 0.5, 1.0]})

    graph.connect("uv", "t", "coordinates")
    graph.connect("t", "m", "coordinates")
    graph.connect("uv", "pm", "coordinates")
    graph.connect("pat_fg", "pattern", "foreground")
    graph.connect("pat_bg", "pattern", "background")
    graph.connect("pm", "pattern", "mask")
    graph.connect("pattern", "blur", "image")
    graph.connect("blur", "out", "foreground")
    graph.connect("bg", "out", "background")
    graph.connect("m", "out", "mask")
    return graph


def _full_png(graph, target, overrides=None):
    import copy

    from texturegraph import TextureEngine, RenderContext

    work = graph
    if overrides:
        work = copy.deepcopy(graph)
        for node_id, params in overrides.items():
            for name, value in params.items():
                work.set_param(node_id, name, value)
    value = TextureEngine(work, EvaluationCache()).evaluate(
        target.node_id, RenderContext(time=target.time),
        target.width, target.height, target.time,
    )
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)


def _workspace(tmp_path, graph):
    return RegionWorkspace(str(tmp_path / "regions"), graph, EvaluationCache())


LEFT = Region(0, 0, 32, 64)
RIGHT = Region(32, 0, 64, 64)
LEFT_OVERRIDES = {
    "m": {"radius": 0.25},
    "blur": {"radius": 5},
    "out": {"invert_mask": True},
}
RIGHT_OVERRIDES = {
    "m": {"softness": 0.1},
    "blur": {"radius": 1},
}


def test_candidate_matches_full_recompute_including_cross_boundary_blur(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)

    candidate = workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    assert candidate.full_digest_match is True

    full = _full_png(graph, target, LEFT_OVERRIDES)
    with Image.open(candidate.tile_path) as image:
        tile = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    assert np.array_equal(tile, full[:, :32])

    # Border pixels exercise the full blur neighbourhood across the split.
    assert np.array_equal(tile[:, -1], full[:, 31])


def test_adopting_one_of_two_adjacent_candidates_keeps_other_region(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    baseline_path = workspace.published_path(target)
    baseline_bytes = open(baseline_path, "rb").read()
    baseline_pixels = np.asarray(Image.open(baseline_path).convert("RGBA"))

    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate("right", target, RIGHT, RIGHT_OVERRIDES)
    workspace.cancel_candidate("right")
    adopted = workspace.adopt_candidate("left")

    # Old publish remains on disk untouched; candidates never overwrite it.
    assert open(baseline_path, "rb").read() == baseline_bytes
    assert adopted.published_version == 2

    new_path = workspace.published_path(target)
    assert new_path != baseline_path
    new_pixels = np.asarray(Image.open(new_path).convert("RGBA"))
    full_left = _full_png(graph, target, LEFT_OVERRIDES)
    assert np.array_equal(np.asarray(new_pixels)[:, :32], full_left[:, :32])
    # The rejected neighbouring region keeps its published content exactly.
    assert np.array_equal(
        np.asarray(new_pixels)[:, 32:], baseline_pixels[:, 32:]
    )


def test_adjacent_pending_candidates_can_be_adopted_in_either_order(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate("right", target, RIGHT, RIGHT_OVERRIDES)

    # Adjacent pending candidates are not a conflict; the second simply
    # stays pending while the first is adopted in a new version.
    first = workspace.adopt_candidate("left")
    assert first.published_version == 2
    assert "right" in workspace.candidates
    second = workspace.adopt_candidate("right")
    assert second.published_version == 3
    assert not workspace.candidates

    full = _full_png(graph, target)
    full_left = _full_png(graph, target, LEFT_OVERRIDES)
    full_right = _full_png(graph, target, RIGHT_OVERRIDES)
    pixels = np.asarray(
        Image.open(workspace.published_path(target)).convert("RGBA")
    )
    assert np.array_equal(np.asarray(pixels)[:, :32], full_left[:, :32])
    assert np.array_equal(np.asarray(pixels)[:, 32:], full_right[:, 32:])
    assert not np.array_equal(np.asarray(pixels), full)


def test_overlapping_pending_candidates_require_explicit_choice(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate(
        "over", target, Region(16, 0, 48, 64), {"blur": {"radius": 2}}
    )

    # Registration allows the overlap; only adopting forces a decision.
    assert set(workspace.candidates) == {"left", "over"}
    with pytest.raises(ValueError, match="cancelled explicitly"):
        workspace.adopt_candidate("left")
    with pytest.raises(ValueError, match="cancelled explicitly"):
        workspace.adopt_candidate("over")

    # Picking either side explicitly is fine; order decides nothing.
    workspace.cancel_candidate("left")
    adopted = workspace.adopt_candidate("over")
    assert adopted.published_version == 2

    workspace.create_candidate("left2", target, LEFT, LEFT_OVERRIDES)
    with pytest.raises(ValueError, match="overlaps adopted"):
        workspace.adopt_candidate("left2")
    replaced = workspace.adopt_candidate("left2", replace=True)
    assert replaced.published_version == 3


def test_overlapping_candidates_are_listed_with_params_versions_and_diffs(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate("left2", target, Region(0, 0, 40, 64), {
        "m": {"radius": 0.35},
    })

    comparison = workspace.compare_candidates(target)
    entries = {entry["id"]: entry for entry in comparison["candidates"]}
    assert set(entries) == {"left", "left2"}
    assert entries["left"]["overrides"] == LEFT_OVERRIDES
    assert entries["left2"]["overrides"] == {"m": {"radius": 0.35}}
    assert entries["left"]["input_version"]
    assert entries["left"]["input_version"] != entries["left2"]["input_version"]
    assert "out" in entries["left"]["node_signatures"]
    assert entries["left"]["diff_vs_published"]["pixels"] == 32 * 64
    assert entries["left2"]["diff_vs_published"]["pixels"] == 40 * 64
    overlap = entries["left"]["overlap_diffs"]["left2"]
    assert overlap["pixels"] == 32 * 64
    assert entries["left2"]["overlap_diffs"]["left"] == overlap


def test_non_square_canvas_overlapping_candidates_keep_full_blur_halo(tmp_path):
    width, height = 96, 48
    graph = _pattern_graph(width=width, height=height)
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", width, height)
    workspace.publish(target)
    region_a = Region(0, 0, 60, height)
    region_b = Region(40, 0, width, height)
    workspace.create_candidate("a", target, region_a, LEFT_OVERRIDES)
    workspace.create_candidate("b", target, region_b, RIGHT_OVERRIDES)

    full_a = _full_png(graph, target, LEFT_OVERRIDES)
    full_b = _full_png(graph, target, RIGHT_OVERRIDES)
    tile_a = np.asarray(
        Image.open(workspace.candidates["a"].tile_path).convert("RGBA")
    )
    tile_b = np.asarray(
        Image.open(workspace.candidates["b"].tile_path).convert("RGBA")
    )
    # Each tile matches the full recompute, including the seam columns
    # where the box blur reads pixels on both sides of the boundary.
    assert np.array_equal(tile_a, full_a[:, :60])
    assert np.array_equal(tile_b, full_b[:, 40:])

    with pytest.raises(ValueError, match="cancelled explicitly"):
        workspace.adopt_candidate("a")
    workspace.cancel_candidate("b")
    workspace.adopt_candidate("a")
    published = np.asarray(
        Image.open(workspace.published_path(target)).convert("RGBA")
    )
    baseline = np.asarray(
        Image.open(
            workspace._published_path(target, 1)
        ).convert("RGBA")
    )
    assert np.array_equal(np.asarray(published)[:, :60], full_a[:, :60])
    # The unchosen side keeps the original published pixels.
    assert np.array_equal(np.asarray(published)[:, 60:], baseline[:, 60:])


def test_pending_overlap_survives_save_resume_and_only_chosen_is_adopted(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate(
        "over", target, Region(16, 0, 48, 64), {"blur": {"radius": 2}}
    )
    workspace.save_state()

    resumed = RegionWorkspace.open(root, graph, cache)
    assert set(resumed.candidates) == {"left", "over"}
    comparison = resumed.compare_candidates(target)
    assert {
        entry["id"] for entry in comparison["candidates"]
    } == {"left", "over"}
    with pytest.raises(ValueError, match="cancelled explicitly"):
        resumed.adopt_candidate("over")
    resumed.cancel_candidate("over")
    adopted = resumed.adopt_candidate("left")
    assert adopted.published_version == 2


def test_revalidate_candidate_after_input_change_keeps_other_pending(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate(
        "over", target, Region(16, 0, 48, 64), {"blur": {"radius": 2}}
    )
    workspace.save_state()
    old_over_digest = workspace.candidates["over"].content_digest
    old_over_signature = workspace.candidates["over"].signature

    # Shared input change: pat_fg colour feeds the whole output.
    graph.set_param("pat_fg", "color", [0.2, 0.9, 0.3, 1.0])
    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.revalidate_candidate("left")
    assert report["inputs_changed"] is True
    assert report["recomputed"] is True

    # The overlapping sibling is still pending with its original plan.
    assert "over" in resumed.candidates
    over = resumed.candidates["over"]
    assert over.content_digest == old_over_digest
    assert over.signature == old_over_signature
    left = resumed.candidates["left"]
    assert left.tile_intact() is True
    expected = _full_png(graph, target, LEFT_OVERRIDES)
    tile = np.asarray(Image.open(left.tile_path).convert("RGBA"))
    assert np.array_equal(np.asarray(tile), expected[:, :32])

    # Adopting still demands an explicit choice for the overlap.
    with pytest.raises(ValueError, match="cancelled explicitly"):
        resumed.adopt_candidate("left")


def test_revalidate_candidate_after_tile_corruption_keeps_other_pending(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate(
        "over", target, Region(16, 0, 48, 64), {"blur": {"radius": 2}}
    )
    workspace.save_state()
    with open(workspace.candidates["left"].tile_path, "r+b") as handle:
        handle.seek(120)
        handle.write(b"\x00" * 20)

    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.revalidate_candidate("left")
    assert report["inputs_changed"] is False
    assert report["tile_damaged"] is True
    assert report["recomputed"] is True
    assert resumed.candidates["left"].tile_intact() is True
    assert resumed.candidates["over"].tile_intact() is True


def test_recover_target_revalidates_pending_candidates_only(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.save_state()

    graph.set_param("pat_fg", "color", [0.2, 0.9, 0.3, 1.0])
    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.recover_target(target)
    assert [item["id"] for item in report["revalidated_candidates"]] == ["left"]
    assert report["revalidated_candidates"][0]["inputs_changed"] is True
    assert "left" in resumed.candidates
    expected = _full_png(graph, target, LEFT_OVERRIDES)
    tile = np.asarray(
        Image.open(resumed.candidates["left"].tile_path).convert("RGBA")
    )
    assert np.array_equal(np.asarray(tile), expected[:, :32])


def test_republish_without_changes_is_a_noop(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    assert workspace.publish(target) is None


def test_save_state_and_resume_recomputes_only_affected_range(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.adopt_candidate("left")
    before_path = workspace.published_path(target)
    before = np.asarray(Image.open(before_path).convert("RGBA"))
    workspace.save_state()

    # Same-name, same-size input changes: pattern foreground colour shifts.
    graph.set_param("pat_fg", "color", [0.2, 0.9, 0.3, 1.0])
    resumed = RegionWorkspace.open(root, graph, cache)
    assert resumed.versions[target.key()] == 2
    assert len([t for t in resumed.adopted if t.valid]) == 1
    report = resumed.recover_target(target)
    assert report["changed_nodes"]
    assert report["affected_tiles"] == ["left"]
    assert report["recomputed"] is True

    after_path = resumed.published_path(target)
    after = np.asarray(Image.open(after_path).convert("RGBA"))
    expected = _full_png(graph, target, LEFT_OVERRIDES)
    assert np.array_equal(np.asarray(after)[:, :32], expected[:, :32])
    # Other region keeps the previously published bytes and source.
    assert np.array_equal(np.asarray(after)[:, 32:], before[:, 32:])


def test_resume_rejudges_scope_after_same_sized_input_change(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.adopt_candidate("left")
    workspace.save_state()

    # A node unrelated to the output changes: no adopted range is affected.
    graph.set_param("lone", "color", [0.1, 0.1, 0.1, 1.0])
    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.recover_target(target)
    assert report["changed_nodes"] == []
    assert report["affected_tiles"] == []
    assert report["recomputed"] is False
    assert resumed.versions[target.key()] == 2


def test_resume_after_corrupted_publish_rebuilds_and_preserves_regions(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.adopt_candidate("left")
    before = np.asarray(
        Image.open(workspace.published_path(target)).convert("RGBA")
    )
    left_bytes = np.asarray(before)[:, :32].copy()
    workspace.save_state()

    corrupt_path = workspace.published_path(target)
    with open(corrupt_path, "r+b") as handle:
        handle.seek(100)
        handle.write(b"\xff" * 40)

    resumed = RegionWorkspace.open(root, graph, cache, recover=False)
    report = resumed.recover_target(target)
    assert report["damaged"] is True
    assert report["recomputed"] is False
    after = np.asarray(
        Image.open(resumed.published_path(target)).convert("RGBA")
    )
    # The adopted region pixels are preserved from the stored partition.
    assert np.array_equal(np.asarray(after)[:, :32], left_bytes)
    assert resumed.versions[target.key()] == 3


def test_pending_candidate_is_restored_and_damage_flagged(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph()
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    candidate = workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.save_state()

    resumed = RegionWorkspace.open(root, graph, cache)
    assert "left" in resumed.candidates
    assert resumed.candidates["left"].tile_intact() is True

    with open(candidate.tile_path, "r+b") as handle:
        handle.seek(120)
        handle.write(b"\x00" * 20)
    damaged = RegionWorkspace.open(root, graph, cache)
    assert damaged.candidates["left"].tile_intact() is False
    assert damaged.candidates["left"].full_digest_match is False
