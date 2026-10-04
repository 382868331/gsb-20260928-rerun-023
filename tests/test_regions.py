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


def test_overlapping_candidates_must_be_chosen_or_cancelled(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    # Overlapping candidates may be registered side by side now.
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate("overlap", target, Region(16, 0, 48, 64), RIGHT_OVERRIDES)

    with pytest.raises(ValueError, match="cancelled explicitly"):
        workspace.adopt_candidate("left")

    # Cancelling one is an explicit choice.
    workspace.cancel_candidate("overlap")
    workspace.adopt_candidate("left")

    workspace.create_candidate("right2", target, RIGHT, RIGHT_OVERRIDES)
    # Adopted regions are adjacent (not overlapping) here, so adopting is fine.
    workspace.adopt_candidate("right2")

    workspace.create_candidate(
        "over", target, Region(16, 0, 48, 64), {"blur": {"radius": 2}}
    )
    with pytest.raises(ValueError, match="overlaps adopted"):
        workspace.adopt_candidate("over")
    replaced = workspace.adopt_candidate("over", replace=True)
    assert replaced.published_version == 4


def test_adjacent_pending_candidates_do_not_block_adoption(tmp_path):
    graph = _pattern_graph()
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 64, 64)
    workspace.publish(target)
    workspace.create_candidate("left", target, LEFT, LEFT_OVERRIDES)
    workspace.create_candidate("right", target, RIGHT, RIGHT_OVERRIDES)
    # Adjacent (non-overlapping) pending candidates are not a conflict.
    adopted = workspace.adopt_candidate("left")
    assert adopted.published_version == 2
    assert "right" in workspace.candidates


def test_overlapping_candidates_listed_side_by_side(tmp_path):
    graph = _pattern_graph(width=96, height=48)
    workspace = _workspace(tmp_path, graph)
    target = RegionTarget("out", 96, 48)
    workspace.publish(target)
    first = Region(8, 4, 60, 44)
    second = Region(40, 4, 88, 44)
    workspace.create_candidate("a", target, first, LEFT_OVERRIDES)
    workspace.create_candidate("b", target, second, RIGHT_OVERRIDES)

    entries = {entry["id"]: entry for entry in workspace.list_candidates(target)}
    assert set(entries) == {"a", "b"}
    assert entries["a"]["overrides"] == LEFT_OVERRIDES
    assert entries["b"]["overrides"] == RIGHT_OVERRIDES
    assert entries["a"]["signature"] != entries["b"]["signature"]
    for entry in entries.values():
        assert entry["intact"] is True
        diff = entry["pixel_differences"]
        assert diff["changed_pixels"] > 0
        assert diff["max_delta"] > 0


def test_pending_overlap_resume_adopts_only_selected_region(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph(width=96, height=48)
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 96, 48)
    workspace.publish(target)
    baseline_pixels = np.asarray(
        Image.open(workspace.published_path(target)).convert("RGBA")
    )
    selected = Region(8, 4, 60, 44)
    other = Region(40, 4, 88, 44)
    workspace.create_candidate("sel", target, selected, LEFT_OVERRIDES)
    workspace.create_candidate("alt", target, other, RIGHT_OVERRIDES)
    workspace.save_state()

    resumed = RegionWorkspace.open(root, graph, cache)
    assert set(resumed.candidates) == {"sel", "alt"}
    # Explicit choice: drop the alternative, then adopt the selected one.
    resumed.cancel_candidate("alt")
    resumed.adopt_candidate("sel")

    new_pixels = np.asarray(
        Image.open(resumed.published_path(target)).convert("RGBA")
    )
    full_selected = _full_png(graph, target, LEFT_OVERRIDES)
    assert np.array_equal(
        new_pixels[selected.y0 : selected.y1, selected.x0 : selected.x1],
        full_selected[selected.y0 : selected.y1, selected.x0 : selected.x1],
    )
    # Blur at the region border still averaged the full neighbourhood.
    assert np.array_equal(
        new_pixels[selected.y0 : selected.y1, selected.x1 - 1],
        full_selected[selected.y0 : selected.y1, selected.x1 - 1],
    )
    # Everything outside the selected rectangle keeps the published pixels,
    # including the area covered only by the rejected candidate.
    mask = np.zeros((48, 96), dtype=bool)
    mask[selected.y0 : selected.y1, selected.x0 : selected.x1] = True
    assert np.array_equal(new_pixels[~mask], baseline_pixels[~mask])


def test_recover_revalidates_only_stale_pending_candidates(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph(width=96, height=48)
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 96, 48)
    workspace.publish(target)
    workspace.create_candidate("a", target, Region(8, 4, 60, 44), LEFT_OVERRIDES)
    workspace.create_candidate("b", target, Region(40, 4, 88, 44), RIGHT_OVERRIDES)
    workspace.save_state()

    # Input change affects both candidates; both are recomputed in place.
    graph.set_param("pat_fg", "color", [0.2, 0.9, 0.3, 1.0])
    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.recover_target(target)
    assert sorted(report["revalidated_candidates"]) == ["a", "b"]
    assert set(resumed.candidates) == {"a", "b"}
    for candidate in resumed.candidates.values():
        assert candidate.tile_intact() is True
        assert candidate.full_digest_match is True
    full_a = _full_png(graph, target, LEFT_OVERRIDES)
    tile_a = np.asarray(
        Image.open(resumed.candidates["a"].tile_path).convert("RGBA")
    )
    assert np.array_equal(tile_a, full_a[4:44, 8:60])

    # A second recovery with no changes revalidates nothing.
    resumed.save_state()
    again = RegionWorkspace.open(root, graph, cache)
    report = again.recover_target(target)
    assert report["revalidated_candidates"] == []


def test_recover_revalidates_damaged_candidate_and_keeps_others(tmp_path):
    root = str(tmp_path / "regions")
    graph = _pattern_graph(width=96, height=48)
    cache = EvaluationCache()
    workspace = RegionWorkspace(root, graph, cache)
    target = RegionTarget("out", 96, 48)
    workspace.publish(target)
    damaged = workspace.create_candidate(
        "a", target, Region(8, 4, 60, 44), LEFT_OVERRIDES
    )
    workspace.create_candidate("b", target, Region(40, 4, 88, 44), RIGHT_OVERRIDES)
    workspace.save_state()

    with open(damaged.tile_path, "r+b") as handle:
        handle.seek(120)
        handle.write(b"\x00" * 20)

    resumed = RegionWorkspace.open(root, graph, cache)
    report = resumed.recover_target(target)
    assert report["revalidated_candidates"] == ["a"]
    assert set(resumed.candidates) == {"a", "b"}
    assert resumed.candidates["a"].tile_intact() is True
    assert resumed.candidates["b"].tile_intact() is True


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
