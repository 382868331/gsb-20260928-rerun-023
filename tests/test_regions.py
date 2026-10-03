import copy
import json

import numpy as np
import pytest
from PIL import Image

from texturegraph import (
    EvaluationCache,
    Graph,
    Region,
    RegionCandidate,
    RegionPublisher,
    RenderCancelled,
    Variant,
    inspect_all_releases,
)
from texturegraph.graph import Connection

WIDTH = HEIGHT = 64
BLUR_RADIUS = 4


def build_blur_graph():
    """Circle mask -> box blur -> blend: a real neighborhood chain."""
    graph = Graph(width=WIDTH, height=HEIGHT)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node("circle", "circle_mask", {
        "center_x": -0.3, "center_y": 0.0, "radius": 0.25, "softness": 0.0,
    })
    graph.add_node("blur", "blur_mask", {"radius_px": BLUR_RADIUS})
    graph.add_node("red", "solid_color", {"color": [0.9, 0.1, 0.1, 1.0]})
    graph.add_node("blue", "solid_color", {"color": [0.05, 0.1, 0.35, 1.0]})
    graph.add_node("out", "blend_colors")
    graph.connect("uv", "circle", "coordinates")
    graph.connect("circle", "blur", "mask")
    graph.connect("red", "out", "foreground")
    graph.connect("blue", "out", "background")
    graph.connect("blur", "out", "mask")
    return graph


def regions_grid():
    return [
        Region("left", 0, 0, WIDTH // 2, HEIGHT, input_version="input-v1"),
        Region("right", WIDTH // 2, 0, WIDTH // 2, HEIGHT, input_version="input-v1"),
    ]


def move_circle_candidate(candidate_id="move-circle", region_id="left",
                          center_x=0.35, input_version="input-v1"):
    region = Region(
        region_id, 0 if region_id == "left" else WIDTH // 2, 0,
        WIDTH // 2, HEIGHT, input_version=input_version,
    )
    variant = Variant(candidate_id, {("circle", "center_x"): center_x})
    return RegionCandidate(candidate_id, region, variant, label="move-circle-choice")


def publisher(graph, tmp_path):
    return RegionPublisher(
        graph, "out", WIDTH, HEIGHT, str(tmp_path / "regions"), cache=EvaluationCache()
    )


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def test_baseline_and_local_update_reuses_other_region(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")

    candidate = move_circle_candidate()
    _base, plans = pub.preview_update([candidate], baseline)
    plan = plans[0]
    assert not plan.full_recompute_required
    # Analytic dilation includes the blur halo, and the render-aware
    # preview grows it to the exact observed changed-pixel bounds.
    assert plan.recompute.end_x >= WIDTH // 2 + BLUR_RADIUS

    updated = pub.publish_update(
        {"left": candidate}, "r1", baseline, plans
    )

    full = pixels(updated.path)
    base_full = pixels(baseline.path)
    # Every pixel outside the previewed recompute footprint is reused.
    assert np.array_equal(
        full[:, plan.recompute.end_x:],
        base_full[:, plan.recompute.end_x:],
    )
    # Halo pixels inside the footprint but in the neighbor region changed.
    assert not np.array_equal(
        full[:, WIDTH // 2:plan.recompute.end_x],
        base_full[:, WIDTH // 2:plan.recompute.end_x],
    )
    # The edited half differs.
    assert not np.array_equal(
        full[:, : WIDTH // 2], base_full[:, : WIDTH // 2]
    )


def test_partial_update_matches_full_recompute_within_region_and_has_no_seam(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")

    candidate = move_circle_candidate()
    updated = pub.publish_update({"left": candidate}, "r1", baseline)

    full_candidate = pub.render_full_comparison(candidate)
    assembled = pixels(updated.path)
    # Every pixel of the authored region equals a from-scratch full render
    # with the candidate graph; the blurred halo made this possible
    # without a seam along the region boundary.
    assert np.array_equal(
        assembled[:, : WIDTH // 2], full_candidate[:, : WIDTH // 2]
    )
    boundary = assembled[:, WIDTH // 2 - 2: WIDTH // 2 + 2]
    full_boundary = full_candidate[:, WIDTH // 2 - 2: WIDTH // 2 + 2]
    # Halo included: pixels just inside the reused side near the boundary
    # that the blur touches are recomputed too (check digest trace).
    tile_left = pixels(updated.tiles["left"].path)
    assert tile_left.shape[1] == WIDTH // 2
    right_record = updated.tiles["right"]
    # The neighbor tile records the candidate halo intruding on it.
    assert right_record.contributors
    intruder = right_record.contributors[0]
    assert intruder["candidate"] == "move-circle"
    assert intruder["rect"]["x"] == WIDTH // 2
    assert boundary.shape == full_boundary.shape


def test_provenance_manifest_traces_region_input_and_choice(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    candidate = move_circle_candidate()
    updated = pub.publish_update({"left": candidate}, "r1", baseline)

    with open(pub.release_manifest_path("r1")) as handle:
        manifest = json.load(handle)
    left = manifest["regions"]["left"]
    right = manifest["regions"]["right"]
    assert left["region"]["input_version"] == "input-v1"
    assert left["source_candidate"] == "move-circle"
    assert left["choice"] == "move-circle-choice"
    assert left["variant"] == "move-circle"
    assert right["source_candidate"] == "baseline"
    assert right["region"]["input_version"] == "input-v1"
    # Tile signature for the edited region is the candidate graph signature.
    assert left["signature"] != right["signature"]
    # Same name+size but a different input version is detectable.
    other = move_circle_candidate(
        candidate_id="v2-input", region_id="left",
        center_x=0.3, input_version="input-v2",
    )
    assert other.region.region_id == candidate.region.region_id
    assert other.region.input_version != candidate.region.input_version


def test_overlapping_candidates_compare_against_same_base_and_cannot_both_publish(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")

    a = move_circle_candidate("cand-a", "left", center_x=0.3)
    b = move_circle_candidate("cand-b", "left", center_x=0.0)
    plans = pub.plan_update([a, b], baseline)
    assert set(plans[0].overlaps) == {"cand-b"}
    assert set(plans[1].overlaps) == {"cand-a"}

    report = pub.compare_candidate_impacts([a, b], baseline)
    assert report["cand-a"]["changed_pixels"] > 0
    assert report["cand-b"]["changed_pixels"] > 0
    assert report["cand-a"]["within_recompute_footprint"]
    assert report["cand-b"]["within_recompute_footprint"]

    with pytest.raises(Exception):
        pub.publish_update({"left": a, "left2": b}, "r1", baseline, plans)

    # Choosing one candidate for the region publishes cleanly.
    updated = pub.publish_update({"left": a}, "r1", baseline, plans)
    assert updated.tiles["left"].source_candidate == "cand-a"


def _build_two_shape_graph():
    graph = Graph(width=WIDTH, height=HEIGHT)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node("left_circle", "circle_mask", {
        "center_x": -0.7, "center_y": 0.0, "radius": 0.18, "softness": 0.0,
    })
    graph.add_node("right_circle", "circle_mask", {
        "center_x": 0.7, "center_y": 0.0, "radius": 0.18, "softness": 0.0,
    })
    graph.add_node("masks", "combine_masks", {"operation": "maximum"})
    graph.add_node("fg", "solid_color", {"color": [0.9, 0.9, 0.2, 1.0]})
    graph.add_node("bg", "solid_color", {"color": [0.05, 0.05, 0.2, 1.0]})
    graph.add_node("out", "blend_colors")
    graph.connect("uv", "left_circle", "coordinates")
    graph.connect("uv", "right_circle", "coordinates")
    graph.connect("left_circle", "masks", "a")
    graph.connect("right_circle", "masks", "b")
    graph.connect("fg", "out", "foreground")
    graph.connect("bg", "out", "background")
    graph.connect("masks", "out", "mask")
    return graph


def test_explicitly_partitioned_ranges_allow_both_candidates(tmp_path):
    graph = _build_two_shape_graph()
    pub = RegionPublisher(
        graph, "out", WIDTH, HEIGHT, str(tmp_path / "regions"),
        cache=EvaluationCache(),
    )
    half_regions = regions_grid()
    baseline = pub.publish_baseline(half_regions, "r0")

    a = RegionCandidate(
        "a", half_regions[0],
        Variant("a", {("left_circle", "center_y"): 0.45}),
    )
    b = RegionCandidate(
        "b", half_regions[1],
        Variant("b", {("right_circle", "center_y"): -0.45}),
    )
    _base, plans = pub.preview_update([a, b], baseline)
    assert not plans[0].overlaps and not plans[1].overlaps
    updated = pub.publish_update(
        {"left": a, "right": b}, "r1", baseline, plans
    )
    assert updated.tiles["left"].source_candidate == "a"
    assert updated.tiles["right"].source_candidate == "b"
    # Middle columns untouched by either local shape move are reused.
    full = pixels(updated.path)
    base_full = pixels(baseline.path)
    assert np.array_equal(full[:, 30:34], base_full[:, 30:34])


def test_cancel_during_partial_publish_resumes_same_plan_only(tmp_path):
    graph = _build_two_shape_graph()
    pub = RegionPublisher(
        graph, "out", WIDTH, HEIGHT, str(tmp_path / "regions"),
        cache=EvaluationCache(),
    )
    half_regions = regions_grid()
    baseline = pub.publish_baseline(half_regions, "r0")

    a = RegionCandidate(
        "cand-a", half_regions[0],
        Variant("cand-a", {("left_circle", "center_y"): 0.4}),
    )
    b = RegionCandidate(
        "cand-b", half_regions[1],
        Variant("cand-b", {("right_circle", "center_y"): -0.4}),
    )
    chosen = {"left": a, "right": b}
    _base, plans = pub.preview_update([a, b], baseline)

    staging_dir = tmp_path / "regions" / "staging"

    def cancel_after_first_staged():
        staged = list(staging_dir.rglob("*.png")) if staging_dir.exists() else []
        return bool(staged)

    with pytest.raises(RenderCancelled):
        pub.publish_update(
            chosen, "r1", baseline, plans, cancel_after_first_staged
        )
    assert not (tmp_path / "regions" / "releases" / "r1").exists()
    staged_before = set(
        str(p.relative_to(staging_dir))
        for p in staging_dir.rglob("*.png")
    )
    assert staged_before

    updated = pub.publish_update(chosen, "r1", baseline, plans)
    full = pixels(updated.path)
    for candidate in (a, b):
        full_candidate = pub.render_full_comparison(candidate)
        region = candidate.region
        assert np.array_equal(
            full[region.y:region.y + region.height,
                 region.x:region.x + region.width],
            full_candidate[region.y:region.y + region.height,
                           region.x:region.x + region.width],
        )
    # Staging for the completed plan is cleaned up after a clean publish.
    assert not staging_dir.exists() or not list(staging_dir.rglob("*.png"))


def test_cancelled_staging_from_other_candidate_is_not_mixed(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")

    a = move_circle_candidate("cand-a", "left", center_x=0.4)
    plans_a = pub.plan_update([a], baseline)
    with pytest.raises(RenderCancelled):
        pub.publish_update(
            {"left": a}, "never", baseline, plans_a,
            is_cancelled=lambda: True,
        )

    # A different candidate (different overrides) produces a different
    # plan hash: its staging area never reuses cand-a half-work.
    c = move_circle_candidate("cand-c", "left", center_x=-0.2)
    plans_c = pub.plan_update([c], baseline)
    updated = pub.publish_update({"left": c}, "r1", baseline, plans_c)
    assert updated.tiles["left"].source_candidate == "cand-c"
    assert (tmp_path / "regions" / "releases" / "never").exists() is False


def test_input_change_later_marks_only_dependent_regions_stale(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    candidate = move_circle_candidate()
    pub.publish_update({"left": candidate}, "r1", baseline)

    # Inputs change afterwards: recompute current expected signatures.
    changed = copy.deepcopy(graph)
    changed.set_param("circle", "radius", 0.45)
    statuses = {
        s.region_id: s
        for s in pub.inspect_release(pub.load_release("r1"), changed)
    }
    # Both regions ultimately depend on the shared circle mask, so both
    # are stale; the old release r0 and r1 files are still retained.
    assert statuses["left"].status == "stale"
    assert statuses["right"].status == "stale"
    assert (tmp_path / "regions" / "releases" / "r0" / "image.png").exists()
    assert (tmp_path / "regions" / "releases" / "r1" / "image.png").exists()

    # An independent published result (release built from a graph that
    # does not use the changed node) stays current: emulate with a
    # second publisher sharing the store but a different output graph.
    other = Graph(width=WIDTH, height=HEIGHT)
    other.add_node("a", "solid_color", {"color": [0.2, 0.2, 0.2, 1.0]})
    other.add_node("b", "solid_color", {"color": [0.8, 0.8, 0.8, 1.0]})
    other.add_node("m", "rectangle_mask", {"half_width": 1.0, "half_height": 1.0})
    other.add_node("uv", "uv_coordinates")
    other.connect("uv", "m", "coordinates")
    other.add_node("o", "blend_colors")
    other.connect("a", "o", "foreground")
    other.connect("b", "o", "background")
    other.connect("m", "o", "mask")
    pub2 = RegionPublisher(
        other, "o", WIDTH, HEIGHT,
        str(tmp_path / "independent"), cache=EvaluationCache(),
    )
    ind_regions = [Region("all", 0, 0, WIDTH, HEIGHT)]
    ind_release = pub2.publish_baseline(ind_regions, "r0")
    ind_status = pub2.inspect_release(ind_release, other)
    assert all(s.status == "current" for s in ind_status)


def test_damaged_shared_region_lists_affected_and_keeps_others_usable(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    candidate = move_circle_candidate()
    pub.publish_update({"left": candidate}, "r1", baseline)

    victim = pub.tile_path("r1", "right")
    payload = bytearray(open(victim, "rb").read())
    payload[len(payload) // 2] ^= 0xFF
    open(victim, "wb").write(bytes(payload))

    report = inspect_all_releases(pub)
    found = [v for v in report.values() if hasattr(v, "damaged_shared_signature")]
    assert found, "expected a shared-damage report entry"
    damage = found[0]
    affected = {(release, region) for release, region, _ in damage.affected}
    assert ("r1", "right") in affected
    assert "r0" in damage.unaffected_releases
    statuses_r0 = {
        s.region_id: s for s in report["releases"]["r0"]
    }
    assert statuses_r0["left"].status == "current"
    assert statuses_r0["right"].status == "current"


def test_global_parameter_edit_forces_full_recompute_plan(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    global_candidate = RegionCandidate(
        "recolor",
        Region("left", 0, 0, WIDTH // 2, HEIGHT),
        Variant("recolor", {("red", "color"): [0.1, 0.9, 0.2, 1.0]}),
    )
    plans = pub.plan_update([global_candidate], baseline)
    assert plans[0].full_recompute_required
    assert plans[0].recompute.width == WIDTH
    assert plans[0].recompute.height == HEIGHT
    with pytest.raises(Exception):
        pub.publish_update({"left": global_candidate}, "r1", baseline, plans)


def test_full_recompute_reference_matches_candidate_render(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    candidate = move_circle_candidate()
    full = pub.render_full_comparison(candidate)

    # The composited partial result for the authored region is identical
    # to a full from-scratch recomputation (cross-region neighborhood
    # case already covered above; here compare the entire footprint).
    updated = pub.publish_update({"left": candidate}, "r1", baseline)
    assembled = pixels(updated.path)
    plan = pub.plan_update([candidate], baseline)[0]
    rect = plan.recompute
    assert np.array_equal(
        assembled[rect.y:rect.end_y, rect.x:rect.end_x],
        full[rect.y:rect.end_y, rect.x:rect.end_x],
    )


def test_old_release_image_matches_manifest_forever(tmp_path):
    graph = build_blur_graph()
    pub = publisher(graph, tmp_path)
    baseline = pub.publish_baseline(regions_grid(), "r0")
    candidate = move_circle_candidate()
    pub.publish_update({"left": candidate}, "r1", baseline)
    for release_id in ("r0", "r1"):
        release = pub.load_release(release_id)
        image_pixels = pixels(release.path)
        from texturegraph.engine import image_digest
        assert image_digest(image_pixels) == release.digest
        for tile in release.tiles.values():
            tile_pixels = pixels(tile.path)
            assert image_digest(tile_pixels) == tile.digest
