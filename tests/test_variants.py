from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from texturegraph import (
    EvaluationCache,
    GraphValidationError,
    RenderCancelled,
    RenderJob,
    TextureEngine,
    Variant,
    VariantOutput,
    affected_variants,
    compare_variants,
    format_comparison,
    load_project,
    render_project,
    render_variant_batch,
    save_project,
)
from texturegraph.variants import apply_variant
from examples.shared_mask import build_graph


def make_outputs(tmp_path):
    return [
        VariantOutput("branch_a", str(tmp_path / "{variant}" / "a.png"), 32, 32),
        VariantOutput("branch_b", str(tmp_path / "{variant}" / "b.png"), 32, 32),
    ]


def make_variants():
    return [
        Variant("base", label="原始方案"),
        Variant(
            "warm",
            overrides={"red": {"color": [0.95, 0.55, 0.10, 1.0]}},
            label="暖色方案",
        ),
        Variant(
            "small",
            overrides={"shared_mask": {"radius": 0.25}},
            label="小圆方案",
        ),
    ]


def read_pixels(path):
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()


def test_batch_shares_intermediate_results_and_keeps_schemes_distinct(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    outputs = make_outputs(tmp_path)
    cache = EvaluationCache()

    results = render_variant_batch(graph, variants, outputs, cache=cache)

    assert set(results) == {"base", "warm", "small"}
    # Every scheme published its own files; nothing replaced anything else.
    for variant in variants:
        for output in outputs:
            path = output.resolve_path(variant.id)
            assert path in results[variant.id].outputs.values()
    warm_a = read_pixels(str(tmp_path / "warm" / "a.png"))
    base_a = read_pixels(str(tmp_path / "base" / "a.png"))
    assert not np.array_equal(warm_a, base_a)
    # "warm" only changed branch_a's color: branch_b is pixel-identical.
    assert np.array_equal(
        read_pixels(str(tmp_path / "warm" / "b.png")),
        read_pixels(str(tmp_path / "base" / "b.png")),
    )
    # "small" changed the shared mask: both of its branches differ.
    assert not np.array_equal(
        read_pixels(str(tmp_path / "small" / "b.png")),
        read_pixels(str(tmp_path / "base" / "b.png")),
    )
    report = format_comparison(compare_variants(graph, variants, outputs))
    assert "all schemes are up to date" in report


def test_shared_mask_evaluated_once_across_variants(tmp_path):
    graph, _ = build_graph()
    variants = [Variant("one"), Variant("two")]
    outputs = make_outputs(tmp_path)
    cache = EvaluationCache()
    render_variant_batch(graph, variants, outputs, cache=cache)
    # Count signature entries: both variants share identical node signatures,
    # so the cache holds exactly one entry per graph node, not two.
    engine = TextureEngine(graph, cache)
    context_count = len(cache._values)
    order = graph.topological_order("branch_a")
    all_nodes = set(order) | set(graph.topological_order("branch_b"))
    assert context_count == len(all_nodes)


def test_identical_effective_params_still_do_not_replace_each_other(tmp_path):
    graph, _ = build_graph()
    # "copy" overrides a param with the very same value: effective params,
    # node names and sizes are identical to "base".
    base_color = graph.nodes["red"].params["color"]
    variants = [Variant("base"), Variant("copy", {"red": {"color": list(base_color)}})]
    outputs = make_outputs(tmp_path)

    results = render_variant_batch(graph, variants, outputs)

    base_file = tmp_path / "base" / "a.png"
    copy_file = tmp_path / "copy" / "a.png"
    assert base_file.exists() and copy_file.exists()
    with Image.open(base_file) as image:
        image.load()
        base_signature = image.info["TextureGraph-Signature"]
    with Image.open(copy_file) as image:
        image.load()
        copy_signature = image.info["TextureGraph-Signature"]
    # Same pixels, but each PNG is attributable to its own scheme.
    assert np.array_equal(read_pixels(base_file), read_pixels(copy_file))
    assert base_signature != copy_signature
    comparisons = compare_variants(graph, variants, outputs)
    assert all(item.status == "current" for item in comparisons)
    assert {item.variant_id for item in comparisons} == {"base", "copy"}


def test_same_path_across_variants_is_rejected(tmp_path):
    graph, _ = build_graph()
    variants = [Variant("a"), Variant("b")]
    outputs = [VariantOutput("branch_a", str(tmp_path / "same.png"), 32, 32)]
    with pytest.raises(GraphValidationError, match="distinct output path"):
        render_variant_batch(graph, variants, outputs)


def test_cancel_then_resume_validates_each_scheme_independently(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    outputs = make_outputs(tmp_path)

    state = {"cancel": False}

    def is_cancelled():
        # Cancel once the first scheme has fully published.
        return state["cancel"]

    # First run: cancel after the first variant completes.
    import texturegraph.variants as variants_module

    original_render_frames = TextureEngine.render_frames
    completed = {"frames": 0}

    def counting_render_frames(self, frames, **kwargs):
        result = original_render_frames(self, frames, **kwargs)
        completed["frames"] += len(result)
        if self.variant_id == "base":
            state["cancel"] = True
        return result

    variants_module.TextureEngine.render_frames = counting_render_frames
    try:
        with pytest.raises(RenderCancelled):
            render_variant_batch(graph, variants, outputs, is_cancelled=is_cancelled)
    finally:
        variants_module.TextureEngine.render_frames = original_render_frames

    # "base" fully published; the other schemes may be incomplete.
    comparisons = compare_variants(graph, variants, outputs)
    by_variant = {}
    for item in comparisons:
        by_variant.setdefault(item.variant_id, set()).add(item.status)
    assert by_variant["base"] == {"current"}
    assert by_variant["warm"] != {"current"}

    # Resume: each scheme is judged on its own published outputs.
    results = render_variant_batch(graph, variants, outputs)
    base_frames = results["base"].frames
    assert all(frame.skipped for frame in base_frames)
    assert any(
        not frame.skipped for frame in results["warm"].frames + results["small"].frames
    )
    assert all(
        item.status == "current"
        for item in compare_variants(graph, variants, outputs)
    )


def test_shared_input_change_lists_affected_schemes_and_keeps_others(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    outputs = make_outputs(tmp_path)
    render_variant_batch(graph, variants, outputs)

    # Change a shared input (the mask transform): every scheme depends on it.
    graph.set_param("circle_transform", "scale_x", 2.0)
    comparisons = compare_variants(graph, variants, outputs)
    affected = affected_variants(comparisons)
    assert set(affected) == {"base", "warm", "small"}
    assert all(item.status == "stale" for item in comparisons)
    # Old versions stay on disk, still readable.
    for item in comparisons:
        with Image.open(item.path) as image:
            image.load()

    # Recompute only one selected scheme; the rest keep their old versions.
    render_variant_batch(graph, variants, outputs, only=["warm"])
    comparisons = compare_variants(graph, variants, outputs)
    status = {(item.variant_id, item.node_id): item.status for item in comparisons}
    assert status[("warm", "branch_a")] == "current"
    assert status[("warm", "branch_b")] == "current"
    assert status[("base", "branch_a")] == "stale"
    assert status[("small", "branch_b")] == "stale"
    report = format_comparison(comparisons)
    assert "OLD VERSION kept" in report
    assert "affected schemes: base, small" in report

    # Later run handles the remaining schemes.
    render_variant_batch(graph, variants, outputs, only=["base", "small"])
    assert all(
        item.status == "current"
        for item in compare_variants(graph, variants, outputs)
    )


def test_change_affecting_one_scheme_only_leaves_others_current(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    outputs = make_outputs(tmp_path)
    render_variant_batch(graph, variants, outputs)

    # "warm" overrides red; changing the base red param only affects schemes
    # that do not override it (base and small).
    graph.set_param("red", "color", [0.1, 0.1, 0.9, 1.0])
    comparisons = compare_variants(graph, variants, outputs)
    affected = affected_variants(comparisons)
    assert set(affected) == {"base", "small"}
    assert {item.node_id for item in affected["base"]} == {"branch_a"}
    warm = [item for item in comparisons if item.variant_id == "warm"]
    assert all(item.status == "current" for item in warm)


def test_corrupted_shared_result_recovers_without_undoing_others(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    outputs = make_outputs(tmp_path)
    render_variant_batch(graph, variants, outputs)

    warm_file = tmp_path / "warm" / "a.png"
    base_file = tmp_path / "base" / "a.png"
    base_bytes_before = base_file.read_bytes()

    # Corrupt one scheme's published output.
    data = bytearray(warm_file.read_bytes())
    data[-20:] = bytes(len(data[-20:]))
    warm_file.write_bytes(bytes(data))

    comparisons = compare_variants(graph, variants, outputs)
    affected = affected_variants(comparisons)
    assert set(affected) == {"warm"}
    assert affected["warm"][0].status == "stale"

    results = render_variant_batch(graph, variants, outputs)
    # The damaged scheme recovered...
    assert all(
        item.status == "current"
        for item in compare_variants(graph, variants, outputs)
    )
    # ...and the other schemes' published files were never touched.
    assert base_file.read_bytes() == base_bytes_before
    assert all(frame.skipped for frame in results["base"].frames)
    assert all(frame.skipped for frame in results["small"].frames)


def test_project_roundtrip_with_variants(tmp_path):
    graph, _ = build_graph()
    variants = make_variants()
    jobs = [
        RenderJob(
            id="preview",
            kind="static",
            node="branch_a",
            path="out/{variant}/a.png",
            width=32,
            height=32,
        ),
        RenderJob(
            id="only-warm",
            kind="static",
            node="branch_b",
            path="out/{variant}/b.png",
            width=32,
            height=32,
            variants=("warm", "small"),
        ),
    ]
    project_path = tmp_path / "project.json"
    save_project(str(project_path), graph, jobs=jobs, variants=variants)

    loaded_graph, _, data = load_project(str(project_path))
    assert data["version"] == 3
    assert [variant.id for variant in data["variants"]] == ["base", "small", "warm"]

    rendered = render_project(str(project_path))
    assert (tmp_path / "out" / "warm" / "a.png").exists()
    assert (tmp_path / "out" / "warm" / "b.png").exists()
    assert (tmp_path / "out" / "base" / "a.png").exists()
    # "only-warm" job does not run for the base scheme.
    assert not (tmp_path / "out" / "base" / "b.png").exists()

    # Recompute only the selected scheme via the project API.
    import os

    os.remove(tmp_path / "out" / "small" / "a.png")
    rendered = render_project(str(project_path), only_variants=["small"])
    assert (tmp_path / "out" / "small" / "a.png").exists()
