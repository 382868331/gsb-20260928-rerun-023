import json
import os

import pytest
from PIL import Image

from texturegraph import (
    EvaluationCache,
    RenderCancelled,
    RenderJob,
    compare_variants,
    render_variant_batch,
    save_project,
)
from texturegraph.variants import load_variants
from examples.shared_mask import build_graph


def _write_variant_project(tmp_path):
    graph, _ = build_graph()
    jobs = [
        RenderJob(
            id="preview_a",
            kind="static",
            node="branch_a",
            path="outputs/branch_a.png",
            width=64,
            height=64,
            time=0.0,
        ),
        RenderJob(
            id="preview_b",
            kind="static",
            node="branch_b",
            path="outputs/branch_b.png",
            width=64,
            height=64,
            time=0.0,
        ),
    ]
    project_path = tmp_path / "project.json"
    save_project(str(project_path), graph, jobs=jobs)
    data = json.loads(project_path.read_text())
    data["variants"] = [
        {"id": "warm", "overrides": {"red.color": [0.95, 0.40, 0.10, 1.0]}},
        {"id": "cool", "overrides": {"red.color": [0.10, 0.30, 0.95, 1.0]}},
    ]
    project_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return str(project_path)


def _status(report, variant_id, node_id):
    for status in report[variant_id]:
        if status.node_id == node_id:
            return status.status
    raise AssertionError(f"no status for {variant_id}/{node_id}")


def _png_label(path):
    with Image.open(path) as image:
        return image.info.get("TextureGraph-Label")


def test_batch_renders_each_variant_without_clobbering(tmp_path):
    project = _write_variant_project(tmp_path)
    results = render_variant_batch(project)

    assert set(results) == {"warm", "cool"}
    warm_path = tmp_path / "variants" / "warm" / "outputs" / "branch_a.png"
    cool_path = tmp_path / "variants" / "cool" / "outputs" / "branch_a.png"
    assert warm_path.exists() and cool_path.exists()
    assert _png_label(warm_path) == "warm"
    assert _png_label(cool_path) == "cool"
    with Image.open(warm_path) as warm, Image.open(cool_path) as cool:
        assert warm.tobytes() != cool.tobytes()


def test_variants_share_intermediate_results(tmp_path):
    project = _write_variant_project(tmp_path)
    cache = EvaluationCache()
    render_variant_batch(project, cache=cache)

    # The shared mask has identical parameters in both variants, so its
    # signature is one shared cache entry computed once for the batch.
    import copy

    from texturegraph import TextureEngine, load_project
    from texturegraph.variants import apply_variant

    graph, _, _ = load_project(project)
    signatures = []
    for variant in load_variants(project):
        variant_graph = copy.deepcopy(graph)
        apply_variant(variant_graph, variant)
        engine = TextureEngine(variant_graph, cache)
        signatures.append(engine.signature_for("shared_mask", 64, 64, 0.0))
    assert signatures[0] == signatures[1]
    assert cache.get(signatures[0]) is not None


def test_cancel_after_first_variant_and_resume(tmp_path):
    project = _write_variant_project(tmp_path)
    warm_output = tmp_path / "variants" / "warm" / "outputs" / "branch_b.png"

    def cancel_once_warm_done():
        return warm_output.exists()

    with pytest.raises(RenderCancelled):
        render_variant_batch(project, is_cancelled=cancel_once_warm_done)

    report = compare_variants(project)
    assert _status(report, "warm", "branch_a") == "current"
    assert _status(report, "warm", "branch_b") == "current"
    assert _status(report, "cool", "branch_a") == "missing"

    render_variant_batch(project)
    report = compare_variants(project)
    assert _status(report, "cool", "branch_a") == "current"
    assert _status(report, "warm", "branch_a") == "current"


def test_shared_input_change_lists_affected_variants_and_outputs(tmp_path):
    project = _write_variant_project(tmp_path)
    render_variant_batch(project)

    data = json.loads(open(project).read())
    for node in data["nodes"]:
        if node["id"] == "dark_blue":
            node["params"]["color"] = [0.5, 0.5, 0.5, 1.0]
    with open(project, "w") as handle:
        json.dump(data, handle)

    report = compare_variants(project)
    # dark_blue only feeds branch_a: both variants' branch_a outputs are
    # affected, every branch_b output stays current and usable.
    assert _status(report, "warm", "branch_a") == "stale"
    assert _status(report, "cool", "branch_a") == "stale"
    assert _status(report, "warm", "branch_b") == "current"
    assert _status(report, "cool", "branch_b") == "current"


def test_selective_recompute_updates_only_chosen_variant(tmp_path):
    project = _write_variant_project(tmp_path)
    render_variant_batch(project)

    data = json.loads(open(project).read())
    for node in data["nodes"]:
        if node["id"] == "shared_mask":
            node["params"]["radius"] = 0.2
    with open(project, "w") as handle:
        json.dump(data, handle)

    results = render_variant_batch(project, only=["warm"])
    assert set(results) == {"warm"}

    report = compare_variants(project)
    assert _status(report, "warm", "branch_a") == "current"
    assert _status(report, "warm", "branch_b") == "current"
    # cool keeps its old published versions, clearly reported as stale.
    assert _status(report, "cool", "branch_a") == "stale"
    assert _status(report, "cool", "branch_b") == "stale"


def test_corrupted_shared_result_recovers_without_undoing_others(tmp_path):
    project = _write_variant_project(tmp_path)
    render_variant_batch(project)

    victim = tmp_path / "variants" / "warm" / "outputs" / "branch_a.png"
    payload = bytearray(victim.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    victim.write_bytes(bytes(payload))

    report = compare_variants(project)
    assert _status(report, "warm", "branch_a") == "stale"
    assert _status(report, "warm", "branch_b") == "current"
    assert _status(report, "cool", "branch_a") == "current"

    results = render_variant_batch(project)
    rendered = {
        node
        for frame in results["warm"]
        for node in frame.rendered
    }
    reused_cool = {
        node
        for frame in results["cool"]
        for node in frame.reused
    }
    assert rendered == {"branch_a"}
    assert reused_cool == {"branch_a", "branch_b"}

    report = compare_variants(project)
    assert _status(report, "warm", "branch_a") == "current"
