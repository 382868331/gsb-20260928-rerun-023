"""Region-scoped partial recomputation example.

Run:

    python examples/region_update.py

The example publishes a baseline scheme, proposes two local candidates
for regions whose rectangles are equal in size and name but carry
different input versions, previews which pixels must be recomputed
(including the box-blur halo), adopts one candidate, cancels/resumes a
multi-region publish, and finally compares the assembled image against a
full from-scratch recomputation.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from texturegraph import (
    EvaluationCache,
    Graph,
    Region,
    RegionCandidate,
    RegionPublisher,
    Variant,
)

WIDTH = HEIGHT = 64


def build_graph() -> Graph:
    graph = Graph(width=WIDTH, height=HEIGHT)
    graph.add_node("uv", "uv_coordinates")
    graph.add_node("circle", "circle_mask", {
        "center_x": -0.3, "center_y": 0.0, "radius": 0.25, "softness": 0.0,
    })
    # Neighborhood operation: region edits feeding this blur must grow
    # their recompute footprint by radius_px in every direction.
    graph.add_node("blur", "blur_mask", {"radius_px": 4})
    graph.add_node("red", "solid_color", {"color": [0.9, 0.1, 0.1, 1.0]})
    graph.add_node("blue", "solid_color", {"color": [0.05, 0.1, 0.35, 1.0]})
    graph.add_node("out", "blend_colors")
    graph.connect("uv", "circle", "coordinates")
    graph.connect("circle", "blur", "mask")
    graph.connect("red", "out", "foreground")
    graph.connect("blue", "out", "background")
    graph.connect("blur", "out", "mask")
    return graph


def main() -> None:
    root = ROOT / "examples" / "region-outputs"
    shutil.rmtree(root, ignore_errors=True)
    graph = build_graph()
    publisher_ = RegionPublisher(
        graph, "out", WIDTH, HEIGHT, str(root), cache=EvaluationCache()
    )

    regions = [
        Region("left", 0, 0, WIDTH // 2, HEIGHT, input_version="photo-v1"),
        Region("right", WIDTH // 2, 0, WIDTH // 2, HEIGHT, input_version="photo-v1"),
    ]
    baseline = publisher_.publish_baseline(regions, "r0-baseline")
    print("baseline:", baseline.path)

    # Two competing candidates for the SAME rectangle (same name+size,
    # different authoring input version): compare against the same base.
    candidate_a = RegionCandidate(
        "move-right",
        Region("left", 0, 0, WIDTH // 2, HEIGHT, input_version="photo-v2"),
        Variant("move-right", {("circle", "center_x"): 0.35}),
        label="shift-circle-right",
    )
    candidate_b = RegionCandidate(
        "move-up",
        Region("left", 0, 0, WIDTH // 2, HEIGHT, input_version="photo-v1"),
        Variant("move-up", {("circle", "center_y"): 0.3}),
        label="shift-circle-up",
    )
    _base, plans = publisher_.preview_update([candidate_a, candidate_b], baseline)
    print(json.dumps([plan.to_dict() for plan in plans], indent=2))

    report = publisher_.compare_candidate_impacts(
        [candidate_a, candidate_b], baseline
    )
    for candidate_id, item in report.items():
        print(
            candidate_id,
            "changed pixels:", item["changed_pixels"],
            "inside footprint:", item["within_recompute_footprint"],
        )

    # Adopt exactly one candidate for the region.
    updated = publisher_.publish_update(
        {"left": candidate_a}, "r1-adopt-a", baseline, plans
    )
    print("released:", updated.path)
    manifest = Path(publisher_.release_manifest_path(updated.release_id))
    print("provenance manifest:", manifest)

    # Cross-check: assembled authored region equals full recomputation.
    assembled = np.array(
        __import__("PIL.Image", fromlist=["Image"]).open(updated.path)
        .convert("RGBA")
    )
    full = publisher_.render_full_comparison(candidate_a)
    same = np.array_equal(assembled[:, : WIDTH // 2], full[:, : WIDTH // 2])
    print("partial region matches full recompute:", same)

    # After a later input change, per-region status shows what is stale;
    # old release r0/r1 files are retained on disk and stay inspectable.
    graph.set_param("circle", "radius", 0.4)
    statuses = publisher_.inspect_release(updated, graph)
    for status in statuses:
        print(status.release_id, status.region_id, status.status, "-", status.reason)


if __name__ == "__main__":
    main()
