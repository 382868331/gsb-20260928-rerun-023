"""Compare several parameter schemes of the shared-mask graph in one batch.

Run:  python examples/variants.py

Renders three schemes (base / warm foreground / smaller shared mask) into
examples/outputs/variants/<scheme>/, then prints the comparison page showing
which outputs are current results and which are kept older versions.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from texturegraph import (
    RenderJob,
    Variant,
    compare_variants,
    format_comparison,
    load_project,
    render_project,
    save_project,
)
from texturegraph.variants import VariantOutput
from examples.shared_mask import build_graph

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "variants"


def build_variants() -> list[Variant]:
    return [
        Variant("base", label="原始方案"),
        Variant(
            "warm",
            overrides={"red": {"color": [0.95, 0.55, 0.10, 1.0]}},
            label="暖色前景",
        ),
        Variant(
            "small",
            overrides={"shared_mask": {"radius": 0.25}},
            label="更小的共享遮罩",
        ),
    ]


def main() -> None:
    graph, _ = build_graph()
    variants = build_variants()
    jobs = [
        RenderJob(
            id="preview-a",
            kind="static",
            node="branch_a",
            path="outputs/variants/{variant}/branch_a.png",
            width=64,
            height=64,
            time=0.5,
        ),
        RenderJob(
            id="preview-b",
            kind="static",
            node="branch_b",
            path="outputs/variants/{variant}/branch_b.png",
            width=64,
            height=64,
            time=0.5,
        ),
    ]
    project_path = Path(__file__).resolve().parent / "variants-project.json"
    save_project(str(project_path), graph, jobs=jobs, variants=variants)

    rendered = render_project(str(project_path))
    for path in sorted(rendered.values()):
        print("published:", path)

    graph, _, data = load_project(str(project_path))
    variants = data["variants"]
    outputs = [
        VariantOutput(
            job.node,
            str(project_path.parent / job.path),
            job.width,
            job.height,
            time=job.time,
        )
        for job in data["jobs"]
    ]
    print(format_comparison(compare_variants(graph, variants, outputs)))


if __name__ == "__main__":
    main()
