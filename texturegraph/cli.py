from __future__ import annotations

import argparse

from .project import load_project, render_project_detailed
from .variants import (
    VariantOutput,
    compare_variants,
    format_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a texture graph project")
    parser.add_argument("project", help="path to a texture graph JSON project")
    parser.add_argument(
        "--variants",
        help="comma-separated variant ids to (re)compute; others keep their "
        "published outputs for a later run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="do not render; only report which scheme outputs are current "
        "results and which are kept older versions",
    )
    args = parser.parse_args()
    only = args.variants.split(",") if args.variants else None

    if args.compare:
        print(_comparison_report(args.project))
        return

    _, results = render_project_detailed(args.project, only_variants=only)
    for result in results:
        scheme = f" scheme={result.variant}" if result.variant else ""
        action = "reused" if result.skipped else "rendered"
        print(f"t={result.time:g}{scheme}: {action}")
        for node_id, path in result.reused.items():
            print(f"  reused {node_id}: {path}")
            print(f"    why: {result.reasons[node_id]}")
        for node_id, path in result.rendered.items():
            print(f"  rendered {node_id}: {path}")
            print(f"    why: {result.reasons[node_id]}")

    report = _comparison_report(args.project)
    if report is not None:
        print(report)


def _comparison_report(project_path: str) -> str | None:
    import os

    graph, _, data = load_project(project_path)
    variants = data.get("variants", [])
    jobs = data.get("jobs", [])
    if not variants:
        return None
    project_directory = os.path.dirname(os.path.abspath(project_path))
    comparisons = []
    for variant in variants:
        variant_outputs: list[VariantOutput] = []
        for job in jobs:
            if job.variants is not None and variant.id not in job.variants:
                continue
            for request in job.frames(project_directory, variant.id):
                for output in request.outputs:
                    variant_outputs.append(
                        VariantOutput(
                            output.node_id,
                            output.path,
                            output.width,
                            output.height,
                            time=float(request.time),
                        )
                    )
        comparisons.extend(compare_variants(graph, [variant], variant_outputs))
    return format_comparison(comparisons)


if __name__ == "__main__":
    main()
