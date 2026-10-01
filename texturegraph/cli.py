from __future__ import annotations

import argparse

from .project import load_project, render_project_detailed
from .variants import compare_variants, render_variant_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a texture graph project")
    parser.add_argument("project", help="path to a texture graph JSON project")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="only process the named variant (repeatable); default is all",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="report per-variant output status without rendering",
    )
    args = parser.parse_args()
    _, _, data = load_project(args.project)
    if data.get("variants"):
        if args.compare:
            report = compare_variants(args.project, only=args.variants)
            for variant_id, statuses in report.items():
                print(f"variant {variant_id}:")
                for status in statuses:
                    print(f"  {status.status} {status.node_id}: {status.path}")
                    print(f"    why: {status.reason}")
            return
        results = render_variant_batch(args.project, only=args.variants)
        for variant_id, variant_results in results.items():
            print(f"variant {variant_id}:")
            _print_results(variant_results)
        return
    rendered, results = render_project_detailed(args.project)
    _print_results(results)


def _print_results(results) -> None:
    for result in results:
        action = "reused" if result.skipped else "rendered"
        print(f"t={result.time:g}: {action}")
        for node_id, path in result.reused.items():
            print(f"  reused {node_id}: {path}")
            print(f"    why: {result.reasons[node_id]}")
        for node_id, path in result.rendered.items():
            print(f"  rendered {node_id}: {path}")
            print(f"    why: {result.reasons[node_id]}")


if __name__ == "__main__":
    main()
