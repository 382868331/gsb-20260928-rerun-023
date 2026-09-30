from __future__ import annotations

import argparse

from .project import render_project_detailed


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a texture graph project")
    parser.add_argument("project", help="path to a texture graph JSON project")
    args = parser.parse_args()
    rendered, results = render_project_detailed(args.project)
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
