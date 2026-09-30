from __future__ import annotations

import argparse

from .project import render_project_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a texture graph project")
    parser.add_argument("project", help="path to a texture graph JSON project")
    args = parser.parse_args()
    rendered, results = render_project_report(args.project)
    for path in rendered:
        action = "重算"
        reason = None
        for result in results:
            if path in result.recomputed:
                action = "重算"
                reason = result.reasons.get(path)
                break
            if path in result.reused:
                action = "复用"
                reason = result.reasons.get(path)
                break
        suffix = f" ({reason})" if reason else ""
        print(f"{action}: {path}{suffix}")


if __name__ == "__main__":
    main()
