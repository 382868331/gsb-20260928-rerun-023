from __future__ import annotations

import argparse

from .project import render_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a texture graph project")
    parser.add_argument("project", help="path to a texture graph JSON project")
    args = parser.parse_args()
    rendered = render_project(args.project)
    for node_id, path in rendered.items():
        print(f"{node_id}: {path}")


if __name__ == "__main__":
    main()
