from __future__ import annotations

import argparse
from pathlib import Path

from .examples.shared_mask import build_shared_mask_example, write_shared_mask_example
from .project import load_project
from .render import render_batch


def _parse_node_path(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected --output NODE_ID=PATH")
    node_id, path = value.split("=", 1)
    return node_id, path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="materialgraph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render a saved project")
    render_parser.add_argument("project", type=Path)
    render_parser.add_argument("--output", action="append", type=_parse_node_path, default=[])

    demo_parser = subparsers.add_parser("demo", help="Create and render shared-mask example")
    demo_parser.add_argument("directory", type=Path, nargs="?", default=Path("dist/shared-mask"))

    args = parser.parse_args(argv)
    if args.command == "render":
        graph, saved_outputs = load_project(args.project)
        output_paths = dict(args.output) if args.output else saved_outputs
        if not output_paths:
            parser.error("Project has no outputs; pass --output NODE_ID=PATH")
        if not args.output:
            output_paths = {
                node_id: args.project.parent / path for node_id, path in output_paths.items()
            }
        rendered = render_batch(graph, output_paths)
        for path in rendered:
            print(path)
    elif args.command == "demo":
        _, paths = write_shared_mask_example(args.directory)
        for path in paths.values():
            print(path)


if __name__ == "__main__":
    main()
