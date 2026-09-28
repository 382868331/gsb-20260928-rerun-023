from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import RenderCancelled
from .types import ImageBuffer


def save_png(image: ImageBuffer, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pil_image = Image.fromarray(np.asarray(image.data, dtype=np.uint8), mode="RGBA")
    pil_image.save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


def render_batch(graph, output_paths: dict[str, str | Path], should_cancel=None):
    if should_cancel is not None and should_cancel():
        raise RenderCancelled("Batch cancelled before rendering")
    results = graph.execute(output_ids=list(output_paths), should_cancel=should_cancel)
    if should_cancel is not None and should_cancel():
        raise RenderCancelled("Batch cancelled before output commit")

    committed = []
    for node_id, output_path in output_paths.items():
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        Image.fromarray(results[node_id].data, mode="RGBA").save(
            temporary, format="PNG", optimize=True
        )
        temporary.replace(output_path)
        committed.append(output_path)
    return committed
