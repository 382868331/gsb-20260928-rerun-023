from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]
Mask = NDArray[np.float32]
Coordinates = NDArray[np.float32]


@dataclass(frozen=True)
class Port:
    name: str
    type: str
    required: bool = True


@dataclass(frozen=True)
class NodeType:
    name: str
    inputs: tuple[Port, ...]
    output_type: str
    execute: Callable[..., Any]

    def input(self, name: str) -> Port:
        for port in self.inputs:
            if port.name == name:
                return port
        raise KeyError(name)


def _params(params: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    return {"width": width, "height": height, **params}


def solid_color(inputs, params, width, height):
    color = params.get("color", [0.0, 0.0, 0.0, 1.0])
    values = np.asarray(color, dtype=np.float32)
    if values.shape != (4,):
        raise ValueError("solid_color color must contain RGBA four numbers")
    return np.broadcast_to(values, (height, width, 4)).copy()


def uv_coordinates(inputs, params, width, height):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    y = (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0 - 1.0
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def transform_coordinates(inputs, params, width, height):
    coords = inputs["coordinates"].copy()
    scale_x = float(params.get("scale_x", 1.0))
    scale_y = float(params.get("scale_y", 1.0))
    offset_x = float(params.get("offset_x", 0.0))
    offset_y = float(params.get("offset_y", 0.0))
    angle = np.deg2rad(float(params.get("rotation_degrees", 0.0)))
    coords[..., 0] -= offset_x
    coords[..., 1] -= offset_y
    cosine, sine = np.cos(angle), np.sin(angle)
    x = coords[..., 0] * cosine + coords[..., 1] * sine
    y = -coords[..., 0] * sine + coords[..., 1] * cosine
    coords[..., 0] = x * scale_x
    coords[..., 1] = y * scale_y
    return coords


def circle_mask(inputs, params, width, height):
    coords = inputs["coordinates"]
    center_x = float(params.get("center_x", 0.0))
    center_y = float(params.get("center_y", 0.0))
    radius = float(params.get("radius", 0.5))
    softness = max(0.0, float(params.get("softness", 0.0)))
    distance = np.sqrt(
        (coords[..., 0] - center_x) ** 2 + (coords[..., 1] - center_y) ** 2
    )
    if softness == 0.0:
        return (distance <= radius).astype(np.float32)
    edge = np.clip((radius + softness - distance) / (2.0 * softness), 0.0, 1.0)
    return edge.astype(np.float32)


def rectangle_mask(inputs, params, width, height):
    coords = inputs["coordinates"]
    center_x = float(params.get("center_x", 0.0))
    center_y = float(params.get("center_y", 0.0))
    half_width = float(params.get("half_width", 0.5))
    half_height = float(params.get("half_height", 0.5))
    softness = max(0.0, float(params.get("softness", 0.0)))
    dx = np.abs(coords[..., 0] - center_x) - half_width
    dy = np.abs(coords[..., 1] - center_y) - half_height
    outside = np.maximum(dx, dy)
    if softness == 0.0:
        return (outside <= 0.0).astype(np.float32)
    return np.clip(softness - outside, 0.0, 1.0).astype(np.float32)


def combine_masks(inputs, params, width, height):
    a = inputs["a"]
    b = inputs["b"]
    operation = params.get("operation", "multiply")
    if operation == "multiply":
        return a * b
    if operation == "add":
        return np.clip(a + b, 0.0, 1.0)
    if operation == "subtract":
        return np.clip(a - b, 0.0, 1.0)
    if operation == "maximum":
        return np.maximum(a, b)
    if operation == "minimum":
        return np.minimum(a, b)
    raise ValueError(f"unsupported mask operation: {operation}")


def blur_mask(inputs, params, width, height):
    """Box blur over a mask.

    Neighborhood operation: an output pixel depends on input pixels within
    ``radius_px`` in every direction, so region edits feeding a blur must
    expand their recompute footprint past the edited rectangle.
    """

    mask = np.asarray(inputs["mask"], dtype=np.float32)
    radius = max(0, int(params.get("radius_px", 1)))
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="edge")
    cumulative = np.pad(
        padded, ((1, 0), (1, 0)), mode="constant"
    ).cumsum(axis=0).cumsum(axis=1)
    diameter = 2 * radius + 1
    total = (
        cumulative[diameter:, diameter:]
        - cumulative[:-diameter, diameter:]
        - cumulative[diameter:, :-diameter]
        + cumulative[:-diameter, :-diameter]
    )
    return (total / (diameter * diameter)).astype(np.float32)


def blend_colors(inputs, params, width, height):
    foreground = inputs["foreground"]
    background = inputs["background"]
    mask = inputs["mask"]
    factor = 1.0 - float(params.get("invert_mask", False))
    if factor == 0.0:
        mask = 1.0 - mask
    alpha = np.clip(mask, 0.0, 1.0)[..., None]
    return (foreground * alpha + background * (1.0 - alpha)).astype(np.float32)


NODE_TYPES: dict[str, NodeType] = {
    node_type.name: node_type
    for node_type in [
        NodeType("solid_color", (), "image", solid_color),
        NodeType("uv_coordinates", (), "coordinates", uv_coordinates),
        NodeType(
            "transform_coordinates",
            (Port("coordinates", "coordinates"),),
            "coordinates",
            transform_coordinates,
        ),
        NodeType(
            "circle_mask",
            (Port("coordinates", "coordinates"),),
            "mask",
            circle_mask,
        ),
        NodeType(
            "rectangle_mask",
            (Port("coordinates", "coordinates"),),
            "mask",
            rectangle_mask,
        ),
        NodeType(
            "combine_masks",
            (Port("a", "mask"), Port("b", "mask")),
            "mask",
            combine_masks,
        ),
        NodeType(
            "blur_mask",
            (Port("mask", "mask"),),
            "mask",
            blur_mask,
        ),
        NodeType(
            "blend_colors",
            (
                Port("foreground", "image"),
                Port("background", "image"),
                Port("mask", "mask"),
            ),
            "image",
            blend_colors,
        ),
    ]
}
