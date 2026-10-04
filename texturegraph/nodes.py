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


# How far (in pixels) a node mixes neighbouring pixels into each output
# pixel.  0 means purely pointwise; filters/neighbourhood ops advertise a
# positive radius so region recomputes can dilate a user rectangle into the
# full halo that must be recomputed to avoid visible seams.
#
# Transformations of coordinates are NOT neighbourhood operations: every
# output pixel still depends on at most one input location.
NEIGHBORHOOD_RADIUS_PARAM: dict[str, str] = {}


def neighborhood_radius(node_type: str, params: dict[str, Any]) -> int:
    """Pixel radius that must be added around a region for ``node_type``."""
    param_name = NEIGHBORHOOD_RADIUS_PARAM.get(node_type)
    if param_name is None:
        return 0
    return max(0, int(round(float(params.get(param_name, 0)))))


def _neighborhood_box(values: NDArray[np.float32], radius: int) -> NDArray[np.float32]:
    if radius <= 0:
        return values
    count = 2 * radius + 1
    padded = np.pad(values, ((radius, radius), (radius, radius)) + ((0, 0),) * (values.ndim - 2), mode="edge")
    # Separable moving average; edge-replication padding keeps frame borders
    # stable so only genuinely neighbouring pixels influence a region edge.
    working = padded
    for axis in (0, 1):
        windows = []
        for offset in range(count):
            index = tuple(
                slice(offset, offset + values.shape[ax])
                if ax == axis
                else slice(None)
                for ax in range(values.ndim)
            )
            windows.append(working[index])
        working = (sum(windows) / float(count)).astype(np.float32)
    return working


def box_blur(inputs, params, width, height):
    """Uniform box filter on an image; a genuine cross-region neighbourhood op."""
    image = inputs["image"]
    radius = max(0, int(round(float(params.get("radius", 1)))))
    return _neighborhood_box(np.asarray(image, dtype=np.float32), radius)


def crop_box(
    values: NDArray[np.float32],
    source_box: "tuple[int, int, int, int]",
    wanted_box: "tuple[int, int, int, int]",
) -> NDArray[np.float32]:
    """Slice a stored crop (located at ``source_box`` on the canvas) to ``wanted_box``."""
    top, left, bottom, right = wanted_box
    src_top, src_left, _, _ = source_box
    row0, row1 = top - src_top, bottom - src_top
    col0, col1 = left - src_left, right - src_left
    if row0 < 0 or col0 < 0 or row1 > values.shape[0] or col1 > values.shape[1]:
        raise ValueError("requested crop extends beyond the available window")
    return values[row0:row1, col0:col1]


def box_blur_window(
    image_crop: NDArray[np.float32],
    params: dict[str, Any],
    output_box: "tuple[int, int, int, int]",
    crop_origin: "tuple[int, int]",
) -> NDArray[np.float32]:
    """Blur a crop that already contains the full halo for ``output_box``.

    Edge-replication padding here only ever feeds halo pixels that are sliced
    off again (or genuine canvas borders), so every published output pixel
    sees its complete real neighbourhood.
    """
    radius = max(0, int(round(float(params.get("radius", 1)))))
    blurred = _neighborhood_box(np.asarray(image_crop, dtype=np.float32), radius)
    out_top, out_left, out_bottom, out_right = output_box
    crop_top, crop_left = crop_origin
    row0 = out_top - crop_top
    col0 = out_left - crop_left
    return blurred[row0 : row0 + (out_bottom - out_top), col0 : col0 + (out_right - out_left)]


NODE_TYPES["box_blur"] = NodeType(
    "box_blur",
    (Port("image", "image"),),
    "image",
    box_blur,
)
NEIGHBORHOOD_RADIUS_PARAM["box_blur"] = "radius"
