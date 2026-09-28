from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .types import Color, CoordinateGrid, ImageBuffer, Mask


@dataclass
class Node:
    id: str
    params: dict = field(default_factory=dict)

    type_name = "node"
    input_ports: ClassVar[dict[str, str]] = {}
    output_type: ClassVar[str] = ""

    def config_signature(self):
        return repr((self.type_name, _stable(self.params)))

    def compute(self, inputs, ctx=None):
        raise NotImplementedError


def _stable(value):
    if isinstance(value, dict):
        return sorted((key, _stable(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


class CoordinatesNode(Node):
    type_name = "coordinates"
    output_type = "coordinates"

    def compute(self, inputs, ctx=None):
        width = int(self.params["width"])
        height = int(self.params["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Coordinate dimensions must be positive")
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        return CoordinateGrid(xx, yy)


class TransformCoordinatesNode(Node):
    type_name = "transform_coordinates"
    input_ports = {"coordinates": "coordinates"}
    output_type = "coordinates"

    def compute(self, inputs, ctx=None):
        grid = inputs["coordinates"]
        x = grid.x - 0.5
        y = grid.y - 0.5
        angle = math.radians(float(self.params.get("rotation_degrees", 0.0)))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        scale_x = float(self.params.get("scale_x", 1.0))
        scale_y = float(self.params.get("scale_y", 1.0))
        if scale_x == 0 or scale_y == 0:
            raise ValueError("Coordinate scale must not be zero")
        rx = cos_a * x - sin_a * y
        ry = sin_a * x + cos_a * y
        out_x = rx / scale_x + 0.5 + float(self.params.get("offset_x", 0.0))
        out_y = ry / scale_y + 0.5 + float(self.params.get("offset_y", 0.0))
        return CoordinateGrid(out_x.astype(np.float32), out_y.astype(np.float32))


class ConstantColorNode(Node):
    type_name = "constant_color"
    output_type = "color"

    def compute(self, inputs, ctx=None):
        return Color(*self.params["rgba"])


class GradientColorNode(Node):
    type_name = "gradient_color"
    input_ports = {"coordinates": "coordinates"}
    output_type = "image"

    def compute(self, inputs, ctx=None):
        grid = inputs["coordinates"]
        axis = self.params.get("axis", "x")
        t = grid.x if axis == "x" else grid.y
        start = np.asarray(self.params["start_rgba"], dtype=np.float32)
        end = np.asarray(self.params["end_rgba"], dtype=np.float32)
        factor = t[..., None]
        pixels = start * (1.0 - factor) + end * factor
        return ImageBuffer(np.rint(pixels).astype(np.uint8))


class CircleMaskNode(Node):
    type_name = "circle_mask"
    input_ports = {"coordinates": "coordinates"}
    output_type = "mask"

    def compute(self, inputs, ctx=None):
        grid = inputs["coordinates"]
        center_x = float(self.params.get("center_x", 0.5))
        center_y = float(self.params.get("center_y", 0.5))
        radius = float(self.params.get("radius", 0.25))
        softness = float(self.params.get("softness", 0.0))
        distance = np.sqrt((grid.x - center_x) ** 2 + (grid.y - center_y) ** 2)
        if softness <= 0.0:
            data = (distance <= radius).astype(np.float32)
        else:
            data = np.clip((radius + softness - distance) / softness, 0.0, 1.0)
        return Mask(data)


class RectangleMaskNode(Node):
    type_name = "rectangle_mask"
    input_ports = {"coordinates": "coordinates"}
    output_type = "mask"

    def compute(self, inputs, ctx=None):
        grid = inputs["coordinates"]
        cx = float(self.params.get("center_x", 0.5))
        cy = float(self.params.get("center_y", 0.5))
        half_w = float(self.params.get("half_width", 0.25))
        half_h = float(self.params.get("half_height", 0.25))
        softness = float(self.params.get("softness", 0.0))
        edge = np.maximum(np.abs(grid.x - cx) - half_w, np.abs(grid.y - cy) - half_h)
        if softness <= 0.0:
            data = (edge <= 0.0).astype(np.float32)
        else:
            data = np.clip((softness - edge) / softness, 0.0, 1.0)
        return Mask(data)


class CombineMasksNode(Node):
    type_name = "combine_masks"
    input_ports = {"a": "mask", "b": "mask"}
    output_type = "mask"

    def compute(self, inputs, ctx=None):
        a = inputs["a"].data
        b = inputs["b"].data
        operation = self.params.get("operation", "intersect")
        if operation == "union":
            data = np.maximum(a, b)
        elif operation == "intersect":
            data = np.minimum(a, b)
        elif operation == "subtract":
            data = np.clip(a - b, 0.0, 1.0)
        elif operation == "xor":
            data = np.abs(a - b)
        else:
            raise ValueError(f"Unknown mask operation: {operation}")
        return Mask(data.astype(np.float32))


class MixColorNode(Node):
    type_name = "mix_color"
    input_ports = {"mask": "mask", "background": "color", "foreground": "color"}
    output_type = "image"

    def compute(self, inputs, ctx=None):
        mask = inputs["mask"].data[..., None]
        bg = inputs["background"].rgba.astype(np.float32)
        fg = inputs["foreground"].rgba.astype(np.float32)
        pixels = bg * (1.0 - mask) + fg * mask
        return ImageBuffer(np.rint(pixels).astype(np.uint8))


class OutputNode(Node):
    type_name = "output"
    input_ports = {"image": "image"}
    output_type = "image"

    def compute(self, inputs, ctx=None):
        return inputs["image"]


NODE_TYPES = {
    node.type_name: node
    for node in (
        CoordinatesNode,
        TransformCoordinatesNode,
        ConstantColorNode,
        GradientColorNode,
        CircleMaskNode,
        RectangleMaskNode,
        CombineMasksNode,
        MixColorNode,
        OutputNode,
    )
}
