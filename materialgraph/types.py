from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Color:
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255

    def __post_init__(self):
        for value in (self.r, self.g, self.b, self.a):
            if not 0 <= int(value) <= 255:
                raise ValueError("Color channels must be in range 0..255")

    @property
    def rgba(self):
        return np.asarray([self.r, self.g, self.b, self.a], dtype=np.uint8)

    def to_dict(self):
        return [self.r, self.g, self.b, self.a]

    @classmethod
    def from_dict(cls, value):
        return cls(*value)


@dataclasses.dataclass(frozen=True)
class CoordinateGrid:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if self.x.shape != self.y.shape or self.x.ndim != 2:
            raise ValueError("Coordinate grids must contain equally shaped 2D arrays")

    @property
    def shape(self):
        return self.x.shape


@dataclasses.dataclass(frozen=True)
class Mask:
    data: np.ndarray

    def __post_init__(self):
        data = np.asarray(self.data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        object.__setattr__(self, "data", np.clip(data, 0.0, 1.0))


@dataclasses.dataclass(frozen=True)
class ImageBuffer:
    data: np.ndarray

    def __post_init__(self):
        data = np.asarray(self.data, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError("Image buffer must have RGBA shape (height, width, 4)")
        object.__setattr__(self, "data", data)
