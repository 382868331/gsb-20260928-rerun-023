"""Time-varying node parameters expressed as keyframe tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

STEP = "step"
LINEAR = "linear"
INTERPOLATIONS = (STEP, LINEAR)


@dataclass(frozen=True)
class Keyframe:
    time: float
    value: Any


@dataclass(frozen=True)
class ParameterTrack:
    """Animated value for one node/parameter pair."""

    node_id: str
    param_name: str
    keyframes: tuple[Keyframe, ...] = ()
    interpolation: str = LINEAR

    def __post_init__(self) -> None:
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"unsupported interpolation: {self.interpolation}")
        if len(self.keyframes) < 1:
            raise ValueError("a track needs at least one keyframe")
        times = [float(frame.time) for frame in self.keyframes]
        if any(left >= right for left, right in zip(times, times[1:])):
            raise ValueError("keyframe times must be strictly increasing")
        shapes = {
            np.asarray(frame.value, dtype=np.float64).shape
            for frame in self.keyframes
        }
        if len(shapes) != 1:
            raise ValueError("all keyframes for one track must share a shape")

    def value_at(self, time: float) -> Any:
        time = float(time)
        first = self.keyframes[0]
        last = self.keyframes[-1]
        if time <= float(first.time):
            return first.value
        if time >= float(last.time):
            return last.value

        for left, right in zip(self.keyframes, self.keyframes[1:]):
            left_time = float(left.time)
            right_time = float(right.time)
            if time < left_time or time > right_time:
                continue
            if self.interpolation == STEP or time == left_time:
                return left.value
            fraction = (time - left_time) / (right_time - left_time)
            left_values = np.asarray(left.value, dtype=np.float64)
            right_values = np.asarray(right.value, dtype=np.float64)
            result = left_values + (right_values - left_values) * fraction
            result = result.reshape(left_values.shape)
            if left_values.ndim == 0:
                return float(result)
            # Preserve the keyframe container type (list stays list) so the
            # serialized signature matches an equivalent static parameter.
            if isinstance(left.value, (list, tuple)):
                return [float(item) for item in result.reshape(-1)]
            return result.tolist()
        return last.value


def tracks_to_list(tracks: Iterable[ParameterTrack]) -> list[dict]:
    documents: list[dict] = []
    for track in sorted(tracks, key=lambda item: (item.node_id, item.param_name)):
        documents.append(
            {
                "node": track.node_id,
                "param": track.param_name,
                "interpolation": track.interpolation,
                "keyframes": [
                    {"time": float(frame.time), "value": frame.value}
                    for frame in track.keyframes
                ],
            }
        )
    return documents


def track_from_dict(document: dict) -> ParameterTrack:
    keyframes = tuple(
        Keyframe(time=float(frame["time"]), value=frame["value"])
        for frame in document["keyframes"]
    )
    return ParameterTrack(
        node_id=document["node"],
        param_name=document["param"],
        keyframes=keyframes,
        interpolation=document.get("interpolation", LINEAR),
    )
