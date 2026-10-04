"""A small cached, deterministic texture-node graph engine."""

from .animation import Keyframe, ParameterTrack
from .engine import (
    CACHE_VERSION,
    EvaluationCache,
    FrameOutput,
    FrameRequest,
    FrameResult,
    RenderContext,
    TextureEngine,
)
from .errors import CycleError, GraphValidationError, RenderCancelled
from .graph import Connection, Graph, Node
from .project import RenderJob, load_project, render_project, save_project
from .variants import (
    Variant,
    VariantOutputStatus,
    compare_variants,
    render_variant_batch,
)

from .regions import (
    LocalEvaluator,
    RegionBox,
    RegionCandidate,
    RegionError,
    RegionWorkspace,
    Zone,
    compute_signatures,
    evaluate_full,
    judge_affected_range,
    load_region_workspace,
)

__all__ = [
    "CACHE_VERSION",
    "Connection",
    "CycleError",
    "EvaluationCache",
    "FrameOutput",
    "FrameRequest",
    "FrameResult",
    "Graph",
    "GraphValidationError",
    "Keyframe",
    "Node",
    "ParameterTrack",
    "LocalEvaluator",
    "RegionBox",
    "RegionCandidate",
    "RegionError",
    "RegionWorkspace",
    "RenderCancelled",
    "RenderContext",
    "RenderJob",
    "TextureEngine",
    "Variant",
    "VariantOutputStatus",
    "Zone",
    "compute_signatures",
    "compare_variants",
    "evaluate_full",
    "judge_affected_range",
    "load_project",
    "load_region_workspace",
    "render_project",
    "render_variant_batch",
    "save_project",
]
