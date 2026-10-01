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
    OutputComparison,
    Variant,
    VariantBatchResult,
    VariantOutput,
    affected_variants,
    apply_variant,
    compare_variants,
    format_comparison,
    render_variant_batch,
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
    "OutputComparison",
    "ParameterTrack",
    "RenderCancelled",
    "RenderContext",
    "RenderJob",
    "TextureEngine",
    "Variant",
    "VariantBatchResult",
    "VariantOutput",
    "affected_variants",
    "apply_variant",
    "compare_variants",
    "format_comparison",
    "load_project",
    "render_project",
    "render_variant_batch",
    "save_project",
]
