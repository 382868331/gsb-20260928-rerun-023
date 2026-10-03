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
    Rect,
    Region,
    RegionCandidate,
    RegionPlan,
    RegionPublisher,
    RegionRelease,
    RegionStatus,
    SharedDamageReport,
    build_plan,
    inspect_all_releases,
    preview_plans,
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
    "RenderCancelled",
    "RenderContext",
    "RenderJob",
    "Rect",
    "Region",
    "RegionCandidate",
    "RegionPlan",
    "RegionPublisher",
    "RegionRelease",
    "RegionStatus",
    "SharedDamageReport",
    "TextureEngine",
    "Variant",
    "VariantOutputStatus",
    "build_plan",
    "compare_variants",
    "inspect_all_releases",
    "load_project",
    "preview_plans",
    "render_project",
    "render_variant_batch",
    "save_project",
]
