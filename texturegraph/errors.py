class GraphValidationError(ValueError):
    """Raised when a graph is structurally invalid."""


class CycleError(GraphValidationError):
    """Raised when a connection cycle is found before evaluation."""


class RenderCancelled(RuntimeError):
    """Raised when a batch or render is cancelled before output is committed."""
