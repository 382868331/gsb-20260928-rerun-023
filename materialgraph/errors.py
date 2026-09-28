class MaterialGraphError(Exception):
    pass


class CycleError(MaterialGraphError):
    def __init__(self, cycle):
        self.cycle = list(cycle)
        super().__init__("Cycle detected: " + " -> ".join(self.cycle))


class PortError(MaterialGraphError):
    pass


class NodeError(MaterialGraphError):
    pass


class RenderCancelled(MaterialGraphError):
    pass
