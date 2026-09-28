from __future__ import annotations

from copy import deepcopy

from .errors import CycleError, NodeError, PortError
from .nodes import NODE_TYPES, Node


class Graph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.connections: dict[tuple[str, str], tuple[str, str]] = {}
        self._cache: dict[str, object] = {}
        self.compute_counts: dict[str, int] = {}

    def add_node(self, type_name: str, node_id: str, params: dict | None = None):
        if not node_id:
            raise NodeError("Node id must not be empty")
        if node_id in self.nodes:
            raise NodeError(f"Duplicate node id: {node_id}")
        if type_name not in NODE_TYPES:
            raise NodeError(f"Unknown node type: {type_name}")
        node = NODE_TYPES[type_name](id=node_id, params=deepcopy(params or {}))
        self.nodes[node_id] = node
        return node

    def connect(self, source_id: str, target_id: str, target_port: str, source_port: str = "output"):
        self._require_node(source_id)
        target = self._require_node(target_id)
        if target_port not in target.input_ports:
            raise PortError(f"Node {target_id} has no input port {target_port}")
        if source_port != "output":
            raise PortError(f"Node {source_id} has no output port {source_port}")
        source = self.nodes[source_id]
        expected = target.input_ports[target_port]
        if source.output_type != expected:
            raise PortError(
                f"Cannot connect {source.type_name} to {target.type_name}.{target_port}: "
                f"{source.output_type} != {expected}"
            )
        key = (target_id, target_port)
        self.connections[key] = (source_id, source_port)
        self.invalidate_downstream(target_id)

    def disconnect(self, target_id: str, target_port: str):
        key = (target_id, target_port)
        if key in self.connections:
            del self.connections[key]
            self.invalidate_downstream(target_id)

    def set_param(self, node_id: str, name: str, value):
        node = self._require_node(node_id)
        node.params[name] = deepcopy(value)
        self.invalidate_downstream(node_id)

    def update_params(self, node_id: str, params: dict):
        node = self._require_node(node_id)
        node.params = deepcopy(params)
        self.invalidate_downstream(node_id)

    def _require_node(self, node_id):
        try:
            return self.nodes[node_id]
        except KeyError:
            raise NodeError(f"Unknown node id: {node_id}") from None

    def invalidate_downstream(self, node_id: str):
        affected = self.descendants(node_id)
        for item_id in affected:
            self._cache.pop(item_id, None)

    def descendants(self, node_id: str):
        reverse = {node_id: [] for node_id in self.nodes}
        for (target_id, _), (source_id, _) in self.connections.items():
            reverse.setdefault(source_id, []).append(target_id)
            reverse.setdefault(target_id, [])
        seen = set()
        pending = [node_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(reverse.get(current, []))
        return seen

    def validate(self):
        for node_id, node in self.nodes.items():
            for port_name in node.input_ports:
                key = (node_id, port_name)
                if key not in self.connections:
                    raise PortError(f"Missing connection for {node_id}.{port_name}")
                source_id, _ = self.connections[key]
                if source_id not in self.nodes:
                    raise PortError(f"Connection to unknown node: {source_id}")
        self._find_cycle()

    def _find_cycle(self):
        graph = {node_id: [] for node_id in self.nodes}
        for (target_id, _), (source_id, _) in self.connections.items():
            graph[source_id].append(target_id)
        state = {}
        stack = []

        def visit(node_id):
            state[node_id] = "active"
            stack.append(node_id)
            for target_id in graph[node_id]:
                if state.get(target_id) == "active":
                    start = stack.index(target_id)
                    raise CycleError(stack[start:] + [target_id])
                if state.get(target_id) is None:
                    visit(target_id)
            stack.pop()
            state[node_id] = "done"

        for node_id in graph:
            if state.get(node_id) is None:
                visit(node_id)

    def execute(self, output_ids=None, should_cancel=None):
        self.validate()
        if should_cancel is not None and should_cancel():
            from .errors import RenderCancelled

            raise RenderCancelled("Render cancelled before computation")
        targets = self._resolve_outputs(output_ids)
        evaluator = _Evaluator(self, should_cancel)
        results = {output_id: evaluator.value(output_id) for output_id in targets}
        return results

    def _resolve_outputs(self, output_ids):
        explicit = list(output_ids or [])
        if explicit:
            for node_id in explicit:
                self._require_node(node_id)
            return explicit
        return [node_id for node_id, node in self.nodes.items() if node.type_name == "output"]


class _Evaluator:
    def __init__(self, graph: Graph, should_cancel=None):
        self.graph = graph
        self.should_cancel = should_cancel
        self.local_cache = {}
        self.compute_counts = graph.compute_counts

    def value(self, node_id: str):
        if node_id in self.graph._cache:
            return self.graph._cache[node_id]
        if node_id in self.local_cache:
            return self.local_cache[node_id]

        node = self.graph.nodes[node_id]
        inputs = {}
        for port_name in node.input_ports:
            source_id, _ = self.graph.connections[(node_id, port_name)]
            inputs[port_name] = self.value(source_id)

        if self.should_cancel is not None and self.should_cancel():
            from .errors import RenderCancelled

            raise RenderCancelled("Render cancelled")

        result = node.compute(inputs, self)
        self.local_cache[node_id] = result
        self.graph._cache[node_id] = result
        self.compute_counts[node_id] = self.compute_counts.get(node_id, 0) + 1
        return result
