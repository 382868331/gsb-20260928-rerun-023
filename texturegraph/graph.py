from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .errors import CycleError, GraphValidationError


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Connection:
    source_node: str
    source_port: str
    target_node: str
    target_port: str


@dataclass
class Graph:
    width: int = 256
    height: int = 256
    nodes: dict[str, Node] = field(default_factory=dict)
    connections: list[Connection] = field(default_factory=list)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        params: dict[str, Any] | None = None,
    ) -> Node:
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        node = Node(node_id, node_type, dict(params or {}))
        self.nodes[node_id] = node
        return node

    def connect(
        self,
        source_node: str,
        target_node: str,
        target_port: str,
        source_port: str = "value",
    ) -> None:
        self.connections.append(
            Connection(source_node, source_port, target_node, target_port)
        )

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        node = self.nodes[node_id]
        node.params[name] = value

    def inputs_for(self, node_id: str) -> dict[str, Connection]:
        result: dict[str, Connection] = {}
        for connection in self.connections:
            if connection.target_node != node_id:
                continue
            if connection.target_port in result:
                raise GraphValidationError(
                    f"input '{connection.target_port}' on '{node_id}' has more than one connection"
                )
            result[connection.target_port] = connection
        return result

    def topological_order(self, output_node: str | None = None) -> list[str]:
        self.validate_references()
        if output_node is not None and output_node not in self.nodes:
            raise GraphValidationError(f"unknown output node: {output_node}")

        reachable = self._reachable_from(output_node)
        indegree = {node_id: 0 for node_id in reachable}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in reachable}
        for connection in self.connections:
            source = connection.source_node
            target = connection.target_node
            if source not in reachable or target not in reachable:
                continue
            outgoing[source].append(target)
            indegree[target] += 1

        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for target in sorted(outgoing[node_id]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
            ready.sort()

        if len(order) != len(reachable):
            cycle = self._find_cycle(reachable)
            raise CycleError(f"cycle detected: {' -> '.join(cycle)}")
        return order

    def validate_references(self) -> None:
        from .nodes import NODE_TYPES

        for node in self.nodes.values():
            if node.type not in NODE_TYPES:
                raise GraphValidationError(f"unknown node type: {node.type}")
        for connection in self.connections:
            if connection.source_node not in self.nodes:
                raise GraphValidationError(f"unknown source node: {connection.source_node}")
            if connection.target_node not in self.nodes:
                raise GraphValidationError(f"unknown target node: {connection.target_node}")

    def _reachable_from(self, output_node: str | None) -> set[str]:
        if output_node is None:
            return set(self.nodes)
        reachable = {output_node}
        pending = [output_node]
        while pending:
            node_id = pending.pop()
            for connection in self.inputs_for(node_id).values():
                if connection.source_node not in reachable:
                    reachable.add(connection.source_node)
                    pending.append(connection.source_node)
        return reachable

    def _find_cycle(self, scope: set[str]) -> list[str]:
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(node_id: str) -> list[str] | None:
            state[node_id] = 1
            stack.append(node_id)
            for connection in self.inputs_for(node_id).values():
                source = connection.source_node
                if source not in scope:
                    continue
                if state.get(source) == 1:
                    start = stack.index(source)
                    return stack[start:] + [source]
                if state.get(source) != 2:
                    found = visit(source)
                    if found:
                        return found
            stack.pop()
            state[node_id] = 2
            return None

        for node_id in sorted(scope):
            found = visit(node_id)
            if found:
                return found
        return list(scope)
