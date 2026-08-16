"""Circuit graph / netlist representation (spec §11.1).

The ``CircuitGraph`` represents a **passive** lumped electrical network.
The source excitation is external (applied by the solver via ``SourceSpec``).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A circuit node.

    Parameters
    ----------
    id : str
        Unique stable identifier (string, not integer).
    label : str
        Human-readable label.  Defaults to *id*.
    is_ground : bool
        Whether this node is the reference (ground) node.
    """

    id: str
    label: str = ""
    is_ground: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Node id must be a non-empty string.")
        if not self.label:
            object.__setattr__(self, "label", self.id)


# ---------------------------------------------------------------------------
# Element kind
# ---------------------------------------------------------------------------


class ElementKind(enum.Enum):
    """Supported element types for the passive circuit graph.

    ``NORTON_SOURCE`` is intentionally absent — the source is external,
    applied by the solver via ``SourceSpec``.
    """

    RESISTOR = "resistor"
    INDUCTOR = "inductor"
    CAPACITOR = "capacitor"
    ONE_PORT_MODEL = "one_port_model"


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Element:
    """A two-terminal circuit element.

    Parameters
    ----------
    id : str
        Unique stable identifier.
    kind : ElementKind
        Element type.
    node_pos : str
        Node ID of the positive terminal.
    node_neg : str
        Node ID of the negative terminal.
    value : float | None
        R (Ω), L (H), or C (F) in SI for primitives.  ``None`` for model
        elements.
    model : OnePortModel | None
        Runtime model for ``ONE_PORT_MODEL`` elements.
    symbolic_role : str | None
        Symbolic annotation, e.g. ``"eom"``, ``"foster_series_L1"``.
    """

    id: str
    kind: ElementKind
    node_pos: str
    node_neg: str
    value: float | None = None
    model: OnePortModel | None = None
    symbolic_role: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Element id must be a non-empty string.")
        if self.node_pos == self.node_neg:
            raise ValueError(
                f"Element '{self.id}' has a self-loop (node_pos == node_neg == '{self.node_pos}')."
            )
        if self.kind in (
            ElementKind.RESISTOR,
            ElementKind.INDUCTOR,
            ElementKind.CAPACITOR,
        ):
            if self.value is None:
                raise ValueError(
                    f"Primitive element '{self.id}' ({self.kind.value}) requires a value."
                )
            if self.value <= 0.0:
                raise ValueError(
                    f"Primitive element '{self.id}' ({self.kind.value}) "
                    f"value must be strictly positive, got {self.value}."
                )
        if self.kind == ElementKind.ONE_PORT_MODEL and self.model is None:
            raise ValueError(f"Element '{self.id}' is ONE_PORT_MODEL but model is None.")


# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """A two-terminal port definition.

    Parameters
    ----------
    node_pos : str
        Node ID of the positive terminal.
    node_neg : str
        Node ID of the negative terminal.
    """

    node_pos: str
    node_neg: str


# ---------------------------------------------------------------------------
# Circuit graph
# ---------------------------------------------------------------------------

# Sentinel for ground in the integer index space.
GROUND: int = -1


class CircuitGraph:
    """Passive lumped circuit netlist.

    The graph contains only passive elements.  The source excitation is
    external and is applied by the solver.

    Attributes
    ----------
    nodes : dict[str, Node]
        All nodes by ID.
    elements : dict[str, Element]
        All elements by ID.
    ground_node_id : str
        The ID of the ground (reference) node.
    input_port : Port
        Two-terminal input port where the source drives the network.
    eom_element_id : str | None
        Element ID of the EOM load.  ``V_EOM = V_pos - V_neg`` of that
        element.
    """

    def __init__(
        self,
        ground_node_id: str,
        input_port: Port,
        eom_element_id: str | None = None,
    ) -> None:
        self.nodes: dict[str, Node] = {}
        self.elements: dict[str, Element] = {}
        self.ground_node_id = ground_node_id
        self.input_port = input_port
        self.eom_element_id = eom_element_id

    # -- builder API --------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Raises
        ------
        ValueError
            If a node with the same ID already exists.
        """
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node ID: '{node.id}'.")
        self.nodes[node.id] = node

    def add_element(self, element: Element) -> None:
        """Add an element to the graph.

        Raises
        ------
        ValueError
            If an element with the same ID already exists.
        """
        if element.id in self.elements:
            raise ValueError(f"Duplicate element ID: '{element.id}'.")
        self.elements[element.id] = element

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Validate the graph for consistency.

        Raises
        ------
        ValueError
            On any structural inconsistency.
        """
        # Ground node
        if self.ground_node_id not in self.nodes:
            raise ValueError(f"Ground node '{self.ground_node_id}' not found in nodes.")
        gnd = self.nodes[self.ground_node_id]
        if not gnd.is_ground:
            raise ValueError(
                f"Node '{self.ground_node_id}' is designated as ground "
                f"but does not have is_ground=True."
            )

        # Input port nodes
        if self.input_port.node_pos not in self.nodes:
            raise ValueError(
                f"Input port node_pos '{self.input_port.node_pos}' not found in nodes."
            )
        if self.input_port.node_neg not in self.nodes:
            raise ValueError(
                f"Input port node_neg '{self.input_port.node_neg}' not found in nodes."
            )

        # EOM element
        if self.eom_element_id is not None and self.eom_element_id not in self.elements:
            raise ValueError(f"EOM element '{self.eom_element_id}' not found in elements.")

        # Element node references
        for elem in self.elements.values():
            if elem.node_pos not in self.nodes:
                raise ValueError(
                    f"Element '{elem.id}' references unknown node_pos '{elem.node_pos}'."
                )
            if elem.node_neg not in self.nodes:
                raise ValueError(
                    f"Element '{elem.id}' references unknown node_neg '{elem.node_neg}'."
                )

    # -- index mapping ------------------------------------------------------

    def node_indices(self) -> dict[str, int]:
        """Return a deterministic mapping from non-ground node IDs to
        integer matrix indices.

        Ground is excluded.  The mapping is sorted alphabetically by node
        ID to ensure determinism regardless of insertion order.

        Returns
        -------
        dict[str, int]
            ``{node_id: matrix_index}`` for all non-ground nodes.
        """
        non_ground = sorted(nid for nid in self.nodes if nid != self.ground_node_id)
        return {nid: idx for idx, nid in enumerate(non_ground)}

    def _resolve_index(self, node_id: str, node_map: dict[str, int]) -> int:
        """Resolve a node ID to its matrix index, returning ``GROUND``
        for the ground node."""
        if node_id == self.ground_node_id:
            return GROUND
        return node_map[node_id]
