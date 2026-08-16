"""Foster network circuit construction (Prompt 04B).

Builds a ``CircuitGraph`` from a topology candidate, sign pattern, and
Foster components.  Uses node aliasing for ZERO_IMPEDANCE branches
(no 0Ω elements).

Orientation invariants are enforced by explicit ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.domain.topology import LOrientation
from foster_eom.foster.foster_form import FosterComponents
from foster_eom.foster.schmidt import BranchRealization
from foster_eom.foster.sign_search import SignPattern
from foster_eom.foster.topology_enum import TopologyCandidate
from foster_eom.models.base import OnePortModel

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltFosterCircuit:
    """A fully constructed Foster L-network circuit."""

    graph: CircuitGraph
    port: Port
    eom_element_id: str
    branch1_element_ids: tuple[str, ...]
    branch2_element_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_foster_chain(
    graph: CircuitGraph,
    components: FosterComponents,
    prefix: str,
    node_pos_id: str,
    node_neg_id: str,
    has_c0: bool,
    has_linf: bool,
) -> list[str]:
    """Add a Foster-I series chain of parallel LC cells.

    Chain: pos → [C0] → (L1||C1) → ... → (LM||CM) → [L_inf] → neg

    Returns list of element IDs created.
    """
    element_ids: list[str] = []
    current_node = node_pos_id
    chain_idx = 0

    # C0 endpoint (series capacitor at beginning)
    if has_c0 and components.c0_f is not None and components.c0_f > 0:
        next_node_id = f"{prefix}_n{chain_idx}"
        graph.add_node(Node(id=next_node_id))
        eid = f"{prefix}_C0"
        graph.add_element(
            Element(
                id=eid,
                kind=ElementKind.CAPACITOR,
                node_pos=current_node,
                node_neg=next_node_id,
                value=components.c0_f,
                symbolic_role=f"{prefix}_endpoint_C0",
            )
        )
        element_ids.append(eid)
        current_node = next_node_id
        chain_idx += 1

    # Foster cells (parallel L||C)
    for cell_i, cell in enumerate(components.cells):
        if cell_i < len(components.cells) - 1 or (has_linf and components.l_inf_h is not None):
            next_node_id = f"{prefix}_n{chain_idx}"
            graph.add_node(Node(id=next_node_id))
        else:
            next_node_id = node_neg_id

        # Inductor
        eid_l = f"{prefix}_L{cell_i + 1}"
        graph.add_element(
            Element(
                id=eid_l,
                kind=ElementKind.INDUCTOR,
                node_pos=current_node,
                node_neg=next_node_id,
                value=cell.l_h,
                symbolic_role=f"{prefix}_cell{cell_i + 1}_L",
            )
        )
        element_ids.append(eid_l)

        # Capacitor (parallel with inductor — same nodes)
        eid_c = f"{prefix}_C{cell_i + 1}"
        graph.add_element(
            Element(
                id=eid_c,
                kind=ElementKind.CAPACITOR,
                node_pos=current_node,
                node_neg=next_node_id,
                value=cell.c_f,
                symbolic_role=f"{prefix}_cell{cell_i + 1}_C",
            )
        )
        element_ids.append(eid_c)

        current_node = next_node_id
        chain_idx += 1

    # L_inf endpoint (series inductor at end)
    if has_linf and components.l_inf_h is not None and components.l_inf_h > 0:
        eid = f"{prefix}_Linf"
        graph.add_element(
            Element(
                id=eid,
                kind=ElementKind.INDUCTOR,
                node_pos=current_node,
                node_neg=node_neg_id,
                value=components.l_inf_h,
                symbolic_role=f"{prefix}_endpoint_Linf",
            )
        )
        element_ids.append(eid)

    return element_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_foster_circuit(
    topology: TopologyCandidate,
    sign_pattern: SignPattern,
    branch1_components: FosterComponents | None,
    branch2_components: FosterComponents | None,
    eom_model: OnePortModel,
) -> BuiltFosterCircuit:
    """Build a Foster L-network circuit graph.

    Parameters
    ----------
    topology : TopologyCandidate
        Topology candidate with cell counts and endpoint flags.
    sign_pattern : SignPattern
        Sign pattern with branch realizations and orientation.
    branch1_components : FosterComponents | None
        Foster components for branch 1 (shunt). None for trivial branches.
    branch2_components : FosterComponents | None
        Foster components for branch 2 (series). None for trivial branches.
    eom_model : OnePortModel
        EOM load model.

    Returns
    -------
    BuiltFosterCircuit

    Raises
    ------
    ValueError
        On orientation mismatch, invalid component/topology combinations,
        or structural inconsistencies.
    """
    # Orientation invariant — explicit production check
    if topology.orientation != sign_pattern.orientation:
        raise ValueError(
            f"Orientation mismatch: topology={topology.orientation}, "
            f"sign_pattern={sign_pattern.orientation}"
        )

    # Defensive validation of branch components vs topology
    _validate_branch_inputs(
        topology,
        sign_pattern,
        branch1_components,
        branch2_components,
    )

    b1_real = sign_pattern.branch1_realization
    b2_real = sign_pattern.branch2_realization
    orientation = topology.orientation

    # Build the circuit based on orientation
    if orientation == LOrientation.SCHMIDT_SHUNT_THEN_SERIES:
        return _build_standard(
            topology,
            sign_pattern,
            branch1_components,
            branch2_components,
            eom_model,
            b1_real,
            b2_real,
        )
    elif orientation == LOrientation.ALTERNATE_L_ORIENTATION:
        return _build_dual(
            topology,
            sign_pattern,
            branch1_components,
            branch2_components,
            eom_model,
            b1_real,
            b2_real,
        )
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")


def _validate_branch_inputs(
    topology: TopologyCandidate,
    sign_pattern: SignPattern,
    branch1_components: FosterComponents | None,
    branch2_components: FosterComponents | None,
) -> None:
    """Validate that branch components agree with topology."""
    b1_real = sign_pattern.branch1_realization
    b2_real = sign_pattern.branch2_realization

    # Trivial branches must have no components
    if b1_real != BranchRealization.FINITE_FOSTER and branch1_components is not None:
        raise ValueError(f"Branch 1 is {b1_real.value} but received non-None components")
    if b2_real != BranchRealization.FINITE_FOSTER and branch2_components is not None:
        raise ValueError(f"Branch 2 is {b2_real.value} but received non-None components")

    # FINITE_FOSTER branches must have matching components
    if b1_real == BranchRealization.FINITE_FOSTER:
        if branch1_components is None:
            raise ValueError("Branch 1 is FINITE_FOSTER but components are None")
        if len(branch1_components.cells) != topology.branch1_cells:
            raise ValueError(
                f"Branch 1 cell count mismatch: topology={topology.branch1_cells}, "
                f"components={len(branch1_components.cells)}"
            )
        # C0/L_inf agreement
        if topology.branch1_has_c0 and branch1_components.c0_f is None:
            raise ValueError("Topology expects branch1 C0 but components.c0_f is None")
        if not topology.branch1_has_c0 and branch1_components.c0_f is not None:
            raise ValueError("Topology does not allow branch1 C0 but components.c0_f is set")
        if topology.branch1_has_linf and branch1_components.l_inf_h is None:
            raise ValueError("Topology expects branch1 L_inf but components.l_inf_h is None")
        if not topology.branch1_has_linf and branch1_components.l_inf_h is not None:
            raise ValueError("Topology does not allow branch1 L_inf but components.l_inf_h is set")

    if b2_real == BranchRealization.FINITE_FOSTER:
        if branch2_components is None:
            raise ValueError("Branch 2 is FINITE_FOSTER but components are None")
        if len(branch2_components.cells) != topology.branch2_cells:
            raise ValueError(
                f"Branch 2 cell count mismatch: topology={topology.branch2_cells}, "
                f"components={len(branch2_components.cells)}"
            )
        if topology.branch2_has_c0 and branch2_components.c0_f is None:
            raise ValueError("Topology expects branch2 C0 but components.c0_f is None")
        if not topology.branch2_has_c0 and branch2_components.c0_f is not None:
            raise ValueError("Topology does not allow branch2 C0 but components.c0_f is set")
        if topology.branch2_has_linf and branch2_components.l_inf_h is None:
            raise ValueError("Topology expects branch2 L_inf but components.l_inf_h is None")
        if not topology.branch2_has_linf and branch2_components.l_inf_h is not None:
            raise ValueError("Topology does not allow branch2 L_inf but components.l_inf_h is set")


def _build_standard(
    topology: TopologyCandidate,
    sign_pattern: SignPattern,
    branch1_components: FosterComponents | None,
    branch2_components: FosterComponents | None,
    eom_model: OnePortModel,
    b1_real: BranchRealization,
    b2_real: BranchRealization,
) -> BuiltFosterCircuit:
    """Standard: source → [in] → shunt(b1) → [gnd], [in] → series(b2) → [eom] → Z_EOM → [gnd]."""
    gnd_id = "gnd"
    eom_id = "eom"

    # Determine node aliasing for ZERO_IMPEDANCE series branch
    if b2_real == BranchRealization.ZERO_IMPEDANCE:
        # Series wire: eom_pos aliased to in
        in_id = "in"
        eom_pos_id = in_id  # aliased
    else:
        in_id = "in"
        eom_pos_id = "eom_pos"

    # Build graph
    port = Port(node_pos=in_id, node_neg=gnd_id)
    graph = CircuitGraph(
        ground_node_id=gnd_id,
        input_port=port,
        eom_element_id=eom_id,
    )

    graph.add_node(Node(id=gnd_id, is_ground=True))
    graph.add_node(Node(id=in_id))
    if eom_pos_id != in_id:
        graph.add_node(Node(id=eom_pos_id))

    # EOM element
    graph.add_element(
        Element(
            id=eom_id,
            kind=ElementKind.ONE_PORT_MODEL,
            node_pos=eom_pos_id,
            node_neg=gnd_id,
            model=eom_model,
            symbolic_role="eom",
        )
    )

    # Branch 1 (shunt): from [in] to [gnd]
    b1_ids: list[str] = []
    if b1_real == BranchRealization.FINITE_FOSTER and branch1_components is not None:
        b1_ids = _add_foster_chain(
            graph,
            branch1_components,
            "b1",
            node_pos_id=in_id,
            node_neg_id=gnd_id,
            has_c0=topology.branch1_has_c0,
            has_linf=topology.branch1_has_linf,
        )
    # OPEN_OMITTED: no elements

    # Branch 2 (series): from [in] to [eom_pos]
    b2_ids: list[str] = []
    if b2_real == BranchRealization.FINITE_FOSTER and branch2_components is not None:
        b2_ids = _add_foster_chain(
            graph,
            branch2_components,
            "b2",
            node_pos_id=in_id,
            node_neg_id=eom_pos_id,
            has_c0=topology.branch2_has_c0,
            has_linf=topology.branch2_has_linf,
        )
    # ZERO_IMPEDANCE: node aliasing already handled, no elements

    return BuiltFosterCircuit(
        graph=graph,
        port=port,
        eom_element_id=eom_id,
        branch1_element_ids=tuple(b1_ids),
        branch2_element_ids=tuple(b2_ids),
    )


def _build_dual(
    topology: TopologyCandidate,
    sign_pattern: SignPattern,
    branch1_components: FosterComponents | None,
    branch2_components: FosterComponents | None,
    eom_model: OnePortModel,
    b1_real: BranchRealization,
    b2_real: BranchRealization,
) -> BuiltFosterCircuit:
    """Dual: source → [in] → series(b2) → [mid] → Z_EOM → [gnd], [mid] → shunt(b1) → [gnd]."""
    gnd_id = "gnd"
    eom_id = "eom"
    in_id = "in"

    mid_id = in_id if b2_real == BranchRealization.ZERO_IMPEDANCE else "mid"

    # Build graph
    port = Port(node_pos=in_id, node_neg=gnd_id)
    graph = CircuitGraph(
        ground_node_id=gnd_id,
        input_port=port,
        eom_element_id=eom_id,
    )

    graph.add_node(Node(id=gnd_id, is_ground=True))
    graph.add_node(Node(id=in_id))
    if mid_id != in_id:
        graph.add_node(Node(id=mid_id))

    # EOM element: [mid] → [gnd]
    graph.add_element(
        Element(
            id=eom_id,
            kind=ElementKind.ONE_PORT_MODEL,
            node_pos=mid_id,
            node_neg=gnd_id,
            model=eom_model,
            symbolic_role="eom",
        )
    )

    # Branch 2 (series): [in] → [mid]
    b2_ids: list[str] = []
    if b2_real == BranchRealization.FINITE_FOSTER and branch2_components is not None:
        b2_ids = _add_foster_chain(
            graph,
            branch2_components,
            "b2",
            node_pos_id=in_id,
            node_neg_id=mid_id,
            has_c0=topology.branch2_has_c0,
            has_linf=topology.branch2_has_linf,
        )
    # ZERO_IMPEDANCE: node aliasing handled

    # Branch 1 (shunt): [mid] → [gnd]
    b1_ids: list[str] = []
    if b1_real == BranchRealization.FINITE_FOSTER and branch1_components is not None:
        b1_ids = _add_foster_chain(
            graph,
            branch1_components,
            "b1",
            node_pos_id=mid_id,
            node_neg_id=gnd_id,
            has_c0=topology.branch1_has_c0,
            has_linf=topology.branch1_has_linf,
        )
    # OPEN_OMITTED: no elements

    return BuiltFosterCircuit(
        graph=graph,
        port=port,
        eom_element_id=eom_id,
        branch1_element_ids=tuple(b1_ids),
        branch2_element_ids=tuple(b2_ids),
    )
