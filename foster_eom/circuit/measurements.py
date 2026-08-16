"""Circuit solution and per-element measurements (spec §14.6).

All quantities use RMS phasor convention:

    S = V · conj(I)      (complex power, no factor of 1/2)
    P = Re(S)            (real / active power)
    Q = Im(S)            (reactive power)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind
from foster_eom.circuit.mna import SolveDiagnostics
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus
from foster_eom.units import s11_db_from_gamma, z_to_gamma

# ---------------------------------------------------------------------------
# Per-element measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementMeasurement:
    """Per-element electrical measurement at a single frequency.

    All voltages and currents are RMS phasors.  Current is defined as
    flowing from ``node_pos`` toward ``node_neg`` through the element.

    Attributes
    ----------
    element_id : str
    voltage : complex
        ``V = V[node_pos] - V[node_neg]``
    current : complex
        Through the element, positive from node_pos toward node_neg.
    complex_power : complex
        ``S = V · conj(I)``
    real_power_w : float
        ``Re(S)``
    reactive_power_var : float
        ``Im(S)``
    """

    element_id: str
    voltage: complex
    current: complex
    complex_power: complex
    real_power_w: float
    reactive_power_var: float


# ---------------------------------------------------------------------------
# Full circuit solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitSolution:
    """Complete solution at a single frequency.

    Attributes whose value is ``None`` indicate a failed solve.
    """

    f_hz: float
    status: CircuitSolveStatus
    diagnostics: SolveDiagnostics

    # Node voltages (RMS phasors, by node ID; ground = 0)
    node_voltages: dict[str, complex] | None = None

    # Per-element measurements (by element ID)
    element_measurements: dict[str, ElementMeasurement] | None = None

    # Input port
    v_port: complex | None = None
    i_port: complex | None = None  # from passive branches
    i_source_droop: complex | None = None  # (V_th - V_port) / Z_s

    # Derived
    z_in: complex | None = None
    gamma: complex | None = None
    s11_db: float | None = None

    # EOM
    v_eom: complex | None = None
    i_eom: complex | None = None

    # Power (RMS convention: S = V · I*, no 1/2)
    s_source_delivered: complex | None = None
    p_source_delivered_w: float | None = None
    s_dissipated_total: complex | None = None
    p_dissipated_total_w: float | None = None
    power_balance_residual: complex | None = None
    power_balance_ok: bool = False


# ---------------------------------------------------------------------------
# Measurement extraction
# ---------------------------------------------------------------------------


def _element_admittance(elem: Element, f_hz: float) -> complex:
    """Compute the two-terminal admittance of an element at a frequency."""

    if elem.kind == ElementKind.RESISTOR:
        assert elem.value is not None
        return 1.0 / elem.value

    if elem.kind == ElementKind.INDUCTOR:
        assert elem.value is not None
        omega = 2.0 * np.pi * f_hz
        z_l = np.complex128(1j * omega * elem.value)
        with np.errstate(divide="ignore", invalid="ignore"):
            return complex(np.complex128(1.0) / z_l)

    if elem.kind == ElementKind.CAPACITOR:
        assert elem.value is not None
        omega = 2.0 * np.pi * f_hz
        return complex(1j * omega * elem.value)

    if elem.kind == ElementKind.ONE_PORT_MODEL:
        assert elem.model is not None
        y_val = elem.model.y(f_hz)
        return complex(y_val)  # type: ignore[arg-type]

    raise ValueError(f"Unsupported element kind: {elem.kind}")


def compute_measurements(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    V: np.ndarray,
    node_map: dict[str, int],
    f_hz: float,
    diagnostics: SolveDiagnostics,
    atol: float = 1e-12,
    rtol: float = 1e-6,
) -> CircuitSolution:
    """Extract all measurements from a solved node-voltage vector.

    Parameters
    ----------
    graph : CircuitGraph
    source_spec : SourceSpec
    V : np.ndarray
        Node voltages from MNA solve.
    node_map : dict[str, int]
        Node ID → matrix index mapping.
    f_hz : float
    diagnostics : SolveDiagnostics
    atol, rtol : float
        Power balance tolerances.

    Returns
    -------
    CircuitSolution
    """

    # -- Node voltages (ground = 0) ----------------------------------------
    node_voltages: dict[str, complex] = {}
    for nid in graph.nodes:
        if nid == graph.ground_node_id:
            node_voltages[nid] = 0.0 + 0.0j
        else:
            node_voltages[nid] = complex(V[node_map[nid]])

    # -- Input port --------------------------------------------------------
    v_pos = node_voltages[graph.input_port.node_pos]
    v_neg = node_voltages[graph.input_port.node_neg]
    v_port = v_pos - v_neg

    # I_port from passive branches at the input port:
    # Sum currents flowing INTO the network from node_pos through all
    # passive elements that touch input_port.node_pos.
    i_port = complex(0.0)
    for elem in graph.elements.values():
        y_elem = _element_admittance(elem, f_hz)
        v_ep = node_voltages[elem.node_pos]
        v_en = node_voltages[elem.node_neg]
        v_elem = v_ep - v_en
        i_elem = y_elem * v_elem  # current from node_pos toward node_neg

        # If element's node_pos is the port node_pos, current flows
        # INTO the element (out of the port) → contributes +i_elem to i_port
        if elem.node_pos == graph.input_port.node_pos:
            i_port += i_elem
        # If element's node_neg is the port node_pos, current flows
        # OUT of the element into the port → contributes -i_elem
        elif elem.node_neg == graph.input_port.node_pos:
            i_port -= i_elem

    # Source-droop current (must agree with i_port)
    z_s = source_spec.z_source
    i_source_droop = (source_spec.vth_phasor - v_port) / z_s

    # Z_in, Gamma, S11
    with np.errstate(divide="ignore", invalid="ignore"):
        z_in = v_port / i_port if abs(i_port) > 0 else complex(np.inf)
    gamma = z_to_gamma(z_in, source_spec.z_ref_ohm)
    s11 = s11_db_from_gamma(abs(gamma))

    # -- EOM measurements --------------------------------------------------
    v_eom: complex | None = None
    i_eom: complex | None = None
    if graph.eom_element_id is not None and graph.eom_element_id in graph.elements:
        eom_elem = graph.elements[graph.eom_element_id]
        v_eom = node_voltages[eom_elem.node_pos] - node_voltages[eom_elem.node_neg]
        y_eom = _element_admittance(eom_elem, f_hz)
        i_eom = y_eom * v_eom

    # -- Per-element measurements ------------------------------------------
    element_measurements: dict[str, ElementMeasurement] = {}
    s_total = complex(0.0)
    for elem in graph.elements.values():
        v_ep = node_voltages[elem.node_pos]
        v_en = node_voltages[elem.node_neg]
        v_elem = v_ep - v_en
        y_elem = _element_admittance(elem, f_hz)
        i_elem = y_elem * v_elem
        s_elem = v_elem * np.conj(i_elem)
        element_measurements[elem.id] = ElementMeasurement(
            element_id=elem.id,
            voltage=v_elem,
            current=i_elem,
            complex_power=s_elem,
            real_power_w=float(np.real(s_elem)),
            reactive_power_var=float(np.imag(s_elem)),
        )
        s_total += s_elem

    # -- Power balance -----------------------------------------------------
    s_delivered = v_port * np.conj(i_port)
    p_delivered = float(np.real(s_delivered))
    p_total = float(np.real(s_total))
    residual = s_delivered - s_total
    balance_ok = abs(residual) < atol + rtol * abs(s_delivered)

    return CircuitSolution(
        f_hz=f_hz,
        status=CircuitSolveStatus.OK,
        diagnostics=diagnostics,
        node_voltages=node_voltages,
        element_measurements=element_measurements,
        v_port=v_port,
        i_port=i_port,
        i_source_droop=i_source_droop,
        z_in=z_in,
        gamma=gamma,
        s11_db=s11,
        v_eom=v_eom,
        i_eom=i_eom,
        s_source_delivered=s_delivered,
        p_source_delivered_w=p_delivered,
        s_dissipated_total=s_total,
        p_dissipated_total_w=p_total,
        power_balance_residual=residual,
        power_balance_ok=balance_ok,
    )
