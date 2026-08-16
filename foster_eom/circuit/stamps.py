"""MNA stamping functions (spec §14.2).

Pure functions that modify a mutable admittance matrix ``Y`` and current
vector ``I`` in-place.  The API is designed so multiport Y-matrix stamps
can be added later without rewriting.
"""

from __future__ import annotations

import numpy as np

from foster_eom.circuit.graph import GROUND, Element, ElementKind
from foster_eom.domain.source import SourceSpec


def stamp_admittance(
    Y: np.ndarray,
    a: int,
    b: int,
    y: complex,
) -> None:
    """Stamp a two-terminal admittance ``y`` between nodes ``a`` and ``b``.

    Implements the canonical 2x2 stamp from spec S14.2::

        Y[a,a] += y;   Y[b,b] += y
        Y[a,b] -= y;   Y[b,a] -= y

    If ``a`` or ``b`` is ``GROUND`` (sentinel = -1), the corresponding
    row/column operations are skipped.

    Parameters
    ----------
    Y : np.ndarray
        Complex admittance matrix (modified in-place).
    a, b : int
        Matrix indices of the two nodes, or ``GROUND``.
    y : complex
        Admittance value.
    """
    if a != GROUND:
        Y[a, a] += y
        if b != GROUND:
            Y[a, b] -= y
    if b != GROUND:
        Y[b, b] += y
        if a != GROUND:
            Y[b, a] -= y


def stamp_element(
    Y: np.ndarray,
    element: Element,
    node_map: dict[str, int],
    ground_node_id: str,
    f_hz: float,
) -> None:
    """Stamp a single element into the admittance matrix.

    Parameters
    ----------
    Y : np.ndarray
        Complex admittance matrix (modified in-place).
    element : Element
        The circuit element to stamp.
    node_map : dict[str, int]
        Mapping from non-ground node IDs to matrix indices.
    ground_node_id : str
        ID of the ground node.
    f_hz : float
        Frequency in Hz.  ``ω = 2π·f_hz`` is computed only where needed.
    """
    a = GROUND if element.node_pos == ground_node_id else node_map[element.node_pos]
    b = GROUND if element.node_neg == ground_node_id else node_map[element.node_neg]

    y: complex

    if element.kind == ElementKind.RESISTOR:
        assert element.value is not None
        y = 1.0 / element.value
        stamp_admittance(Y, a, b, y)

    elif element.kind == ElementKind.INDUCTOR:
        assert element.value is not None
        omega = 2.0 * np.pi * f_hz
        # At f_hz=0, jωL = 0 -> y = inf.  Use numpy complex128 division
        # which produces inf instead of raising ZeroDivisionError.
        # The solver's nonfinite pre-screen catches this.
        z_l = np.complex128(1j * omega * element.value)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = complex(np.complex128(1.0) / z_l)
        stamp_admittance(Y, a, b, y)

    elif element.kind == ElementKind.CAPACITOR:
        assert element.value is not None
        omega = 2.0 * np.pi * f_hz
        y = 1j * omega * element.value
        stamp_admittance(Y, a, b, y)

    elif element.kind == ElementKind.ONE_PORT_MODEL:
        assert element.model is not None
        y_val = element.model.y(f_hz)
        y = complex(y_val)  # type: ignore[arg-type]
        stamp_admittance(Y, a, b, y)

    else:
        raise ValueError(f"Unsupported element kind: {element.kind}")


def stamp_norton_source(
    Y: np.ndarray,
    I_vec: np.ndarray,
    source_spec: SourceSpec,
    port_pos_idx: int,
    port_neg_idx: int,
) -> None:
    """Stamp the Thévenin-to-Norton source at the input port.

    Converts the Thévenin source to a Norton equivalent:
    - ``Y_s = 1 / Z_source`` stamped at the port
    - ``I_N = V_th_phasor / Z_source`` injected at port_pos (subtracted
      at port_neg)

    This function is called by the solver, not stored as a graph element.

    Parameters
    ----------
    Y : np.ndarray
        Complex admittance matrix (modified in-place).
    I_vec : np.ndarray
        RHS current vector (modified in-place).
    source_spec : SourceSpec
        The source specification.
    port_pos_idx, port_neg_idx : int
        Matrix indices of the input port nodes, or ``GROUND``.
    """
    z_s = source_spec.z_source
    y_s = 1.0 / z_s
    i_n = source_spec.vth_phasor / z_s

    stamp_admittance(Y, port_pos_idx, port_neg_idx, y_s)

    if port_pos_idx != GROUND:
        I_vec[port_pos_idx] += i_n
    if port_neg_idx != GROUND:
        I_vec[port_neg_idx] -= i_n
