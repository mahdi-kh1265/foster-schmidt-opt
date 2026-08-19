"""Derivative stamps for MNA matrices."""

import numpy as np

from foster_eom.circuit.graph import GROUND, Element, ElementKind
from foster_eom.circuit.stamps import stamp_admittance


def stamp_capacitor_derivative(
    Y_p: np.ndarray,
    element: Element,
    node_map: dict[str, int],
    ground_node_id: str,
    f_hz: float,
) -> None:
    """Stamp dY/dC into the derivative matrix Y_p.

    Parameters
    ----------
    Y_p : np.ndarray
        Complex admittance derivative matrix (modified in-place).
    element : Element
        The capacitor circuit element.
    node_map : dict[str, int]
        Mapping from non-ground node IDs to matrix indices.
    ground_node_id : str
        ID of the ground node.
    f_hz : float
        Frequency in Hz.
    """
    assert element.kind == ElementKind.CAPACITOR
    omega = 2.0 * np.pi * f_hz
    dy_dC = 1j * omega

    a = GROUND if element.node_pos == ground_node_id else node_map[element.node_pos]
    b = GROUND if element.node_neg == ground_node_id else node_map[element.node_neg]
    stamp_admittance(Y_p, a, b, dy_dC)


def stamp_inductor_derivative(
    Y_p: np.ndarray,
    element: Element,
    node_map: dict[str, int],
    ground_node_id: str,
    f_hz: float,
) -> None:
    """Stamp dY/dL into the derivative matrix Y_p.

    Parameters
    ----------
    Y_p : np.ndarray
        Complex admittance derivative matrix (modified in-place).
    element : Element
        The inductor circuit element.
    node_map : dict[str, int]
        Mapping from non-ground node IDs to matrix indices.
    ground_node_id : str
        ID of the ground node.
    f_hz : float
        Frequency in Hz.
    """
    assert element.kind == ElementKind.INDUCTOR
    assert element.value is not None

    omega = 2.0 * np.pi * f_hz

    # y = 1 / (j * omega * L)
    # dy/dL = -1 / (j * omega * L^2)
    z_l_sq = np.complex128(1j * omega * (element.value**2))

    with np.errstate(divide="ignore", invalid="ignore"):
        dy_dL = complex(np.complex128(-1.0) / z_l_sq)

    a = GROUND if element.node_pos == ground_node_id else node_map[element.node_pos]
    b = GROUND if element.node_neg == ground_node_id else node_map[element.node_neg]
    stamp_admittance(Y_p, a, b, dy_dL)
