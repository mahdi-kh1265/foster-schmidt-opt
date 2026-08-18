import numpy as np
import pytest

from foster_eom.circuit.graph import Element, ElementKind
from foster_eom.circuit.stamps import stamp_element
from foster_eom.sensitivities.stamps import (
    stamp_capacitor_derivative,
    stamp_inductor_derivative,
)

def test_stamp_capacitor_derivative_fd():
    """Verify analytical dY/dC against central finite difference."""
    f_hz = 10e6
    C_val = 100e-12
    h = 1e-15  # Small step for FD
    
    node_map = {"n1": 0, "n2": 1}
    gnd = "gnd"
    
    # Analytical
    elem = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val)
    Y_p = np.zeros((2, 2), dtype=np.complex128)
    stamp_capacitor_derivative(Y_p, elem, node_map, gnd, f_hz)
    
    # FD
    elem_plus = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val + h)
    elem_minus = Element("C1", ElementKind.CAPACITOR, "n1", "n2", C_val - h)
    
    Y_plus = np.zeros((2, 2), dtype=np.complex128)
    Y_minus = np.zeros((2, 2), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)
    
    Y_fd = (Y_plus - Y_minus) / (2 * h)
    
    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)

def test_stamp_inductor_derivative_fd():
    """Verify analytical dY/dL against central finite difference."""
    f_hz = 10e6
    L_val = 1e-6
    h = 1e-9  # Small step for FD
    
    node_map = {"n1": 0, "n2": 1}
    gnd = "gnd"
    
    # Analytical
    elem = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val)
    Y_p = np.zeros((2, 2), dtype=np.complex128)
    stamp_inductor_derivative(Y_p, elem, node_map, gnd, f_hz)
    
    # FD
    elem_plus = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val + h)
    elem_minus = Element("L1", ElementKind.INDUCTOR, "n1", "n2", L_val - h)
    
    Y_plus = np.zeros((2, 2), dtype=np.complex128)
    Y_minus = np.zeros((2, 2), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)
    
    Y_fd = (Y_plus - Y_minus) / (2 * h)
    
    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)

def test_stamp_capacitor_derivative_gnd():
    """Verify analytical dY/dC against central finite difference with ground."""
    f_hz = 10e6
    C_val = 100e-12
    h = 1e-15
    
    node_map = {"n1": 0}
    gnd = "gnd"
    
    elem = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val)
    Y_p = np.zeros((1, 1), dtype=np.complex128)
    stamp_capacitor_derivative(Y_p, elem, node_map, gnd, f_hz)
    
    elem_plus = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val + h)
    elem_minus = Element("C1", ElementKind.CAPACITOR, "n1", gnd, C_val - h)
    
    Y_plus = np.zeros((1, 1), dtype=np.complex128)
    Y_minus = np.zeros((1, 1), dtype=np.complex128)
    stamp_element(Y_plus, elem_plus, node_map, gnd, f_hz)
    stamp_element(Y_minus, elem_minus, node_map, gnd, f_hz)
    
    Y_fd = (Y_plus - Y_minus) / (2 * h)
    
    np.testing.assert_allclose(Y_p, Y_fd, rtol=1e-5, atol=1e-12)
