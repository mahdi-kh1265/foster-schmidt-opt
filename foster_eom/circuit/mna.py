"""MNA assembly and linear solve (spec §14.1).

Multi-criterion validity checking:

1. Nonfinite pre-screen (NaN/Inf in Y or I).
2. Condition number check.
3. Linear solve with exception handling.
4. Solution finiteness.
5. Normalized residual check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.stamps import stamp_element, stamp_norton_source
from foster_eom.domain.source import SourceSpec
from foster_eom.errors import CircuitSolveStatus

# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverOptions:
    """Configurable thresholds for the MNA solver.

    Attributes
    ----------
    condition_threshold : float
        Flag ill-conditioning above this value.
    residual_threshold : float
        Normalized ``||Y*V - I|| / max(||I||, eps)`` threshold.
    power_balance_atol : float
        Absolute tolerance for complex-power balance (watts).
    power_balance_rtol : float
        Relative tolerance for complex-power balance.
    """

    condition_threshold: float = 1e14
    residual_threshold: float = 1e-10
    power_balance_atol: float = 1e-12
    power_balance_rtol: float = 1e-6


# ---------------------------------------------------------------------------
# Solve diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveDiagnostics:
    """Diagnostic information from a single-frequency MNA solve.

    Attributes
    ----------
    condition_number : float
        Condition number of Y (inf if nonfinite or not computed).
    residual_norm : float
        Normalized residual ``||Y*V - I|| / max(||I||, eps)`` (inf if not computed).
    nonfinite_in_matrix : bool
        True if Y contained NaN or Inf before solve.
    nonfinite_in_rhs : bool
        True if I contained NaN or Inf before solve.
    nonfinite_in_solution : bool
        True if V contained NaN or Inf after solve.
    """

    condition_number: float = float("inf")
    residual_norm: float = float("inf")
    nonfinite_in_matrix: bool = False
    nonfinite_in_rhs: bool = False
    nonfinite_in_solution: bool = False


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_mna(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    f_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Assemble the nodal admittance matrix and RHS current vector.

    Parameters
    ----------
    graph : CircuitGraph
        The passive circuit netlist.
    source_spec : SourceSpec
        External Thévenin source specification.
    f_hz : float
        Frequency in Hz.

    Returns
    -------
    Y : np.ndarray
        ``(N, N)`` complex admittance matrix.
    I_vec : np.ndarray
        ``(N,)`` complex RHS current vector.
    node_map : dict[str, int]
        Node ID → matrix index mapping.
    """
    node_map = graph.node_indices()
    n = len(node_map)

    Y = np.zeros((n, n), dtype=np.complex128)
    I_vec = np.zeros(n, dtype=np.complex128)

    # Stamp all passive elements
    for elem in graph.elements.values():
        stamp_element(Y, elem, node_map, graph.ground_node_id, f_hz)

    # Stamp the Norton-equivalent source at the input port
    port_pos = graph._resolve_index(graph.input_port.node_pos, node_map)
    port_neg = graph._resolve_index(graph.input_port.node_neg, node_map)
    stamp_norton_source(Y, I_vec, source_spec, port_pos, port_neg)

    return Y, I_vec, node_map


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve_mna(
    Y: np.ndarray,
    I_vec: np.ndarray,
    opts: SolverOptions | None = None,
) -> tuple[np.ndarray | None, CircuitSolveStatus, SolveDiagnostics]:
    """Solve the linear system ``Y · V = I``.

    Uses multi-criterion validity checking:

    1. Nonfinite pre-screen.
    2. Condition number check.
    3. ``np.linalg.solve`` with exception handling.
    4. Solution finiteness.
    5. Normalized residual check.

    Parameters
    ----------
    Y : np.ndarray
        Complex admittance matrix.
    I_vec : np.ndarray
        Complex RHS current vector.
    opts : SolverOptions | None
        Solver options.  Defaults to ``SolverOptions()``.

    Returns
    -------
    V : np.ndarray | None
        Node voltages, or ``None`` if the solve failed.
    status : CircuitSolveStatus
        Outcome status.
    diagnostics : SolveDiagnostics
        Diagnostic details.
    """
    if opts is None:
        opts = SolverOptions()

    # -- 1. Nonfinite pre-screen -------------------------------------------
    nf_matrix = not np.all(np.isfinite(Y))
    nf_rhs = not np.all(np.isfinite(I_vec))
    if nf_matrix or nf_rhs:
        return (
            None,
            CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED,
            SolveDiagnostics(
                nonfinite_in_matrix=nf_matrix,
                nonfinite_in_rhs=nf_rhs,
            ),
        )

    # -- 2. Condition number -----------------------------------------------
    try:
        cond = float(np.linalg.cond(Y))
    except (np.linalg.LinAlgError, ValueError):
        cond = float("inf")

    if cond > opts.condition_threshold:
        return (
            None,
            CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED,
            SolveDiagnostics(condition_number=cond),
        )

    # -- 3. Solve ----------------------------------------------------------
    try:
        V = np.linalg.solve(Y, I_vec)
    except np.linalg.LinAlgError:
        return (
            None,
            CircuitSolveStatus.SINGULAR_OR_ILL_CONDITIONED,
            SolveDiagnostics(condition_number=cond),
        )

    # -- 4. Solution finiteness --------------------------------------------
    nf_sol = not np.all(np.isfinite(V))
    if nf_sol:
        return (
            None,
            CircuitSolveStatus.NUMERICAL_ERROR,
            SolveDiagnostics(
                condition_number=cond,
                nonfinite_in_solution=True,
            ),
        )

    # -- 5. Residual check -------------------------------------------------
    residual = float(np.linalg.norm(Y @ V - I_vec))
    i_norm = max(float(np.linalg.norm(I_vec)), 1e-30)
    residual_norm = residual / i_norm

    if residual_norm > opts.residual_threshold:
        return (
            None,
            CircuitSolveStatus.NUMERICAL_ERROR,
            SolveDiagnostics(
                condition_number=cond,
                residual_norm=residual_norm,
            ),
        )

    return (
        V,
        CircuitSolveStatus.OK,
        SolveDiagnostics(
            condition_number=cond,
            residual_norm=residual_norm,
        ),
    )
