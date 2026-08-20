"""Top-level circuit solve entry points.

Ties together graph → MNA assembly → solve → measurements.
"""

from __future__ import annotations

import numpy as np

from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.measurements import CircuitSolution, compute_measurements
from foster_eom.circuit.mna import SolvedMNASystem, SolverOptions, assemble_mna, solve_mna
from foster_eom.domain.source import SourceSpec


def solve_circuit_single(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    f_hz: float,
    opts: SolverOptions | None = None,
) -> CircuitSolution:
    """Solve the circuit at a single frequency.

    Parameters
    ----------
    graph : CircuitGraph
        The passive circuit netlist.
    source_spec : SourceSpec
        External Thévenin source specification.
    f_hz : float
        Frequency in Hz.
    opts : SolverOptions | None
        Solver options.  Defaults to ``SolverOptions()``.

    Returns
    -------
    CircuitSolution
        Full solution with measurements, or failed status.
    """
    sol, _ = solve_circuit_single_with_state(graph, source_spec, f_hz, opts)
    return sol


def solve_circuit_single_with_state(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    f_hz: float,
    opts: SolverOptions | None = None,
) -> tuple[CircuitSolution, SolvedMNASystem | None]:
    """Solve at one frequency, additionally returning the raw MNA system (P12.5-F).

    Numerically identical to :func:`solve_circuit_single` — it *is* the body of
    that function — but it also hands back the assembled ``(Y, I, node_map)``
    and the nominal solution ``V`` so a derivative consumer can reuse them
    instead of assembling and screening the same system a second time.

    The raw system is returned only when the solve succeeded; a failed solve
    yields ``None`` so no consumer can reuse a state that has no nominal
    solution.
    """
    if opts is None:
        opts = SolverOptions()

    graph.validate()

    Y, I_vec, node_map = assemble_mna(graph, source_spec, f_hz)
    V, status, diagnostics = solve_mna(Y, I_vec, opts)

    if V is None:
        return (
            CircuitSolution(
                f_hz=f_hz,
                status=status,
                diagnostics=diagnostics,
            ),
            None,
        )

    sol = compute_measurements(
        graph,
        source_spec,
        V,
        node_map,
        f_hz,
        diagnostics,
        atol=opts.power_balance_atol,
        rtol=opts.power_balance_rtol,
    )
    system = SolvedMNASystem(
        f_hz=f_hz,
        y=Y,
        b=I_vec,
        node_map=node_map,
        v_nominal=V,
        status=status,
        diagnostics=diagnostics,
    )
    return sol, system


def solve_circuit(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    f_hz_array: np.ndarray | list[float],
    opts: SolverOptions | None = None,
) -> list[CircuitSolution]:
    """Solve the circuit across an array of frequencies.

    Parameters
    ----------
    graph : CircuitGraph
        The passive circuit netlist.
    source_spec : SourceSpec
        External Thévenin source specification.
    f_hz_array : array-like
        Frequencies in Hz.
    opts : SolverOptions | None
        Solver options.

    Returns
    -------
    list[CircuitSolution]
        One solution per frequency.
    """
    if opts is None:
        opts = SolverOptions()

    graph.validate()

    return [solve_circuit_single(graph, source_spec, float(f), opts) for f in f_hz_array]
