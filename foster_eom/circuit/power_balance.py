"""Complex-power balance diagnostic (spec §14.8).

For passive networks at each frequency:

    S_delivered - sum(S_element) ~= 0

within combined absolute/relative tolerance.
"""

from __future__ import annotations

from foster_eom.circuit.measurements import CircuitSolution


def check_power_balance(
    solution: CircuitSolution,
    atol: float = 1e-12,
    rtol: float = 1e-6,
) -> tuple[complex, bool]:
    """Check complex-power balance for a circuit solution.

    Parameters
    ----------
    solution : CircuitSolution
        A solved circuit solution with power data.
    atol : float
        Absolute tolerance (watts).
    rtol : float
        Relative tolerance.

    Returns
    -------
    residual : complex
        ``S_delivered - sum(S_element)``
    ok : bool
        ``|residual| < atol + rtol * |S_delivered|``
    """
    if solution.power_balance_residual is None or solution.s_source_delivered is None:
        return complex(0.0), False

    residual = solution.power_balance_residual
    s_del = solution.s_source_delivered
    ok = abs(residual) < atol + rtol * abs(s_del)
    return residual, ok
