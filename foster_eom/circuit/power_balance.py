"""Complex-power balance diagnostic (spec S14.8).

For passive networks at each frequency:

    S_delivered - sum(S_element) ~= 0

within combined absolute/relative tolerance.

The scale denominator uses ``max(|S_port|, sum(|S_k|))`` so that
high-Q resonant networks with large cancelling reactive powers
are properly tested.
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
        ``|residual| < atol + rtol * S_scale`` where
        ``S_scale = max(|S_port|, sum(|S_k|))``
    """
    if (
        solution.power_balance_residual is None
        or solution.s_source_delivered is None
        or solution.element_measurements is None
    ):
        return complex(0.0), False

    residual = solution.power_balance_residual
    s_del = solution.s_source_delivered
    s_elem_abs_sum = sum(abs(em.complex_power) for em in solution.element_measurements.values())
    s_scale = max(abs(s_del), s_elem_abs_sum)
    ok = abs(residual) < atol + rtol * s_scale
    return residual, ok
