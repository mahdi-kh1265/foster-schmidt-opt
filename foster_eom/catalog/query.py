"""Component query specification (Prompt 08).

``ComponentQuery`` defines filter criteria for eligible components.
All non-None fields are conjoined (AND). ``freq_range_hz`` requires
model validity to cover the entire requested range.
"""

from __future__ import annotations

from dataclasses import dataclass

from foster_eom.catalog.component import ComponentKind, ModelTier


@dataclass
class ComponentQuery:
    """Filter specification for component eligibility queries.

    All ``None`` fields are ignored. Non-None constraints are conjoined (AND).
    """

    kind: ComponentKind | None = None
    vendor: str | None = None
    package: str | None = None

    # Value range (SI, inclusive bounds)
    value_min: float | None = None
    value_max: float | None = None
    tol_max_frac: float | None = None  # e.g. 0.10 → ≤10%

    # Ratings (minimum acceptable)
    voltage_min_v: float | None = None
    current_min_a: float | None = None
    current_sat_min_a: float | None = None

    # RF conditions
    srf_min_hz: float | None = None
    q_min: float | None = None  # requires q_at_f_hz IS NOT NULL
    esr_max_ohm: float | None = None

    # Model requirements
    model_tier_min: ModelTier | None = None

    # Frequency eligibility — model validity must cover ENTIRE range
    freq_range_hz: tuple[float, float] | None = None

    # Availability
    in_stock_only: bool = False

    # Pattern match
    part_number_glob: str | None = None  # SQL LIKE pattern
