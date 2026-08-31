"""Realization specification types (Prompt 09).

``RealizationSpec``   — user-facing configuration for one realization run.
``SlotSpec``          — per-element eligibility config (auto-built or user-supplied).
``RealizationBudget`` — MNA-solve budget limit.
``NeighborhoodEntry`` — one frozen catalog candidate binding component + model IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from foster_eom.catalog.component import FallbackPolicy, ModelTier

# ---------------------------------------------------------------------------
# Per-slot eligibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """Per-element query configuration for one Foster slot.

    Parameters
    ----------
    element_id : str
        Canonical element ID as produced by the circuit builder, e.g.
        ``"b1_C0"``, ``"b1_L1"``, ``"b2_C1"``.
    value_nom : float
        Continuous target value in SI (H or F).
    value_ratio : float
        Search window: ``[value_nom/ratio, value_nom*ratio]``.
    freq_range_hz : tuple[float, float] | None
        Verification band.  Model validity must cover the entire range.
        ``None`` means no frequency filter.
    package : str | None
        Restrict to this package footprint (SQL exact match).
    voltage_min_v : float | None
        Minimum voltage rating.
    current_min_a : float | None
        Minimum current rating.
    current_sat_min_a : float | None
        Minimum saturation current.
    required_tier : ModelTier | None
        Minimum model tier required.
    fallback_policy : FallbackPolicy
        How to handle tier mismatches in ``build_model``.
    in_stock_only : bool
        Restrict to eligible model conditions.
    """

    element_id: str
    value_nom: float
    value_ratio: float = 1.5
    freq_range_hz: tuple[float, float] | None = None
    package: str | None = None
    voltage_min_v: float | None = None
    current_min_a: float | None = None
    current_sat_min_a: float | None = None
    required_tier: ModelTier | None = None
    fallback_policy: FallbackPolicy = FallbackPolicy.STRICT
    in_stock_only: bool = False


# ---------------------------------------------------------------------------
# Frozen neighborhood entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeighborhoodEntry:
    """One frozen catalog candidate for a single slot.

    Both IDs are resolved at neighborhood-generation time so that later
    catalog changes cannot alter a frozen realization.
    """

    component_id: str
    model_condition_id: str  # selected MC ID at query time
    vendor: str
    part_number: str
    value_nom: float  # catalog nominal value (SI)
    value_tol_frac: float | None
    model_tier: ModelTier
    log_ratio: float  # abs(log(value_nom / slot.value_nom)); sort key


# ---------------------------------------------------------------------------
# Realization spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizationSpec:
    """User-facing configuration for one discrete realization run.

    Parameters
    ----------
    slot_specs : tuple[SlotSpec, ...]
        Per-element eligibility specs.  Normally auto-built by
        ``build_slot_specs()``; user may override per-slot.
    k_max : int
        Maximum catalog candidates per slot.
    exhaustive_threshold : int
        If total combo count <= this, enumerate exhaustively; otherwise beam.
    beam_width : int
        Beam width for beam search (used only when not exhaustive).
    random_seed : int
        Tie-breaking seed for beam search.
    combination_mode : str
        ``"single"`` only in P09.
    verify_top_k : int
        Number of top (Deb-ordered) combos to run through P06 verification.
    """

    slot_specs: tuple[SlotSpec, ...]
    k_max: int = 5
    exhaustive_threshold: int = 64
    beam_width: int = 8
    random_seed: int = 0
    combination_mode: str = "single"
    verify_top_k: int = 3


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class RealizationBudget:
    """MNA-solve budget for one realization run."""

    max_mna_solves: int = 512
    used: int = field(default=0, compare=False, repr=False)

    def remaining(self) -> int:
        return max(0, self.max_mna_solves - self.used)

    def consume(self, n: int) -> None:
        self.used += n

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_mna_solves
