"""P10 uncertainty model.

Defines per-slot uncertainty specifications and draw functions.
Multiple additive UncertaintyTerm records may exist per slot.
Manufacturing tolerance source identity is preserved independently
of which perturbation method represents it in the circuit.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from foster_eom.realization.result import CatalogCombo


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UncertaintySource(enum.StrEnum):
    """Semantic origin of an uncertainty term (informational tag)."""

    MANUFACTURING_TOLERANCE = "manufacturing_tolerance"
    OPERATING_CONDITION = "operating_condition"
    MODEL_UNCERTAINTY = "model_uncertainty"


class PerturbMethod(enum.StrEnum):
    """How the drawn value deviation is applied to the circuit model.

    This is a *method* tag, not a source tag.  Manufacturing tolerance
    may be represented via either method depending on the model tier.
    """

    IDEAL_LC = "ideal_lc"
    """Ideal/parametric model: scale the L or C parameter directly."""

    MEASURED_RESIDUAL = "measured_residual"
    """Measured tabular model: apply first-order impedance correction.

    For inductor (delta = L_draw/L_nom - 1):
        Z_draw(f) = Z_meas(f) + j*2*pi*f * L_nom * delta

    For capacitor (delta = C_draw/C_nom - 1):
        Z_draw(f) = Z_meas(f) - delta / (j*2*pi*f * C_nom * (1 + delta))

    This preserves the full measured impedance shape with a first-order
    reactive correction.  It is an approximation; recorded in perturbation_notes.
    """

    NONE = "none"
    """No perturbation: slot is deterministic (no tolerance data available)."""


# ---------------------------------------------------------------------------
# Uncertainty term
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyTerm:
    """One additive uncertainty contribution for a slot.

    Multiple terms may exist per slot (e.g. manufacturing + operating-condition).
    Total fractional deviation per sample is the sum of all term draws.

    Parameters
    ----------
    source : UncertaintySource
        Semantic origin (informational only; does not affect calculation).
    tol_frac : float | None
        Symmetric ±fraction (e.g. 0.05 → ±5%). Mutually exclusive with
        lo_frac / hi_frac.
    lo_frac : float | None
        Lower bound fraction (≤ 0).  Set together with hi_frac for asymmetric.
    hi_frac : float | None
        Upper bound fraction (≥ 0).  Set together with lo_frac.
    distribution : str
        "uniform" — draw uniformly in [lo, hi].
        "normal_3sigma" — draw from N(0, σ=tol_frac/3), clipped at ±3σ.
    """

    source: UncertaintySource
    tol_frac: float | None = None
    lo_frac: float | None = None
    hi_frac: float | None = None
    distribution: Literal["uniform", "normal_3sigma"] = "uniform"

    def __post_init__(self) -> None:
        has_sym = self.tol_frac is not None
        has_asym = self.lo_frac is not None or self.hi_frac is not None
        if has_sym and has_asym:
            raise ValueError("Set tol_frac OR (lo_frac, hi_frac), not both.")
        if has_asym and (self.lo_frac is None or self.hi_frac is None):
            raise ValueError("Both lo_frac and hi_frac must be set for asymmetric bounds.")
        if has_asym:
            # post-check: lo_frac and hi_frac are not None here (validated above)
            assert self.lo_frac is not None and self.hi_frac is not None
            if self.lo_frac > 0:
                raise ValueError("lo_frac must be <= 0.")
            if self.hi_frac < 0:
                raise ValueError("hi_frac must be >= 0.")

    @property
    def effective_lo(self) -> float:
        """Lower bound fraction (non-positive)."""
        if self.tol_frac is not None:
            return -self.tol_frac
        return self.lo_frac or 0.0

    @property
    def effective_hi(self) -> float:
        """Upper bound fraction (non-negative)."""
        if self.tol_frac is not None:
            return self.tol_frac
        return self.hi_frac or 0.0

    @property
    def is_zero(self) -> bool:
        """True if this term produces zero deviation (degenerate)."""
        return abs(self.effective_lo) < 1e-15 and abs(self.effective_hi) < 1e-15


# ---------------------------------------------------------------------------
# Slot uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotUncertainty:
    """Aggregated uncertainty specification for one circuit slot.

    Parameters
    ----------
    element_id : str
    terms : tuple[UncertaintyTerm, ...]
        Additive uncertainty terms.  Empty tuple → deterministic slot.
    has_tol_frac : bool
        Whether the catalog entry provided value_tol_frac.
        Persisted even when no uncertainty is available.
    catalog_tol_frac : float | None
        Raw catalog value_tol_frac (None = not declared in catalog).
    perturb_method : PerturbMethod
        How the drawn deviation is applied to the circuit model.
    """

    element_id: str
    terms: tuple[UncertaintyTerm, ...]
    has_tol_frac: bool
    catalog_tol_frac: float | None
    perturb_method: PerturbMethod

    @property
    def is_stochastic(self) -> bool:
        """True if any term produces non-zero uncertainty."""
        return any(not t.is_zero for t in self.terms)

    @property
    def total_sym_tol(self) -> float:
        """Sum of absolute tol_frac values across all symmetric terms (informational)."""
        total = 0.0
        for t in self.terms:
            total += abs(t.effective_hi)
        return total


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_slot_uncertainties(
    combo: CatalogCombo,
    op_condition_overrides: dict[str, float] | None = None,
    model_uncertainty_overrides: dict[str, float] | None = None,
) -> list[SlotUncertainty]:
    """Build SlotUncertainty list from a frozen P09 CatalogCombo.

    Parameters
    ----------
    combo : CatalogCombo
        Frozen P09 realization result (best combo).
    op_condition_overrides : dict[str, float] | None
        Optional user-supplied operating-condition fractional tolerance per
        element_id (e.g. {"b1_L1": 0.03} for ±3% thermal shift).
        Never auto-generated from missing data.
    model_uncertainty_overrides : dict[str, float] | None
        Optional user-supplied model-uncertainty fractional tolerance per
        element_id. Never auto-generated.

    Returns
    -------
    list[SlotUncertainty]
        One entry per slot. Warns on non-stochastic slots.
    """
    from foster_eom.catalog.component import ModelTier

    op_overrides = op_condition_overrides or {}
    model_overrides = model_uncertainty_overrides or {}

    result: list[SlotUncertainty] = []
    non_stochastic: list[str] = []

    for element_id, entry in combo.slot_entries.items():
        terms: list[UncertaintyTerm] = []

        # --- Manufacturing tolerance ---
        catalog_tol = entry.value_tol_frac
        has_tol = catalog_tol is not None
        if catalog_tol is not None and catalog_tol > 0:
            terms.append(
                UncertaintyTerm(
                    source=UncertaintySource.MANUFACTURING_TOLERANCE,
                    tol_frac=catalog_tol,
                    distribution="uniform",
                )
            )

        # --- Operating-condition shift (user-supplied only) ---
        op_frac = op_overrides.get(element_id)
        if op_frac is not None and op_frac > 0:
            terms.append(
                UncertaintyTerm(
                    source=UncertaintySource.OPERATING_CONDITION,
                    tol_frac=op_frac,
                    distribution="uniform",
                )
            )

        # --- Model uncertainty (user-supplied only) ---
        mod_frac = model_overrides.get(element_id)
        if mod_frac is not None and mod_frac > 0:
            terms.append(
                UncertaintyTerm(
                    source=UncertaintySource.MODEL_UNCERTAINTY,
                    tol_frac=mod_frac,
                    distribution="normal_3sigma",
                )
            )

        # --- Perturbation method (depends on model tier, not source) ---
        if not terms:
            perturb = PerturbMethod.NONE
        elif entry.model_tier == ModelTier.MEASURED:
            perturb = PerturbMethod.MEASURED_RESIDUAL
        else:
            perturb = PerturbMethod.IDEAL_LC

        su = SlotUncertainty(
            element_id=element_id,
            terms=tuple(terms),
            has_tol_frac=has_tol,
            catalog_tol_frac=catalog_tol,
            perturb_method=perturb,
        )
        result.append(su)

        if not su.is_stochastic:
            non_stochastic.append(element_id)

    if non_stochastic:
        warnings.warn(
            f"P10: {len(non_stochastic)} slot(s) have no uncertainty data and will be "
            f"held at nominal value throughout Monte Carlo: {non_stochastic}. "
            "No uncertainty is invented for missing catalog tolerance specifications.",
            stacklevel=2,
        )

    return result
