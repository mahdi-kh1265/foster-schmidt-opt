"""Unit tests for P10 uncertainty model."""
from __future__ import annotations

import warnings

import pytest

from foster_eom.robustness.uncertainty import (
    PerturbMethod,
    SlotUncertainty,
    UncertaintySource,
    UncertaintyTerm,
    build_slot_uncertainties,
)

# ---------------------------------------------------------------------------
# UncertaintyTerm validation
# ---------------------------------------------------------------------------


class TestUncertaintyTerm:
    def test_symmetric_tol(self) -> None:
        t = UncertaintyTerm(source=UncertaintySource.MANUFACTURING_TOLERANCE, tol_frac=0.05)
        assert t.effective_lo == -0.05
        assert t.effective_hi == 0.05
        assert not t.is_zero

    def test_asymmetric_bounds(self) -> None:
        t = UncertaintyTerm(
            source=UncertaintySource.OPERATING_CONDITION,
            lo_frac=-0.02,
            hi_frac=0.08,
        )
        assert t.effective_lo == -0.02
        assert t.effective_hi == 0.08

    def test_mutual_exclusion(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            UncertaintyTerm(
                source=UncertaintySource.MANUFACTURING_TOLERANCE,
                tol_frac=0.05,
                lo_frac=-0.05,
                hi_frac=0.05,
            )

    def test_asymmetric_requires_both(self) -> None:
        with pytest.raises(ValueError, match="Both lo_frac and hi_frac"):
            UncertaintyTerm(
                source=UncertaintySource.MANUFACTURING_TOLERANCE,
                lo_frac=-0.05,
            )

    def test_lo_must_be_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="lo_frac must be <= 0"):
            UncertaintyTerm(
                source=UncertaintySource.MANUFACTURING_TOLERANCE,
                lo_frac=0.01,
                hi_frac=0.05,
            )

    def test_is_zero_for_zero_tol(self) -> None:
        t = UncertaintyTerm(source=UncertaintySource.MANUFACTURING_TOLERANCE, tol_frac=0.0)
        assert t.is_zero


# ---------------------------------------------------------------------------
# SlotUncertainty
# ---------------------------------------------------------------------------


class TestSlotUncertainty:
    def test_stochastic_with_tol(self) -> None:
        t = UncertaintyTerm(source=UncertaintySource.MANUFACTURING_TOLERANCE, tol_frac=0.05)
        su = SlotUncertainty(
            element_id="b1_L1",
            terms=(t,),
            has_tol_frac=True,
            catalog_tol_frac=0.05,
            perturb_method=PerturbMethod.IDEAL_LC,
        )
        assert su.is_stochastic
        assert su.total_sym_tol == pytest.approx(0.05)

    def test_deterministic_no_terms(self) -> None:
        su = SlotUncertainty(
            element_id="b1_C1",
            terms=(),
            has_tol_frac=False,
            catalog_tol_frac=None,
            perturb_method=PerturbMethod.NONE,
        )
        assert not su.is_stochastic
        assert su.total_sym_tol == 0.0

    def test_multiple_terms_sum_tol(self) -> None:
        t1 = UncertaintyTerm(source=UncertaintySource.MANUFACTURING_TOLERANCE, tol_frac=0.05)
        t2 = UncertaintyTerm(source=UncertaintySource.OPERATING_CONDITION, tol_frac=0.02)
        su = SlotUncertainty(
            element_id="b1_L1",
            terms=(t1, t2),
            has_tol_frac=True,
            catalog_tol_frac=0.05,
            perturb_method=PerturbMethod.IDEAL_LC,
        )
        assert su.total_sym_tol == pytest.approx(0.07)  # 0.05 + 0.02


# ---------------------------------------------------------------------------
# build_slot_uncertainties
# ---------------------------------------------------------------------------


def _make_combo(entries: dict) -> object:
    """Create a minimal mock CatalogCombo."""
    from unittest.mock import MagicMock

    from foster_eom.catalog.component import ModelTier

    combo = MagicMock()
    slot_entries = {}
    for eid, (value_nom, tol_frac, tier) in entries.items():
        entry = MagicMock()
        entry.value_nom = value_nom
        entry.value_tol_frac = tol_frac
        entry.model_tier = ModelTier(tier)
        slot_entries[eid] = entry
    combo.slot_entries = slot_entries
    return combo


class TestBuildSlotUncertainties:
    def test_basic_mfg_tolerance(self) -> None:
        combo = _make_combo({"b1_L1": (10e-9, 0.05, "ideal")})
        sus = build_slot_uncertainties(combo)
        assert len(sus) == 1
        su = sus[0]
        assert su.element_id == "b1_L1"
        assert su.has_tol_frac is True
        assert su.catalog_tol_frac == 0.05
        assert su.is_stochastic
        assert su.perturb_method == PerturbMethod.IDEAL_LC
        assert len(su.terms) == 1
        assert su.terms[0].source == UncertaintySource.MANUFACTURING_TOLERANCE

    def test_missing_tol_frac_persisted_as_non_stochastic(self) -> None:
        combo = _make_combo({"b1_C1": (10e-12, None, "ideal")})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sus = build_slot_uncertainties(combo)
            assert len(w) == 1
            assert "no uncertainty data" in str(w[0].message)
        su = sus[0]
        assert not su.is_stochastic
        assert su.has_tol_frac is False
        assert su.catalog_tol_frac is None
        assert su.perturb_method == PerturbMethod.NONE

    def test_measured_tier_uses_residual_method(self) -> None:
        combo = _make_combo({"b1_L1": (10e-9, 0.05, "measured")})
        sus = build_slot_uncertainties(combo)
        assert sus[0].perturb_method == PerturbMethod.MEASURED_RESIDUAL

    def test_manufacturing_source_preserved_for_measured_tier(self) -> None:
        """Manufacturing tolerance source is preserved even when
        perturb_method=measured_residual."""
        combo = _make_combo({"b1_L1": (10e-9, 0.05, "measured")})
        sus = build_slot_uncertainties(combo)
        su = sus[0]
        assert su.perturb_method == PerturbMethod.MEASURED_RESIDUAL
        assert su.terms[0].source == UncertaintySource.MANUFACTURING_TOLERANCE

    def test_op_condition_override_adds_term(self) -> None:
        combo = _make_combo({"b1_L1": (10e-9, 0.05, "ideal")})
        sus = build_slot_uncertainties(combo, op_condition_overrides={"b1_L1": 0.03})
        su = sus[0]
        assert len(su.terms) == 2
        sources = {t.source for t in su.terms}
        assert UncertaintySource.MANUFACTURING_TOLERANCE in sources
        assert UncertaintySource.OPERATING_CONDITION in sources

    def test_model_uncertainty_override_adds_term(self) -> None:
        combo = _make_combo({"b1_L1": (10e-9, 0.05, "ideal")})
        sus = build_slot_uncertainties(combo, model_uncertainty_overrides={"b1_L1": 0.01})
        su = sus[0]
        sources = {t.source for t in su.terms}
        assert UncertaintySource.MODEL_UNCERTAINTY in sources
        # Verify distribution for model uncertainty is normal
        model_term = next(t for t in su.terms if t.source == UncertaintySource.MODEL_UNCERTAINTY)
        assert model_term.distribution == "normal_3sigma"
