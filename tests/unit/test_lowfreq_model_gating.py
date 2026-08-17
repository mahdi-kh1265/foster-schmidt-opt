"""Regression tests: low-frequency (5-12 MHz) model-validity gating.

Ensures _select_best_mc correctly rejects a measured model whose validity
range does not cover the requested frequency band, and verifies that the
correct fallback tier is selected.

These tests are UNIT tests that do not require the real vendor DB.
They mock ModelCondition objects with explicit validity ranges.
"""

from __future__ import annotations

from foster_eom.catalog.component import FallbackPolicy, ModelTier
from foster_eom.realization.spec import SlotSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mc(
    *,
    tier: str,
    validity_lo: float | None = None,
    validity_hi: float | None = None,
    mc_id: str = "mc1",
) -> object:
    """Create a minimal ModelCondition-like object for testing."""
    from unittest.mock import MagicMock

    from foster_eom.catalog.component import ModelTier

    mc = MagicMock()
    mc.id = mc_id
    mc.model_tier = ModelTier(tier)

    if validity_lo is not None and validity_hi is not None:
        mc.validity_hz.return_value = (validity_lo, validity_hi)
    else:
        mc.validity_hz.return_value = None  # unlimited (ideal)

    return mc


def _slot(band_hz: tuple[float, float], value: float = 10e-12, eid: str = "b1_C1") -> SlotSpec:
    return SlotSpec(
        element_id=eid,
        value_nom=value,
        freq_range_hz=band_hz,
        fallback_policy=FallbackPolicy.STRICT,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLowFreqModelGating:
    """_select_best_mc rejects 100 MHz+ measured models for a 5-12 MHz band."""

    EOM_BAND = (5e6, 12e6)

    def test_measured_100mhz_rejected_for_5_12mhz(self) -> None:
        """Murata GJM measured model (100 MHz - 30 GHz) must be rejected
        when the required band is 5-12 MHz."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        murata_measured = _make_mc(
            tier="measured",
            validity_lo=100e6,
            validity_hi=30e9,
            mc_id="murata_measured",
        )
        slot = _slot(self.EOM_BAND)
        result = _select_best_mc([murata_measured], slot)
        assert result is None, (
            "Measured model valid 100MHz-30GHz must not be selected for 5-12 MHz band"
        )

    def test_ideal_unlimited_accepted_for_5_12mhz(self) -> None:
        """Ideal model (validity=None = unlimited) must be accepted for 5-12 MHz."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        ideal_mc = _make_mc(tier="ideal", mc_id="ideal_cap")
        slot = _slot(self.EOM_BAND)
        result = _select_best_mc([ideal_mc], slot)
        assert result is not None
        assert result.model_tier == ModelTier.IDEAL

    def test_strict_measured_plus_ideal_returns_none(self) -> None:
        """STRICT: When measured is out-of-band and ideal is available, it does NOT
        silently downgrade to ideal. It marks the part ineligible (returns None)."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        murata_measured = _make_mc(
            tier="measured",
            validity_lo=100e6,
            validity_hi=30e9,
            mc_id="measured_oob",
        )
        ideal_mc = _make_mc(tier="ideal", mc_id="ideal_fb")
        slot = _slot(self.EOM_BAND)  # STRICT by default
        result = _select_best_mc([murata_measured, ideal_mc], slot)
        assert result is None, "STRICT must reject part if highest tier model is out of band"

    def test_allow_lower_tier_measured_plus_ideal_selects_ideal(self) -> None:
        """ALLOW_LOWER_TIER: When measured is out-of-band and ideal is available,
        it explicitly permits fallback to ideal."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        murata_measured = _make_mc(
            tier="measured",
            validity_lo=100e6,
            validity_hi=30e9,
            mc_id="measured_oob",
        )
        ideal_mc = _make_mc(tier="ideal", mc_id="ideal_fb")
        slot = SlotSpec(
            element_id="b1_C1",
            value_nom=10e-12,
            freq_range_hz=self.EOM_BAND,
            fallback_policy=FallbackPolicy.ALLOW_LOWER_TIER,
        )

        result = _select_best_mc([murata_measured, ideal_mc], slot)
        assert result is not None, (
            "ALLOW_LOWER_TIER must fallback to ideal when measured is out-of-band"
        )
        assert result.id == "ideal_fb"
        assert result.model_tier == ModelTier.IDEAL

    def test_measured_1mhz_to_9ghz_accepted_for_5_12mhz(self) -> None:
        """Coilcraft measured model (1 MHz - 9 GHz) covers 5-12 MHz: must be accepted
        and preferred over ideal."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        coilcraft_measured = _make_mc(
            tier="measured",
            validity_lo=1e6,
            validity_hi=9e9,
            mc_id="coilcraft_meas",
        )
        ideal_mc = _make_mc(tier="ideal", mc_id="ideal_ind")
        slot = _slot(self.EOM_BAND, value=10e-9, eid="b1_L1")
        result = _select_best_mc([coilcraft_measured, ideal_mc], slot)
        assert result is not None
        assert result.id == "coilcraft_meas", (
            "Coilcraft measured (1MHz-9GHz) must win over ideal when covering 5-12 MHz"
        )

    def test_strict_policy_returns_none_when_no_valid_model(self) -> None:
        """STRICT policy: if the only model is measured-OOB and there is no ideal,
        _select_best_mc must return None (do not fabricate a result)."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        only_oob = _make_mc(
            tier="measured",
            validity_lo=100e6,
            validity_hi=30e9,
            mc_id="only_oob",
        )
        slot = _slot(self.EOM_BAND)
        result = _select_best_mc([only_oob], slot)
        assert result is None

    def test_measured_exactly_covers_band_boundary(self) -> None:
        """Model valid [5 MHz, 12 MHz] covers exactly [5 MHz, 12 MHz]: accepted."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        exact_mc = _make_mc(
            tier="measured",
            validity_lo=5e6,
            validity_hi=12e6,
            mc_id="exact_band",
        )
        slot = _slot(self.EOM_BAND)
        result = _select_best_mc([exact_mc], slot)
        assert result is not None
        assert result.id == "exact_band"

    def test_measured_partially_overlaps_rejected(self) -> None:
        """Model valid [8 MHz, 30 GHz] starts mid-band: does NOT fully cover
        [5 MHz, 12 MHz] (8 MHz > 5 MHz) -> rejected."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        partial_mc = _make_mc(
            tier="measured",
            validity_lo=8e6,
            validity_hi=30e9,
            mc_id="partial_mc",
        )
        slot = _slot(self.EOM_BAND)
        result = _select_best_mc([partial_mc], slot)
        assert result is None, (
            "Model valid 8-30000 MHz does not cover 5 MHz lower edge; must be rejected"
        )

    def test_no_freq_filter_on_slot_accepts_any_model(self) -> None:
        """SlotSpec with freq_range_hz=None applies no frequency gating."""
        from foster_eom.realization.neighborhoods import _select_best_mc

        oob_mc = _make_mc(
            tier="measured",
            validity_lo=100e6,
            validity_hi=30e9,
            mc_id="oob_no_filter",
        )
        slot = SlotSpec(
            element_id="b1_C1",
            value_nom=10e-12,
            freq_range_hz=None,
            fallback_policy=FallbackPolicy.STRICT,
        )
        result = _select_best_mc([oob_mc], slot)
        assert result is not None, (
            "No freq_range on slot -> no frequency gating -> OOB model accepted"
        )
