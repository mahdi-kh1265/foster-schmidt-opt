"""Test fixtures and synthetic models."""

from __future__ import annotations

from typing import Any

from foster_eom.domain.eom import ExtrapolationPolicy, MotionalBranch
from foster_eom.models.eom_mbvd import MBVDModel


def create_synthetic_mbvd() -> MBVDModel:
    """Create a synthetic mBVD EOM model for testing.

    This implements the non-predictive fixture described in spec §7.3.
    It must never be presented as the expected real POSM EOM.
    """

    class LabeledMBVDModel(MBVDModel):
        """MBVD model that overrides metadata to enforce SYNTHETIC_TEST_ONLY label."""

        def metadata(self) -> dict[str, Any]:
            meta = super().metadata()
            meta["label"] = "SYNTHETIC_TEST_ONLY"
            return meta

    return LabeledMBVDModel(
        c0_f=12e-12,
        g0_s=2e-5,
        rs_ohm=0.5,
        ls_h=15e-9,
        motional_branches=[
            MotionalBranch(
                rm_ohm=8.0,
                lm_h=50e-6,
                cm_f=9.0e-12,
            )
        ],
        extrapolation_policy=ExtrapolationPolicy.ERROR,
    )
