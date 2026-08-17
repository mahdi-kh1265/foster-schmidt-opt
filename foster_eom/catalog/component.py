"""Component library data structures (Prompt 08).

Defines ``LibraryComponent``, ``ModelCondition``, and supporting enums
for the SQLite-backed component catalog.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComponentKind(enum.StrEnum):
    """Physical component type."""

    INDUCTOR = "inductor"
    CAPACITOR = "capacitor"
    RESISTOR = "resistor"


class ModelTier(enum.StrEnum):
    """Model fidelity tier, ordered highest-first."""

    MEASURED = "measured"
    PARAMETRIC = "parametric"
    IDEAL = "ideal"


# Tier rank for comparison (higher = better fidelity)
_TIER_RANK: dict[ModelTier, int] = {
    ModelTier.MEASURED: 3,
    ModelTier.PARAMETRIC: 2,
    ModelTier.IDEAL: 1,
}


def tier_rank(tier: ModelTier) -> int:
    """Return integer rank for tier comparison (higher = better)."""
    return _TIER_RANK[tier]


class ModelOrigin(enum.StrEnum):
    """Origin/provenance of a model record."""

    VENDOR_TOUCHSTONE = "vendor_touchstone"
    LAB_MEASUREMENT = "lab_measurement"
    VENDOR_PARAMETRIC = "vendor_parametric"
    DERIVED_PARAMETRIC = "derived_parametric"
    IDEAL = "ideal"


class FallbackPolicy(enum.StrEnum):
    """Model tier fallback behavior."""

    STRICT = "strict"
    ALLOW_LOWER_TIER = "allow_lower_tier"


# ---------------------------------------------------------------------------
# LibraryComponent
# ---------------------------------------------------------------------------


@dataclass
class LibraryComponent:
    """A physical component in the library (one row in ``components``)."""

    id: str
    kind: ComponentKind
    vendor: str
    part_number: str
    package: str = ""
    description: str = ""

    # Nominal value in SI (H / F / Ω)
    value_nom: float = 0.0
    value_tol_frac: float | None = None

    # Ratings
    voltage_max_v: float | None = None
    current_max_a: float | None = None
    current_sat_a: float | None = None
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    power_max_w: float | None = None

    # Availability snapshot
    stock_status: str | None = None  # 'in_stock'|'low_stock'|'out_of_stock'
    stock_ts: str | None = None  # UTC ISO-8601

    datasheet_url: str | None = None

    # Provenance
    import_source: str = ""
    import_sha256: str | None = None
    import_ts: str = ""
    content_sha256: str = ""
    user_notes: str = ""

    def compute_content_sha256(self) -> str:
        """Compute deterministic SHA-256 of canonical component metadata.

        This is used for idempotent duplicate detection.
        """
        canonical = (
            self.vendor,
            self.part_number,
            self.kind.value,
            self.value_nom,
            self.value_tol_frac,
            self.package,
            self.description,
            self.voltage_max_v,
            self.current_max_a,
            self.current_sat_a,
            self.temp_min_c,
            self.temp_max_c,
            self.power_max_w,
        )
        return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ModelCondition
# ---------------------------------------------------------------------------


@dataclass
class ModelCondition:
    """A model record attached to a component (one row in ``model_conditions``).

    Multiple records per component per tier are allowed (different frequency
    spans, temperature, bias, fixture, or provenance).
    """

    id: str
    component_id: str
    model_tier: ModelTier
    model_origin: ModelOrigin

    # File store reference (measured / Touchstone models)
    model_file_sha256: str | None = None
    model_file_ext: str | None = None  # '.s1p', '.s2p', etc.
    n_ports: int | None = None

    # Fixture semantics for multiport → one-port extraction
    fixture_type: str | None = None  # 'shunt'|'series'|'floating_dut'
    fixture_port_z: int | None = None
    fixture_port_gnd: int | None = None

    # Parametric model params (dict decoded from JSON)
    parametric_params: dict[str, Any] | None = None

    # RF conditions (model validity, not ratings)
    srf_hz: float | None = None
    q_at_f_hz: float | None = None
    q_value: float | None = None
    esr_ohm: float | None = None
    validity_hz_lo: float | None = None
    validity_hz_hi: float | None = None

    # Measurement conditions
    measurement_temp_c: float | None = None
    measurement_bias_v: float | None = None

    variant_label: str = ""
    import_ts: str = ""

    def validity_hz(self) -> tuple[float, float] | None:
        """Return (lo, hi) or None."""
        if self.validity_hz_lo is not None and self.validity_hz_hi is not None:
            return (self.validity_hz_lo, self.validity_hz_hi)
        return None

    def parametric_params_json(self) -> str | None:
        """Serialize parametric_params to JSON string."""
        if self.parametric_params is None:
            return None
        return json.dumps(self.parametric_params, sort_keys=True)

    @staticmethod
    def parametric_params_from_json(s: str | None) -> dict[str, Any] | None:
        """Deserialize JSON string to dict."""
        if s is None:
            return None
        return json.loads(s)  # type: ignore[no-any-return]
