"""Model bridge: construct OnePortModel from library metadata (Prompt 08).

Implements explicit tier selection with STRICT / ALLOW_LOWER_TIER fallback
policy.  Never silently downgrades when a higher-fidelity model exists but
is invalid for the requested conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foster_eom.catalog.component import (
    ComponentKind,
    FallbackPolicy,
    ModelCondition,
    ModelTier,
    tier_rank,
)
from foster_eom.catalog.fixture import FixtureSpec, FixtureType, extract_one_port
from foster_eom.models.base import OnePortModel
from foster_eom.models.components import (
    IdealCapacitor,
    IdealInductor,
    IdealResistor,
    LumpedLossyCapacitor,
    LumpedLossyInductor,
)

if TYPE_CHECKING:
    from foster_eom.catalog.component import LibraryComponent
    from foster_eom.catalog.file_store import ContentAddressedStore


class ModelNotAvailableError(Exception):
    """Raised when no eligible model is available for the requested conditions."""


def build_model(
    component: LibraryComponent,
    conditions: list[ModelCondition],
    file_store: ContentAddressedStore | None = None,
    *,
    required_tier: ModelTier | None = None,
    freq_range: tuple[float, float] | None = None,
    fallback: FallbackPolicy = FallbackPolicy.STRICT,
) -> OnePortModel:
    """Build a OnePortModel from library component + model conditions.

    Parameters
    ----------
    component : LibraryComponent
        The component metadata.
    conditions : list[ModelCondition]
        All model conditions for this component.
    file_store : ContentAddressedStore | None
        File store for retrieving measured model files.
    required_tier : ModelTier | None
        If set, only use this specific tier.
    freq_range : tuple[float, float] | None
        Required frequency coverage (both endpoints must be covered).
    fallback : FallbackPolicy
        STRICT (default): error if higher-fidelity model exists but is
        invalid. ALLOW_LOWER_TIER: use best available.

    Returns
    -------
    OnePortModel

    Raises
    ------
    ModelNotAvailableError
        If no eligible model is available.
    """
    if not conditions:
        raise ModelNotAvailableError(
            f"No model conditions found for component '{component.vendor}/{component.part_number}'."
        )

    # 1. Filter by required_tier if set
    candidates = list(conditions)
    if required_tier is not None:
        candidates = [c for c in candidates if c.model_tier == required_tier]
        if not candidates:
            available = sorted({c.model_tier.value for c in conditions})
            raise ModelNotAvailableError(
                f"Required tier '{required_tier.value}' not available for "
                f"'{component.vendor}/{component.part_number}'. "
                f"Available tiers: {available}"
            )

    # 2. Filter by freq_range eligibility
    def covers_range(mc: ModelCondition) -> bool:
        if freq_range is None:
            return True
        vr = mc.validity_hz()
        if vr is None:
            # Ideal models with no validity range are treated as covering all
            return mc.model_tier == ModelTier.IDEAL
        return vr[0] <= freq_range[0] and vr[1] >= freq_range[1]

    eligible = [c for c in candidates if covers_range(c)]

    # 3. STRICT check
    if fallback == FallbackPolicy.STRICT and freq_range is not None:
        # Check if any higher-tier models exist but were excluded
        excluded = [c for c in candidates if not covers_range(c)]
        excluded_tiers = {c.model_tier for c in excluded}
        eligible_tiers = {c.model_tier for c in eligible}

        for exc_tier in excluded_tiers:
            if eligible_tiers and all(tier_rank(exc_tier) > tier_rank(et) for et in eligible_tiers):
                exc_reasons = []
                for c in excluded:
                    if c.model_tier == exc_tier:
                        vr = c.validity_hz()
                        vr_str = f"[{vr[0]:.0f}, {vr[1]:.0f}] Hz" if vr else "None"
                        exc_reasons.append(
                            f"  {c.model_origin.value}/{c.variant_label or 'default'}: "
                            f"validity={vr_str}"
                        )
                raise ModelNotAvailableError(
                    f"STRICT fallback policy: higher-fidelity "
                    f"'{exc_tier.value}' model(s) exist for "
                    f"'{component.vendor}/{component.part_number}' but do not "
                    f"cover requested range "
                    f"[{freq_range[0]:.0f}, {freq_range[1]:.0f}] Hz:\n"
                    + "\n".join(exc_reasons)
                    + f"\nUse FallbackPolicy.ALLOW_LOWER_TIER to permit "
                    f"fallback to '{next(iter(eligible_tiers)).value if eligible_tiers else 'none'}'."
                )

    if not eligible:
        available_info = []
        for c in candidates:
            vr = c.validity_hz()
            vr_str = f"[{vr[0]:.0f}, {vr[1]:.0f}] Hz" if vr else "unlimited"
            available_info.append(
                f"  {c.model_tier.value}/{c.model_origin.value}: validity={vr_str}"
            )
        fr_str = f"[{freq_range[0]:.0f}, {freq_range[1]:.0f}] Hz" if freq_range else "any"
        raise ModelNotAvailableError(
            f"No eligible model for '{component.vendor}/{component.part_number}' "
            f"covering {fr_str}. Available:\n" + "\n".join(available_info)
        )

    # 4. Sort by tier rank (highest first), then narrowest validity span
    def sort_key(mc: ModelCondition) -> tuple[int, float]:
        rank = -tier_rank(mc.model_tier)  # negative for descending
        vr = mc.validity_hz()
        span = (vr[1] - vr[0]) if vr else float("inf")
        return (rank, span)

    eligible.sort(key=sort_key)
    selected = eligible[0]

    # 5. Construct the model
    return _construct_model(component, selected, file_store)


def _construct_model(
    component: LibraryComponent,
    mc: ModelCondition,
    file_store: ContentAddressedStore | None,
) -> OnePortModel:
    """Construct a OnePortModel from a selected ModelCondition."""

    if mc.model_tier == ModelTier.IDEAL:
        return _build_ideal(component.kind, component.value_nom)

    if mc.model_tier == ModelTier.PARAMETRIC:
        return _build_parametric(component, mc)

    if mc.model_tier == ModelTier.MEASURED:
        return _build_measured(component, mc, file_store)

    raise ModelNotAvailableError(f"Unknown model tier: {mc.model_tier}")


def _build_ideal(kind: ComponentKind, value: float) -> OnePortModel:
    if kind == ComponentKind.INDUCTOR:
        return IdealInductor(l_h=value)
    if kind == ComponentKind.CAPACITOR:
        return IdealCapacitor(c_f=value)
    if kind == ComponentKind.RESISTOR:
        return IdealResistor(r_ohm=value)
    raise ModelNotAvailableError(f"Cannot build ideal model for kind '{kind.value}'.")


def _build_parametric(
    component: LibraryComponent,
    mc: ModelCondition,
) -> OnePortModel:
    params = mc.parametric_params or {}
    validity = mc.validity_hz()

    if component.kind == ComponentKind.INDUCTOR:
        return LumpedLossyInductor(
            l_h=component.value_nom,
            r_dcr_ohm=params.get("dcr_ohm", 0.0),
            c_par_f=params.get("c_par_f", 0.0),
            validity_hz=validity,
        )
    if component.kind == ComponentKind.CAPACITOR:
        return LumpedLossyCapacitor(
            c_f=component.value_nom,
            r_esr_ohm=params.get("esr_ohm", 0.0),
            l_esl_h=params.get("esl_h", 0.0),
            validity_hz=validity,
        )
    if component.kind == ComponentKind.RESISTOR:
        # For resistors, parametric = ideal (no parasitic model yet)
        return IdealResistor(r_ohm=component.value_nom)

    raise ModelNotAvailableError(
        f"Cannot build parametric model for kind '{component.kind.value}'."
    )


def _build_measured(
    component: LibraryComponent,
    mc: ModelCondition,
    file_store: ContentAddressedStore | None,
) -> OnePortModel:
    if file_store is None:
        raise ModelNotAvailableError("File store required for measured models.")
    if mc.model_file_sha256 is None or mc.model_file_ext is None:
        raise ModelNotAvailableError(
            f"Model condition '{mc.id}' has tier=measured but no file reference."
        )

    stored_path = file_store.retrieve(mc.model_file_sha256, mc.model_file_ext)

    # If multiport, use fixture extraction
    if mc.n_ports is not None and mc.n_ports > 1:
        if mc.fixture_type is None:
            raise ModelNotAvailableError(
                f"FixtureSpec required for {mc.n_ports}-port model '{mc.model_file_sha256[:12]}…'."
            )
        fixture = FixtureSpec(
            fixture_type=FixtureType(mc.fixture_type),
            port_z=mc.fixture_port_z or 0,
            port_gnd=mc.fixture_port_gnd or 1,
        )
        return extract_one_port(stored_path, fixture)

    # One-port: direct extraction
    return extract_one_port(stored_path, fixture=None)
