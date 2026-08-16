"""Factory for EOM models."""

from __future__ import annotations

from foster_eom.domain.eom import EOMModelSpec, EOMModelType
from foster_eom.models.base import EOMModel
from foster_eom.models.eom_ideal import IdealCapacitorEOM
from foster_eom.models.eom_lossy import LossyCapacitorEOM
from foster_eom.models.eom_mbvd import MBVDModel


def build_eom_model(spec: EOMModelSpec) -> EOMModel:
    """Instantiate a runtime EOM model from a schema specification.

    Parameters
    ----------
    spec : EOMModelSpec
        The validated EOM model specification.

    Returns
    -------
        A concrete EOMModel instance.

    Raises
    ------
    NotImplementedError
        If the model type is not yet supported.
    ValueError
        If required tabular data cannot be loaded.
    """
    if spec.model_type == EOMModelType.IDEAL_CAPACITOR:
        assert spec.c0_f is not None  # Schema validation guarantees this
        return IdealCapacitorEOM(
            c0_f=spec.c0_f,
            extrapolation_policy=spec.extrapolation_policy,
        )

    if spec.model_type == EOMModelType.LOSSY_CAPACITOR:
        assert spec.c0_f is not None
        return LossyCapacitorEOM(
            c0_f=spec.c0_f,
            rs_ohm=spec.rs_ohm or 0.0,
            ls_h=spec.ls_h or 0.0,
            g0_s=spec.g0_s or 0.0,
            validity_hz=spec.validity_hz,
            extrapolation_policy=spec.extrapolation_policy,
        )

    if spec.model_type == EOMModelType.MBVD:
        assert spec.c0_f is not None
        return MBVDModel(
            c0_f=spec.c0_f,
            g0_s=spec.g0_s or 0.0,
            rs_ohm=spec.rs_ohm or 0.0,
            ls_h=spec.ls_h or 0.0,
            motional_branches=list(spec.motional_branches),
            validity_hz=spec.validity_hz,
            extrapolation_policy=spec.extrapolation_policy,
        )

    if spec.model_type == EOMModelType.TABULAR:
        # Prompt 02 note: Tabular data reading from file is deferred to Prompt 07.
        # For now, we mock the load or raise.
        raise NotImplementedError("Tabular file parsing is deferred to Prompt 07.")

    if spec.model_type == EOMModelType.RATIONAL:
        raise NotImplementedError("Rational fitted models are deferred.")

    raise NotImplementedError(f"Unsupported EOM model type: {spec.model_type}")
