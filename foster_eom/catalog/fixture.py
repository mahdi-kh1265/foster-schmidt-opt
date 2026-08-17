"""Multiport Touchstone → one-port extraction via fixture semantics (Prompt 08).

Supports three fixture topologies for extracting the two-terminal impedance
of a DUT from a multiport S-parameter measurement.  Uses scikit-rf for all
network transformations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foster_eom.models.base import OnePortModel
from foster_eom.models.components import TabularImpedanceComponent


class FixtureType(enum.StrEnum):
    """Fixture topology for multiport → one-port extraction."""

    SHUNT = "shunt"
    SERIES = "series"
    FLOATING_DUT = "floating_dut"


@dataclass(frozen=True)
class FixtureSpec:
    """Specification for extracting one-port impedance from a multiport network.

    Parameters
    ----------
    fixture_type : FixtureType
        Topology of the measurement fixture.
    port_z : int
        Port index carrying the DUT impedance (0-indexed).
    port_gnd : int
        Port index used as ground reference (0-indexed).
    """

    fixture_type: FixtureType
    port_z: int = 0
    port_gnd: int = 1


def extract_one_port(
    touchstone_path: Path,
    fixture: FixtureSpec | None = None,
) -> OnePortModel:
    """Extract a one-port impedance model from a Touchstone file.

    For ``.s1p`` files, no fixture is needed — the data is already one-port.
    For ``.s2p`` and higher, a ``FixtureSpec`` is required.

    Parameters
    ----------
    touchstone_path : Path
        Path to the Touchstone file.
    fixture : FixtureSpec | None
        Fixture specification for multiport extraction.

    Returns
    -------
    OnePortModel
        Tabular impedance model.
    """
    import skrf  # type: ignore[import-untyped]

    ntwk = skrf.Network(str(touchstone_path))
    f_hz = np.asarray(ntwk.f, dtype=np.float64)

    if ntwk.nports == 1:
        # One-port: use P07 load path via TabularImpedanceComponent
        z0 = ntwk.z0[:, 0].real
        z_ref = float(z0[0])
        s11 = ntwk.s[:, 0, 0]
        z = z_ref * (1.0 + s11) / (1.0 - s11)
        return TabularImpedanceComponent(
            f_hz=f_hz,
            z_complex=z.astype(np.complex128),
        )

    if fixture is None:
        raise ValueError(
            f"FixtureSpec required for {ntwk.nports}-port Touchstone file '{touchstone_path.name}'."
        )

    if fixture.port_z >= ntwk.nports or fixture.port_gnd >= ntwk.nports:
        raise ValueError(
            f"Fixture ports ({fixture.port_z}, {fixture.port_gnd}) exceed "
            f"network port count ({ntwk.nports})."
        )

    z_dut = _extract_z(ntwk, fixture, f_hz)

    return TabularImpedanceComponent(
        f_hz=f_hz,
        z_complex=z_dut,
    )


def _extract_z(
    ntwk: object,
    fixture: FixtureSpec,
    f_hz: np.ndarray,
) -> np.ndarray:
    """Extract DUT impedance from multiport network using fixture semantics.

    All conversions use scikit-rf's Z-parameter properties for proper
    reference impedance handling.
    """
    # Get Z-parameter matrix from scikit-rf
    z_matrix = ntwk.z  # type: ignore[attr-defined]  # shape (n_freq, n_ports, n_ports)

    p = fixture.port_z
    g = fixture.port_gnd

    if fixture.fixture_type == FixtureType.SHUNT:
        # Shunt DUT: component connected between the common node and ground,
        # with both ports connected to the common node.
        # Z-parameters: Z11 = Z12 = Z21 = Z22 = Z_dut
        # Use Z11 for extraction.
        z_dut = z_matrix[:, p, p]

    elif fixture.fixture_type == FixtureType.SERIES:
        # Series DUT: component in series between port_z and port_gnd.
        # Z_dut = Z[p,p] + Z[g,g] - Z[p,g] - Z[g,p]
        z_dut = z_matrix[:, p, p] + z_matrix[:, g, g] - z_matrix[:, p, g] - z_matrix[:, g, p]

    elif fixture.fixture_type == FixtureType.FLOATING_DUT:
        # Floating two-terminal DUT between port_z and port_gnd.
        # Z_dut = (Z[p,p]*Z[g,g] - Z[p,g]*Z[g,p]) / Z[g,p]
        num = z_matrix[:, p, p] * z_matrix[:, g, g] - z_matrix[:, p, g] * z_matrix[:, g, p]
        den = z_matrix[:, g, p]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_dut = np.where(
                np.abs(den) > 1e-30,
                num / den,
                np.complex128(np.inf + 0j),
            )

    else:
        raise ValueError(f"Unknown fixture type: {fixture.fixture_type}")

    return np.asarray(z_dut, dtype=np.complex128)
