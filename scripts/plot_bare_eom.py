#!/usr/bin/env python3
"""Dev script: solve and plot the synthetic mBVD EOM directly from 50 Ω.

Usage::

    python scripts/plot_bare_eom.py

Produces plots of |Z_in|, |Γ|, |V_EOM| vs frequency.
All plotting is outside the numerical core.
"""

from __future__ import annotations

import numpy as np

from foster_eom.circuit import (
    CircuitGraph,
    Element,
    ElementKind,
    Node,
    Port,
    solve_circuit,
)
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.models import create_synthetic_mbvd


def main() -> None:
    # Build the EOM model
    eom = create_synthetic_mbvd()
    print(f"Synthetic mBVD EOM: {eom.metadata()}")

    # Build graph: source → EOM → ground
    g = CircuitGraph(
        ground_node_id="gnd",
        input_port=Port("in", "gnd"),
        eom_element_id="eom",
    )
    g.add_node(Node("gnd", is_ground=True))
    g.add_node(Node("in"))
    g.add_element(
        Element(
            id="eom",
            kind=ElementKind.ONE_PORT_MODEL,
            node_pos="in",
            node_neg="gnd",
            model=eom,
            symbolic_role="eom",
        )
    )

    # Source: 1 Vrms, 50 Ω
    source = SourceSpec(
        mode=SourceMode.THEVENIN,
        thevenin_vrms=1.0,
        z_source_real_ohm=50.0,
        z_ref_ohm=50.0,
    )

    # Sweep 1-20 MHz
    freqs = np.linspace(1e6, 20e6, 2000)
    solutions = solve_circuit(g, source, freqs)

    # Extract
    f_mhz = freqs / 1e6
    z_in_mag = np.array([abs(s.z_in) if s.z_in else np.nan for s in solutions])
    gamma_mag = np.array([abs(s.gamma) if s.gamma else np.nan for s in solutions])
    v_eom_mag = np.array([abs(s.v_eom) if s.v_eom else np.nan for s in solutions])
    pbal_ok = np.array([s.power_balance_ok for s in solutions])

    # Report power balance
    n_fail = np.sum(~pbal_ok)
    print(f"Power balance: {np.sum(pbal_ok)}/{len(solutions)} OK, {n_fail} failed")
    if n_fail > 0:
        for s in solutions:
            if not s.power_balance_ok:
                print(f"  f={s.f_hz / 1e6:.3f} MHz: residual={s.power_balance_residual}")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    _fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].semilogy(f_mhz, z_in_mag)
    axes[0].set_ylabel("|Z_in| (Ω)")
    axes[0].set_title("Bare EOM: 50 Ω source, no matching network")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_mhz, gamma_mag)
    axes[1].set_ylabel("|Γ|")
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f_mhz, v_eom_mag * 1e3)
    axes[2].set_ylabel("|V_EOM| (mV)")
    axes[2].set_xlabel("Frequency (MHz)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bare_eom_response.png", dpi=150)
    print("Saved bare_eom_response.png")
    plt.show()


if __name__ == "__main__":
    main()
