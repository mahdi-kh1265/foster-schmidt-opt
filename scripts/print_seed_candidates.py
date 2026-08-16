#!/usr/bin/env python
"""Print seed candidates for a design spec.

Usage:
    python scripts/print_seed_candidates.py path/to/design_spec.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Print Foster seed candidates")
    parser.add_argument("design_spec", type=Path, help="Path to design spec YAML")
    parser.add_argument("--max-seeds", type=int, default=10, help="Max seeds to report")
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.05,
        help="Max allowable relative match error (default 0.05)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1000,
        help="Sign search beam width (default 1000)",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=256,
        help="Max sign patterns (default 256)",
    )
    args = parser.parse_args()

    # Lazy imports
    from foster_eom.domain.component import ContinuousLimits
    from foster_eom.foster.seed import SignSearchOptions, generate_seeds
    from foster_eom.models.factory import build_eom_model
    from foster_eom.persistence.yaml_io import load_project

    # Load project
    spec = load_project(args.design_spec)
    print(f"Project: {spec.project.name}")
    print(f"Schema:  {spec.schema_version}")

    # Build EOM model
    eom = build_eom_model(spec.eom)

    # Source spec
    source = spec.source

    # Target frequencies
    f_targets = np.array(
        [t.frequency_hz for t in spec.frequencies.targets],
        dtype=np.float64,
    )
    print(f"Targets: {len(f_targets)} frequencies")
    for t in spec.frequencies.targets:
        print(f"  {t.label}: {t.frequency_hz / 1e6:.3f} MHz")

    # Continuous limits
    comp_spec = spec.components
    if comp_spec and comp_spec.continuous_limits:
        limits = comp_spec.continuous_limits
    else:
        limits = ContinuousLimits()

    # R_match
    r_match = source.z_ref_ohm if source.z_ref_ohm else 50.0
    print(f"R_match: {r_match:.1f} ohm")

    # Generate seeds
    print(f"\nSearching seeds (beam_width={args.beam_width}, max_patterns={args.max_patterns})...")
    result = generate_seeds(
        r_match_ohm=r_match,
        source_spec=source,
        eom_model=eom,
        f_targets_hz=f_targets,
        topo_spec=spec.topology,
        component_limits=limits,
        match_tolerance=args.match_tolerance,
        max_seeds=args.max_seeds,
        sign_search_options=SignSearchOptions(
            beam_width=args.beam_width,
            max_patterns=args.max_patterns,
        ),
    )

    # Print diagnostics
    d = result.diagnostics
    print(f"\n{'=' * 60}")
    print("DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"Orientations tried:    {d.n_orientation_attempts}")
    print(f"Sign patterns found:   {d.n_sign_patterns}")
    print(f"Topologies enumerated: {d.n_topologies}")
    print(f"Pole layout pairs:     {d.n_pole_layout_pairs}")
    print(f"Solver attempts:       {d.n_solver_attempts}")
    print(f"MNA attempts:          {d.n_mna_attempts}")
    print(f"Sign search exhaustive: {d.sign_search_exhaustive}")
    print(f"Sign search truncated:  {d.sign_search_truncated}")

    if d.rejection_counts:
        print("\nRejection counts:")
        for code, count in sorted(d.rejection_counts.items(), key=lambda x: -x[1]):
            # code is an enum, we can print code.name
            code_name = getattr(code, "name", str(code))
            print(f"  {code_name}: {count}")

    if d.representative_failures:
        print("\nRepresentative failures:")
        # Group by code for nicer display
        from collections import defaultdict

        grouped = defaultdict(list)
        for f in d.representative_failures:
            grouped[f.code].append(f)

        for code, failures_for_code in grouped.items():
            code_name = getattr(code, "name", str(code))
            print(f"  {code_name}:")
            for f in failures_for_code:
                msg = f.reason if f.reason else ""
                print(f"    - {msg}")

    # Print seeds
    print(f"\n{'=' * 60}")
    print(f"SEED CANDIDATES: {len(result.seeds)}")
    print(f"{'=' * 60}")

    for i, seed in enumerate(result.seeds):
        print(f"\n--- Seed {i + 1} ---")
        print(f"  Orientation: {seed.orientation.value}")
        print(f"  Sign pattern: {seed.sign_pattern.signs}")
        print(f"  Branch1 realization: {seed.sign_pattern.branch1_realization.value}")
        print(f"  Branch2 realization: {seed.sign_pattern.branch2_realization.value}")
        print(
            f"  Topology: b1_cells={seed.topology.branch1_cells}, b2_cells={seed.topology.branch2_cells}"
        )
        print(f"  Reactive elements: {seed.topology.n_reactive}")

        v = seed.validation
        print(f"  Max match error: {v.max_match_error:.6e}")
        print(f"  RMS match error: {v.rms_match_error:.6e}")

        for j, f in enumerate(f_targets):
            print(
                f"    f={f / 1e6:.3f}MHz: S11={v.s11_db_at_targets[j]:.1f}dB, "
                f"|gamma|={abs(v.gamma_at_targets[j]):.4f}, "
                f"err={v.match_error_at_targets[j]:.4e}"
            )

        if seed.branch1_components is not None:
            c = seed.branch1_components
            print("  Branch1 components:")
            if c.c0_f:
                print(f"    C0 = {c.c0_f * 1e12:.2f} pF")
            for ci, cell in enumerate(c.cells):
                print(
                    f"    Cell {ci + 1}: L={cell.l_h * 1e6:.4f} uH, "
                    f"C={cell.c_f * 1e12:.2f} pF, "
                    f"f_pole={cell.f_pole_hz / 1e6:.3f} MHz"
                )
            if c.l_inf_h:
                print(f"    L_inf = {c.l_inf_h * 1e6:.4f} uH")

        if seed.branch2_components is not None:
            c = seed.branch2_components
            print("  Branch2 components:")
            if c.c0_f:
                print(f"    C0 = {c.c0_f * 1e12:.2f} pF")
            for ci, cell in enumerate(c.cells):
                print(
                    f"    Cell {ci + 1}: L={cell.l_h * 1e6:.4f} uH, "
                    f"C={cell.c_f * 1e12:.2f} pF, "
                    f"f_pole={cell.f_pole_hz / 1e6:.3f} MHz"
                )
            if c.l_inf_h:
                print(f"    L_inf = {c.l_inf_h * 1e6:.4f} uH")

    if not result.seeds:
        print("\n  No seeds accepted. Check rejection diagnostics above.")

    print()


if __name__ == "__main__":
    main()
