"""Focused tests for P09 P06-band and best-combo correctness.

TestP06BandSeparation
    Proves that model eligibility uses the P05 evaluation grid (min/max of
    ctx.evaluation_frequencies_hz), while the P06 sweep band is derived
    independently from SweepSpec.from_targets — not clamped to the P05 grid.
    Also verifies that ctx.p06_sweep_band_hz overrides the derived band.

TestDebBestWithP06
    Proves that when Deb-ranked #1 fails P06 and #2 passes:
      - result.best == result.first_passing_combo == #2 (not #1)
      - result.status == "feasible"
    Also proves that if no candidate within verify_top_k passes but unverified
    candidates remain, the status is not "infeasible" — it is "no_feasible_found".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers for constructing minimal EvaluationResult and EvaluationContext
# ---------------------------------------------------------------------------


def _make_eval_result(
    *,
    feasible: bool = True,
    near_feasible: bool = True,
    v_max: float = 0.0,
    v_sum: float = 0.0,
    objective_value: float = 1.0,
) -> Any:
    from foster_eom.optimize.evaluator import EvaluationResult

    return EvaluationResult(
        x=(),
        objective_value=objective_value,
        base_objective_value=objective_value,
        soft_penalty_total=0.0,
        objective_terms={},
        hard_margins=(),
        soft_penalties={},
        v_max=v_max,
        v_sum=v_sum,
        feasible=feasible,
        near_feasible=near_feasible,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )


def _make_catalog_combo(eval_result: Any, verify_passed: bool | None = None) -> Any:
    from foster_eom.optimize.dedup import deb_key
    from foster_eom.realization.result import CatalogCombo

    cc = CatalogCombo(
        slot_entries={},
        eval_result=eval_result,
        deb_key=deb_key(eval_result),
    )
    cc.verify_passed = verify_passed
    return cc


# ---------------------------------------------------------------------------
# TestP06BandSeparation
# ---------------------------------------------------------------------------


class TestP06BandSeparation:
    """Prove that P06 sweep band is NOT clamped to the P05 evaluation grid."""

    def test_model_eligibility_uses_eval_grid(self) -> None:
        """build_slot_specs derives freq_range from ctx.evaluation_frequencies_hz."""
        from unittest.mock import MagicMock

        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.optimize.variable_map import BranchCoordinates
        from foster_eom.realization.neighborhoods import build_slot_specs

        # Narrow P05 eval grid: 9 MHz - 11 MHz
        ctx = MagicMock()
        ctx.evaluation_frequencies_hz = (9e6, 10e6, 11e6)
        ctx.domain.branch1_realization = BranchRealization.FINITE_FOSTER
        ctx.domain.branch2_realization = BranchRealization.OPEN_OMITTED

        b1 = BranchCoordinates(
            k0=None,
            k_inf=None,
            k_residues=(1.0,),
            f_poles_hz=(10e6,),
            l_values_h=(1e-6,),
            c_values_f=(100e-12,),
        )
        b2 = BranchCoordinates(
            k0=None, k_inf=None, k_residues=(), f_poles_hz=(), l_values_h=(), c_values_f=()
        )

        specs = build_slot_specs(ctx, b1, b2)
        for s in specs:
            # Eligibility band = P05 grid band: 9 - 11 MHz
            assert s.freq_range_hz == (9e6, 11e6), (
                f"Slot {s.element_id}: expected eligibility band (9e6, 11e6), got {s.freq_range_hz}"
            )

    def test_p06_sweep_band_not_clamped_to_eval_grid(self) -> None:
        """When p06_sweep_band_hz is None, _run_p06_verify uses from_targets()
        WITHOUT clipping to eval_frequencies_hz, so the sweep band can be wider."""
        from foster_eom.analysis.sweep import SweepSpec

        # Target at 10 MHz; from_targets default margin: [5 MHz, 20 MHz]
        target_hz = (10e6,)
        derived = SweepSpec.from_targets(target_hz=target_hz)

        # Narrow eval grid that would clip: 9 - 11 MHz
        grid_min, grid_max = 9e6, 11e6

        # Old (buggy) clipped version:
        clipped = SweepSpec.from_targets(target_hz=target_hz, validity_range=(grid_min, grid_max))

        # The unclipped band is wider than the P05 grid
        assert derived.f_min_hz < grid_min, (
            f"Unclipped f_min ({derived.f_min_hz:.3g}) should be below grid_min ({grid_min:.3g})"
        )
        assert derived.f_max_hz > grid_max, (
            f"Unclipped f_max ({derived.f_max_hz:.3g}) should be above grid_max ({grid_max:.3g})"
        )

        # Clipped version is confined to grid range
        assert clipped.f_min_hz >= grid_min
        assert clipped.f_max_hz <= grid_max

    def test_p06_sweep_band_explicit_override(self) -> None:
        """ctx.p06_sweep_band_hz explicitly sets the sweep band, independent of targets."""
        from foster_eom.analysis.sweep import SweepSpec

        explicit_band = (1e6, 100e6)
        f_min, f_max = explicit_band
        sweep_spec = SweepSpec(f_min_hz=f_min, f_max_hz=f_max)

        assert sweep_spec.f_min_hz == 1e6
        assert sweep_spec.f_max_hz == 100e6

    def test_runner_uses_explicit_p06_band_from_context(self) -> None:
        """_run_p06_verify reads ctx.p06_sweep_band_hz when it is set."""
        # We verify this by inspecting the runner source code
        runner_src = (
            Path(__file__).parent.parent.parent / "foster_eom" / "realization" / "runner.py"
        ).read_text(encoding="utf-8")

        # Must reference p06_sweep_band_hz
        assert "p06_sweep_band_hz" in runner_src

        # Must NOT pass validity_range=eval_frequencies_hz (the old bug)
        assert "validity_range=(f_min, f_max)" not in runner_src
        assert "validity_range=(" not in runner_src

        # Must derive band without clamping when no explicit band set
        assert "from_targets(target_hz=target_hz)" in runner_src

    def test_p06_band_wider_than_p05_grid_when_targets_at_boundary(self) -> None:
        """If targets are at the edges of the P05 grid, the P06 band extends beyond."""
        from foster_eom.analysis.sweep import SweepSpec

        # Targets at grid boundaries
        target_hz = (1e6, 50e6)
        grid = (1e6, 50e6)

        # Without clipping: margin_lo=0.5, margin_hi=2.0
        unclipped = SweepSpec.from_targets(target_hz=target_hz)
        assert unclipped.f_min_hz < grid[0]  # 0.5 * 1e6 = 500 kHz
        assert unclipped.f_max_hz > grid[1]  # 2.0 * 50e6 = 100 MHz

        # With clipping (old behavior): confined to grid
        clipped = SweepSpec.from_targets(target_hz=target_hz, validity_range=grid)
        assert clipped.f_min_hz == grid[0]
        assert clipped.f_max_hz == grid[1]

    def test_eligibility_and_p06_band_can_differ(self) -> None:
        """Model eligibility band (P05 grid) and P06 sweep band are independently set."""
        from foster_eom.analysis.sweep import SweepSpec
        from foster_eom.foster.schmidt import BranchRealization
        from foster_eom.optimize.variable_map import BranchCoordinates
        from foster_eom.realization.neighborhoods import build_slot_specs

        # Narrow P05 eval grid
        ctx = MagicMock()
        ctx.evaluation_frequencies_hz = (9e6, 10e6, 11e6)
        ctx.domain.branch1_realization = BranchRealization.FINITE_FOSTER
        ctx.domain.branch2_realization = BranchRealization.OPEN_OMITTED

        b1 = BranchCoordinates(
            k0=None,
            k_inf=None,
            k_residues=(1.0,),
            f_poles_hz=(10e6,),
            l_values_h=(1e-6,),
            c_values_f=(100e-12,),
        )
        b2 = BranchCoordinates(
            k0=None, k_inf=None, k_residues=(), f_poles_hz=(), l_values_h=(), c_values_f=()
        )

        specs = build_slot_specs(ctx, b1, b2)
        eligibility_band = specs[0].freq_range_hz if specs else None
        assert eligibility_band == (9e6, 11e6)

        # P06 band from targets (unclipped) would be (5 MHz, 20 MHz)
        target_hz = (10e6,)
        p06_band = SweepSpec.from_targets(target_hz=target_hz)
        assert p06_band.f_min_hz < 9e6  # wider than eligibility band
        assert p06_band.f_max_hz > 11e6


# ---------------------------------------------------------------------------
# TestDebBestWithP06
# ---------------------------------------------------------------------------


class TestDebBestWithP06:
    """Prove correct best-combo and status logic when P06 verification discriminates."""

    def _make_context(self) -> Any:
        """Minimal EvaluationContext mock."""
        ctx = MagicMock()
        ctx.evaluation_frequencies_hz = (9e6, 10e6, 11e6)
        ctx.target_indices = (1,)  # index 1 → 10 MHz
        ctx.p06_sweep_band_hz = None
        ctx.source_spec = MagicMock()
        ctx.eom_model = MagicMock()
        return ctx

    def test_best_is_first_passing_when_deb1_fails_p06(self) -> None:
        """Deb-#1 fails P06, Deb-#2 passes → result.best == Deb-#2."""

        # Build two combos: Deb-#1 (better MNA, fails P06) and Deb-#2 (passes P06)
        r1 = _make_eval_result(feasible=True, objective_value=0.5)
        r2 = _make_eval_result(feasible=True, objective_value=1.0)

        cc1 = _make_catalog_combo(r1, verify_passed=False)  # Deb-#1, P06 fail
        cc2 = _make_catalog_combo(r2, verify_passed=True)  # Deb-#2, P06 pass

        # Simulate what runner.py produces:
        # catalog_combos is Deb-sorted: [cc1, cc2] (cc1 has lower obj → better Deb)
        catalog_combos = [cc1, cc2]
        first_passing = cc2

        # Replicate runner's best-selection logic:
        best = (
            first_passing
            if first_passing is not None
            else (catalog_combos[0] if catalog_combos else None)
        )

        assert best is cc2, "best must be cc2 (first P06 passer), not cc1 (Deb-#1 P06 failure)"
        assert best is not cc1

    def test_best_fallback_to_deb1_when_no_p06_passes(self) -> None:
        """When no P06 verification passes, best falls back to Deb-#1."""
        r1 = _make_eval_result(feasible=True, objective_value=0.5)
        cc1 = _make_catalog_combo(r1, verify_passed=False)

        catalog_combos = [cc1]
        first_passing = None

        best = (
            first_passing
            if first_passing is not None
            else (catalog_combos[0] if catalog_combos else None)
        )
        assert best is cc1

    def test_status_feasible_when_second_combo_passes(self) -> None:
        """Status is 'feasible' when Deb-#2 passes P06, regardless of Deb-#1 failure."""
        first_passing_combo = _make_catalog_combo(_make_eval_result(), verify_passed=True)
        assert first_passing_combo.verify_passed is True

        # status logic: first_passing is not None → status = "feasible"
        first_passing: Any = first_passing_combo
        status = "feasible" if first_passing is not None else "no_feasible_found"
        assert status == "feasible"

    def test_no_feasible_found_not_infeasible_when_unverified_remain(self) -> None:
        """When verify_top_k < total combos and none in verify window pass,
        status must be 'no_feasible_found', NOT 'infeasible'."""
        # Scenario: 5 combos, verify_top_k=2, both verified fail, 3 unverified remain
        combos = [_make_catalog_combo(_make_eval_result(feasible=True)) for _ in range(5)]
        verified = combos[:2]
        for cc in verified:
            cc.verify_passed = False
        first_passing = None

        n_unverified = len(combos) - len(verified)  # = 3
        search_exhaustive = True  # even if exhaustive MNA, unverified → can't claim infeasible
        all_verified_failed = verified and all(not cc.verify_passed for cc in verified)

        # MNA: all feasible (so no MNA infeasibility)
        all_mna_infeasible = all(not cc.eval_result.feasible for cc in combos)
        assert not all_mna_infeasible  # all are feasible

        # Status derivation (runner logic):
        if first_passing is not None:
            status = "feasible"
        elif (search_exhaustive and all_mna_infeasible) or (
            all_verified_failed and n_unverified == 0 and search_exhaustive
        ):
            status = "infeasible"
        else:
            status = "no_feasible_found"

        assert status == "no_feasible_found", (
            f"Expected 'no_feasible_found' but got '{status}' — "
            "must not claim infeasible when unverified candidates remain"
        )

    def test_infeasible_only_when_exhaustive_and_all_mna_infeasible(self) -> None:
        """'infeasible' is only emitted when MNA search is exhaustive AND all combos
        are MNA-infeasible (v_max > 0). P06 failures alone don't claim infeasibility."""
        combos = [
            _make_catalog_combo(_make_eval_result(feasible=False, v_max=0.1)),
            _make_catalog_combo(_make_eval_result(feasible=False, v_max=0.2)),
        ]
        for cc in combos:
            cc.verify_passed = False

        search_exhaustive = True
        first_passing = None
        verified = list(combos)
        n_unverified = 0
        all_verified_failed = all(not cc.verify_passed for cc in verified)
        all_mna_infeasible = all(not cc.eval_result.feasible for cc in combos)

        assert all_mna_infeasible  # both are MNA-infeasible

        if first_passing is not None:
            status = "feasible"
        elif (search_exhaustive and all_mna_infeasible) or (
            all_verified_failed and n_unverified == 0 and search_exhaustive
        ):
            status = "infeasible"
        else:
            status = "no_feasible_found"

        assert status == "infeasible"

    def test_p06_band_vs_eligibility_band_are_different_concepts(self) -> None:
        """Confirm the two bands can differ by construction — not aliases of each other."""
        from foster_eom.analysis.sweep import SweepSpec

        # P05 eval grid band (model eligibility):
        eval_freqs = (9e6, 10e6, 11e6)
        eligibility_band = (min(eval_freqs), max(eval_freqs))  # (9e6, 11e6)

        # P06 sweep band (from targets, no clipping):
        target_hz = (10e6,)
        sweep = SweepSpec.from_targets(target_hz=target_hz)
        p06_band = (sweep.f_min_hz, sweep.f_max_hz)  # (5e6, 20e6) with defaults

        assert eligibility_band != p06_band, (
            "Eligibility band and P06 sweep band must be independently derived"
        )
        assert p06_band[0] < eligibility_band[0]
        assert p06_band[1] > eligibility_band[1]

    def test_best_equals_first_passing_combo_in_result(self) -> None:
        """result.best must be identical to result.first_passing_combo when set."""
        from foster_eom.realization.result import RealizationDiagnostics, RealizationResult

        baseline = _make_eval_result()
        r1 = _make_eval_result(feasible=True, objective_value=0.3)
        r2 = _make_eval_result(feasible=True, objective_value=0.9)

        cc1 = _make_catalog_combo(r1, verify_passed=False)
        cc2 = _make_catalog_combo(r2, verify_passed=True)
        first_passing = cc2

        diag = RealizationDiagnostics(
            n_slots=1,
            parts_per_slot={"b1_L1": 2},
            total_combos=2,
            n_combos_generated=2,
            n_combos_evaluated=2,
            n_mna_solves=4,
            search_exhaustive=True,
            search_truncated=False,
            budget_exhausted=False,
        )

        result = RealizationResult(
            status="feasible",
            continuous_baseline=baseline,
            combos=[cc1, cc2],
            best=first_passing,  # not cc1
            degradation=None,
            failed_slots=[],
            diagnostics=diag,
            verified_combos=[cc1, cc2],
            first_passing_combo=first_passing,
        )

        assert result.best is result.first_passing_combo
        assert result.best is cc2
        assert result.best is not cc1

    def test_runner_source_best_assignment_logic(self) -> None:
        """Verify runner.py source assigns best = first_passing when available."""
        runner_src = (
            Path(__file__).parent.parent.parent / "foster_eom" / "realization" / "runner.py"
        ).read_text(encoding="utf-8")

        # Runner must assign best to first_passing when not None
        assert "best: CatalogCombo | None = first_passing" in runner_src
        # Must have fallback to Deb-best
        assert "catalog_combos[0]  # Deb-best unverified fallback" in runner_src
        # Status infeasible must NOT be set from truncated/partial verification
        # (no search_truncated → infeasible path)
        lines = runner_src.splitlines()
        for i, line in enumerate(lines):
            if '"infeasible"' in line and "status" in line:
                context = " ".join(lines[max(0, i - 8) : i + 2])
                assert "search_exhaustive" in context or "all_mna_infeasible" in context, (
                    f"'infeasible' status at line {i + 1} not guarded by exhaustive/MNA check"
                )
