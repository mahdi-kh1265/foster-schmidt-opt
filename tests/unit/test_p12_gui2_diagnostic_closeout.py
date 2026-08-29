"""P12-GUI2 Diagnostic Closeout Regression Tests.

Covers:
- A. Local-polish provenance for all four outcome states
- B. Human-readable constraint descriptor mapping
- C. Violated/closest-active constraint summary
- D. hard_207 identification via layout reconstruction
"""

from __future__ import annotations

from foster_eom.domain.constraints import (
    ConstraintSeverity,
    MatchConstraints,
    StressConstraints,
)
from foster_eom.domain.results import CandidateResult
from foster_eom.gui.view_models.optimize_vm import (
    CandidateDetailVM,
    format_polish_provenance,
)
from foster_eom.optimize.constraints import (
    ConstraintDescriptor,
    compile_constraint_layout,
    human_label,
    layout_human_labels,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_candidate(**overrides) -> CandidateResult:
    """Build a CandidateResult with defaults suitable for provenance testing."""
    defaults = dict(
        candidate_id="test_001",
        topology_id="topo_a",
        orientation="shunt",
        domain_id="dom1",
        branch1_realization="foster",
        branch2_realization="foster",
        branch1_cells=1,
        branch2_cells=0,
        feasible=False,
        near_feasible=True,
        v_max=0.0003,
        v_sum=0.0003,
        base_objective_value=0.934523,
        soft_penalty_total=0.0,
        numerical_status="ok",
        objective_terms={"total": 0.934523, "base": 0.934523, "soft_penalty": 0.0},
        constraint_margins={
            "gamma_f1000000Hz": 0.15,
            "r_max_f1000000Hz": 0.02,
            "r_min_f1000000Hz": 0.30,
            "x_bound_f1000000Hz": 0.25,
            "i_source_f1000000Hz": 0.80,
            "comp_L_hi_b1_m0": 0.10,
            "comp_L_lo_b1_m0": 0.05,
            "comp_C_hi_b1_m0": 0.45,
            "comp_C_lo_b1_m0": -0.0003,  # violated
        },
        seed_source="foster_schmidt_04b",
        local_polish_method="",
        local_polish_outcome="",
        local_polish_success=False,
        local_polish_iterations=0,
        local_polish_evaluations=0,
    )
    defaults.update(overrides)
    return CandidateResult(**defaults)


def _make_fast_layout(
    n_cells_b1: int = 3,
    n_cells_b2: int = 3,
    n_off_target: int = 198,
):
    """Compile a constraint layout matching the FAST smoke case topology.

    3 target frequencies, ~198 off-target coarse grid points, 3 cells per branch.
    This matches the ~209-row layout from the actual FAST run.
    """
    match_c = MatchConstraints(
        gamma_max=0.25,
        resistance_min_ohm=35.0,
        resistance_max_ohm=70.0,
        max_abs_reactance_ohm=20.0,
    )
    stress_c = StressConstraints(
        source_current_rms_max_a=0.5,
        off_target_eom_peak_rms_v=50.0,
    )

    target_freqs_hz = (1e6, 2e6, 3e6)
    # Build evaluation freq grid: targets + coarse grid
    import numpy as np

    sweep_min, sweep_max = 0.5e6, 6e6
    coarse_grid = np.linspace(sweep_min, sweep_max, 200)
    all_freqs = np.unique(np.concatenate([coarse_grid, np.array(target_freqs_hz)]))
    eval_freqs = tuple(float(f) for f in all_freqs)

    target_set = set(target_freqs_hz)
    target_indices = tuple(i for i, f in enumerate(eval_freqs) if f in target_set)
    off_target_indices = tuple(i for i, f in enumerate(eval_freqs) if f not in target_set)

    layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=target_freqs_hz,
        evaluation_frequencies_hz=eval_freqs,
        target_indices=target_indices,
        off_target_indices=off_target_indices,
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=n_cells_b1,
        n_cells_b2=n_cells_b2,
        z_ref_ohm=50.0,
    )
    return layout, eval_freqs


# ===========================================================================
# A. Local-Polish Provenance Tests
# ===========================================================================


class TestPolishProvenance:
    """Regression: all four provenance states produce non-blank display."""

    def test_polished_retained(self):
        """ANALYTICAL polish attempted and polished candidate retained."""
        c = _make_candidate(
            local_polish_method="trust-constr",
            local_polish_outcome="polished_retained",
            local_polish_success=True,
            local_polish_iterations=42,
        )
        lines = format_polish_provenance(c.local_polish_method, c.local_polish_outcome)
        assert len(lines) >= 1
        assert any("ANALYTICAL" in line for line in lines)
        assert any("trust-constr" in line for line in lines)
        assert any("polished candidate retained" in line for line in lines)

    def test_pre_polish_retained(self):
        """ANALYTICAL polish attempted but pre-polish candidate retained."""
        c = _make_candidate(
            local_polish_method="trust-constr",
            local_polish_outcome="pre_polish_retained",
            local_polish_success=True,
        )
        lines = format_polish_provenance(c.local_polish_method, c.local_polish_outcome)
        assert len(lines) >= 1
        assert any("ANALYTICAL" in line for line in lines)
        assert any("pre-polish candidate retained" in line for line in lines)

    def test_fd_fallback(self):
        """REFERENCE_FD fallback occurred."""
        c = _make_candidate(
            local_polish_method="trust-constr",
            local_polish_outcome="fd_fallback",
            local_polish_success=True,
        )
        lines = format_polish_provenance(c.local_polish_method, c.local_polish_outcome)
        assert len(lines) >= 1
        assert any("REFERENCE_FD" in line for line in lines)

    def test_not_selected(self):
        """Candidate was outside polish_top_k and was never polished."""
        c = _make_candidate(
            local_polish_method="",
            local_polish_outcome="not_selected",
        )
        lines = format_polish_provenance(c.local_polish_method, c.local_polish_outcome)
        assert len(lines) >= 1
        assert any("not selected for local polish" in line for line in lines)

    def test_all_outcomes_are_nonblank(self):
        """Every defined outcome produces at least one non-blank line."""
        outcomes = [
            ("trust-constr", "polished_retained"),
            ("trust-constr", "pre_polish_retained"),
            ("trust-constr", "fd_fallback"),
            ("", "not_selected"),
        ]
        for method, outcome in outcomes:
            lines = format_polish_provenance(method, outcome)
            assert len(lines) >= 1, f"Blank provenance for {outcome}"
            assert all(line.strip() for line in lines), f"Empty line for {outcome}"

    def test_legacy_blank_outcome_handled(self):
        """Empty outcome (legacy CandidateResult) does not crash."""
        lines = format_polish_provenance("", "")
        assert len(lines) >= 1
        # Should indicate no provenance rather than blank
        assert any("no provenance" in line.lower() for line in lines)


# ===========================================================================
# B. Constraint Descriptor Mapping Tests
# ===========================================================================


class TestConstraintDescriptorLabels:
    """Verify human_label() maps all constraint types correctly."""

    def test_gamma_label(self):
        d = ConstraintDescriptor(
            name="gamma_f1000000Hz",
            constraint_type="gamma",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (1e6,))
        assert "Γ" in label
        assert "1.00 MHz" in label

    def test_r_max_label(self):
        d = ConstraintDescriptor(
            name="r_max_f1000000Hz",
            constraint_type="r_max",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (1e6,))
        assert "R_in ≤ limit" in label
        assert "1.00 MHz" in label

    def test_r_min_label(self):
        d = ConstraintDescriptor(
            name="r_min_f1000000Hz",
            constraint_type="r_min",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (1e6,))
        assert "R_in ≥ limit" in label

    def test_x_bound_label(self):
        d = ConstraintDescriptor(
            name="x_bound_f1000000Hz",
            constraint_type="x_bound",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (1e6,))
        assert "|X_in|" in label

    def test_i_source_label(self):
        d = ConstraintDescriptor(
            name="i_source_f1000000Hz",
            constraint_type="i_source",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (1e6,))
        assert "I_source" in label

    def test_offtarget_label(self):
        d = ConstraintDescriptor(
            name="offtarget_veom_500000Hz",
            constraint_type="offtarget",
            frequency_scope="off_target",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (500e3,))
        assert "Off-target V_EOM" in label
        assert "500.0 kHz" in label

    def test_comp_L_hi_label(self):
        d = ConstraintDescriptor(
            name="comp_L_hi_b1_m0",
            constraint_type="comp_L_hi",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            branch=1,
            cell_index=0,
        )
        label = human_label(d)
        assert "L_b1[0]" in label
        assert "L_max" in label

    def test_comp_L_lo_label(self):
        d = ConstraintDescriptor(
            name="comp_L_lo_b2_m1",
            constraint_type="comp_L_lo",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            branch=2,
            cell_index=1,
        )
        label = human_label(d)
        assert "L_b2[1]" in label
        assert "L_min" in label

    def test_comp_C_hi_label(self):
        d = ConstraintDescriptor(
            name="comp_C_hi_b1_m0",
            constraint_type="comp_C_hi",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            branch=1,
            cell_index=0,
        )
        label = human_label(d)
        assert "C_b1[0]" in label
        assert "C_max" in label

    def test_comp_C_lo_label(self):
        d = ConstraintDescriptor(
            name="comp_C_lo_b1_m2",
            constraint_type="comp_C_lo",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            branch=1,
            cell_index=2,
        )
        label = human_label(d)
        assert "C_b1[2]" in label
        assert "C_min" in label

    def test_pole_sep_label(self):
        d = ConstraintDescriptor(
            name="pole_sep_b1_m0m1",
            constraint_type="pole_sep",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            branch=1,
            cell_index=0,
        )
        label = human_label(d)
        assert "Pole separation" in label
        assert "b1[0-1]" in label

    def test_custom_type_falls_back_to_name(self):
        d = ConstraintDescriptor(
            name="custom_q_constraint_f1MHz",
            constraint_type="custom",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
        )
        label = human_label(d)
        assert label == "custom_q_constraint_f1MHz"

    def test_frequency_formatting_ghz(self):
        d = ConstraintDescriptor(
            name="gamma_ghz",
            constraint_type="gamma",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (2.5e9,))
        assert "2.50 GHz" in label

    def test_frequency_formatting_hz(self):
        d = ConstraintDescriptor(
            name="gamma_hz",
            constraint_type="gamma",
            frequency_scope="all_targets",
            severity=ConstraintSeverity.HARD,
            freq_index=0,
        )
        label = human_label(d, (50.0,))
        assert "50.0 Hz" in label


class TestLayoutHumanLabels:
    """Verify layout_human_labels produces correct count and labels."""

    def test_layout_label_count_matches_descriptors(self):
        layout, eval_freqs = _make_fast_layout(n_cells_b1=1, n_cells_b2=1)
        labels = layout_human_labels(layout, eval_freqs)
        assert len(labels) == layout.n

    def test_layout_labels_are_nonempty(self):
        layout, eval_freqs = _make_fast_layout(n_cells_b1=2, n_cells_b2=2)
        labels = layout_human_labels(layout, eval_freqs)
        for label in labels:
            assert label, "Empty label found in layout"

    def test_layout_labels_contain_expected_types(self):
        layout, eval_freqs = _make_fast_layout(n_cells_b1=2, n_cells_b2=2)
        labels = layout_human_labels(layout, eval_freqs)
        all_text = " ".join(labels)
        assert "Γ" in all_text
        assert "R_in" in all_text
        assert "Off-target V_EOM" in all_text
        assert "L_b1" in all_text
        assert "C_b2" in all_text
        assert "Pole separation" in all_text

    def test_constraint_ordering_unchanged(self):
        """Layout descriptor order must be deterministic and unchanged."""
        layout1, _ = _make_fast_layout(n_cells_b1=2, n_cells_b2=2)
        layout2, _ = _make_fast_layout(n_cells_b1=2, n_cells_b2=2)
        names1 = [d.name for d in layout1.descriptors]
        names2 = [d.name for d in layout2.descriptors]
        assert names1 == names2, "Constraint ordering changed!"


# ===========================================================================
# C. Candidate Detail UX Tests
# ===========================================================================


class TestCandidateDetailVM:
    """Verify violated/closest-active summary logic."""

    def test_violated_sorted_worst_first(self):
        """Violated constraints must be sorted most-negative first."""
        margins = {
            "c1": -0.0003,
            "c2": 0.5,
            "c3": -0.05,
            "c4": -0.001,
            "c5": 0.001,
        }
        c = _make_candidate(constraint_margins=margins)
        vm = CandidateDetailVM.from_candidate(1, c)
        assert vm.violated_count == 3
        assert vm.violated[0].margin == -0.05  # worst
        assert vm.violated[1].margin == -0.001
        assert vm.violated[2].margin == -0.0003

    def test_closest_active_sorted_smallest_first(self):
        """Closest active constraints: nonneg margins, sorted ascending."""
        margins = {
            "c1": 0.5,
            "c2": 0.001,
            "c3": 0.1,
            "c4": 0.02,
            "c5": 0.0001,
        }
        c = _make_candidate(constraint_margins=margins, feasible=True, v_max=0.0)
        vm = CandidateDetailVM.from_candidate(1, c)
        assert vm.violated_count == 0
        assert len(vm.closest_active) == 5
        assert vm.closest_active[0].margin == 0.0001
        assert vm.closest_active[1].margin == 0.001

    def test_closest_active_capped_at_10(self):
        """At most 10 closest-active entries."""
        margins = {f"c{i}": 0.01 * i for i in range(50)}
        c = _make_candidate(constraint_margins=margins, feasible=True, v_max=0.0)
        vm = CandidateDetailVM.from_candidate(1, c)
        assert len(vm.closest_active) <= 10

    def test_no_violations_message(self):
        """When all margins >= 0, violated list is empty."""
        margins = {"c1": 0.1, "c2": 0.5, "c3": 0.001}
        c = _make_candidate(
            constraint_margins=margins, feasible=True, near_feasible=True, v_max=0.0
        )
        vm = CandidateDetailVM.from_candidate(1, c)
        assert vm.violated_count == 0
        assert vm.violated == []

    def test_large_constraint_list_summarized(self):
        """200+ constraints should produce a summary, not all rows in violated/active."""
        margins = {}
        for i in range(209):
            if i == 207:
                margins[f"gamma_f{1e6 + i}Hz"] = -0.0003  # one violation
            else:
                margins[f"gamma_f{1e6 + i}Hz"] = 0.01 + 0.001 * i
        c = _make_candidate(constraint_margins=margins, v_max=0.0003)
        vm = CandidateDetailVM.from_candidate(1, c)
        assert vm.total_hard == 209
        assert vm.violated_count == 1
        assert len(vm.closest_active) == 10  # capped
        # Total displayed << 209
        displayed = vm.violated_count + len(vm.closest_active)
        assert displayed < 20

    def test_all_constraints_contains_full_list(self):
        """all_constraints should contain every margin entry."""
        margins = {f"c{i}": 0.01 * i for i in range(50)}
        c = _make_candidate(constraint_margins=margins, feasible=True, v_max=0.0)
        vm = CandidateDetailVM.from_candidate(1, c)
        assert len(vm.all_constraints) == 50


# ===========================================================================
# D. hard_207 Identification
# ===========================================================================


class TestHard207Identification:
    """Identify what hard_207 physically represents in the FAST smoke case."""

    def test_layout_has_sufficient_rows(self):
        """The FAST-preset layout with 3 cells per branch should have ~209+ rows."""
        layout, _ = _make_fast_layout(n_cells_b1=3, n_cells_b2=3)
        assert layout.n >= 209, f"Only {layout.n} constraints, expected >=209"

    def test_hard_207_is_identifiable(self):
        """Row 207 (0-indexed) must have a meaningful descriptor."""
        layout, eval_freqs = _make_fast_layout(n_cells_b1=3, n_cells_b2=3)
        if layout.n > 207:
            desc = layout.descriptors[207]
            label = human_label(desc, eval_freqs)
            assert label, "hard_207 has empty label"
            assert desc.constraint_type, "hard_207 has no constraint type"
            # Print for diagnostic report (captured by pytest -v)
            print("\n  hard_207 physical meaning:")
            print("    row index: 207")
            print(f"    descriptor name: {desc.name}")
            print(f"    constraint_type: {desc.constraint_type}")
            print(f"    human label: {label}")
            print(f"    branch: {desc.branch}")
            print(f"    cell_index: {desc.cell_index}")
            print(f"    freq_index: {desc.freq_index}")
            if desc.freq_index is not None and desc.freq_index < len(eval_freqs):
                print(f"    frequency: {eval_freqs[desc.freq_index]:.0f} Hz")

    def test_layout_ordering_deterministic(self):
        """Constraint ordering must be identical across compilations."""
        l1, _ = _make_fast_layout()
        l2, _ = _make_fast_layout()
        for i in range(min(l1.n, l2.n)):
            assert l1.descriptors[i].name == l2.descriptors[i].name, (
                f"Row {i}: {l1.descriptors[i].name} != {l2.descriptors[i].name}"
            )


# ===========================================================================
# E. Integration: Polish Outcome in CandidateResult
# ===========================================================================


class TestCandidateResultPolishOutcome:
    """Verify the new local_polish_outcome field."""

    def test_field_exists_and_defaults_empty(self):
        """New field exists with empty string default (backward compat)."""
        c = CandidateResult()
        assert c.local_polish_outcome == ""

    def test_field_roundtrips_through_model(self):
        """Pydantic model accepts and stores all outcome values."""
        for outcome in ("polished_retained", "pre_polish_retained", "fd_fallback", "not_selected"):
            c = CandidateResult(local_polish_outcome=outcome)
            assert c.local_polish_outcome == outcome

    def test_candidate_detail_vm_propagates_outcome(self):
        """CandidateDetailVM correctly propagates the outcome field."""
        c = _make_candidate(
            local_polish_method="trust-constr",
            local_polish_outcome="polished_retained",
        )
        vm = CandidateDetailVM.from_candidate(1, c)
        assert vm.local_polish_outcome == "polished_retained"
        assert vm.local_polish_method == "trust-constr"

class TestDataModelStability:
    """Verifies that the canonical constraint model is stable and decoupled from presentation."""

    def test_canonical_ids_stable(self):
        import numpy as np

        from foster_eom.optimize.evaluator import DomainEvaluatorCache, evaluate
        from tests.unit.test_p12_5_e_analytical_polish import _build_case
        ctx = _build_case(feasible=True)
        cache = DomainEvaluatorCache()
        res = evaluate(np.array([0.5, 0.5, 0.5]), ctx, cache)
        from foster_eom.optimize.engine import _build_candidate_result
        cr = _build_candidate_result(res, ctx.domain, ctx, "test")

        # Keys must be hard_0, hard_1, etc.
        assert list(cr.constraint_margins.keys()) == [f"hard_{i}" for i in range(len(res.hard_margins))]

    def test_duplicate_labels_no_overwrite(self):
        from foster_eom.domain.results import CandidateResult, CoarseGridSummary
        from foster_eom.gui.view_models.optimize_vm import CandidateDetailVM

        cr = CandidateResult(
            candidate_id="test", topology_id="test", orientation="test", domain_id="test",
            branch1_realization="test", branch2_realization="test", branch1_cells=1, branch2_cells=1,
            branch1_has_c0=False, branch1_has_linf=False, branch2_has_c0=False, branch2_has_linf=False,
            x_pre_polish=(0.0,), objective_value=0.0, base_objective_value=0.0,
            soft_penalty_total=0.0, feasible=False, near_feasible=False, v_max=1.0,
            numerical_status="ok", local_polish_method="", local_polish_outcome="",
            seed_source="", target_solutions=[], all_solutions=[],
            coarse_grid_summary=CoarseGridSummary(coarse_evaluated=False, off_target_n_points=0, off_target_v_eom_peak_v=0.0),
            objective_terms={},
            constraint_margins={"hard_0": -1.0, "hard_1": -2.0}
        )

        label_map = {"hard_0": "Duplicate Label", "hard_1": "Duplicate Label"}
        vm = CandidateDetailVM.from_candidate(1, cr, label_map=label_map)

        assert len(vm.all_constraints) == 2
        assert vm.all_constraints[0].label == "Duplicate Label"
        assert vm.all_constraints[1].label == "Duplicate Label"
        # Values must be retained properly despite duplicate presentation labels
        assert vm.all_constraints[0].margin == -1.0
        assert vm.all_constraints[1].margin == -2.0
