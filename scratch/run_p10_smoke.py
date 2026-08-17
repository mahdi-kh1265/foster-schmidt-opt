import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from unittest.mock import MagicMock

from foster_eom.catalog.component import FallbackPolicy
from foster_eom.catalog.library import ComponentLibrary
from foster_eom.circuit.graph import CircuitGraph, Element, ElementKind, Node, Port
from foster_eom.circuit.mna import SourceSpec
from foster_eom.domain.constraints import MatchConstraints, StressConstraints
from foster_eom.domain.source import SourceMode
from foster_eom.optimize.constraints import ConstraintSeverity, compile_constraint_layout
from foster_eom.optimize.evaluator import EvaluationContext
from foster_eom.optimize.objective import ObjectiveConfig
from foster_eom.realization.neighborhoods import build_neighborhoods
from foster_eom.realization.result import CatalogCombo
from foster_eom.realization.spec import SlotSpec
from foster_eom.robustness.runner import run_robustness
from foster_eom.robustness.sampler import RobustnessSpec


def run_smoke():
    DB = Path("vendor_packs/real_parts.fseom.db")

    BAND = (100e6, 200e6)
    TARGET = (150e6,)

    L_nom = 10e-9
    C_nom = 10e-12
    R_load = 50.0

    # Parallel RLC circuit like in the integration test to give a reasonable nominal
    graph = CircuitGraph(
        ground_node_id="gnd",
        input_port=Port("n_in", "gnd"),
        eom_element_id="R_load",
    )
    graph.add_node(Node(id="n_in", is_ground=False))
    graph.add_node(Node(id="gnd", is_ground=True))
    graph.add_element(
        Element(id="b1_L1", kind=ElementKind.INDUCTOR, node_pos="n_in", node_neg="gnd", value=L_nom)
    )
    graph.add_element(
        Element(
            id="b1_C1", kind=ElementKind.CAPACITOR, node_pos="n_in", node_neg="gnd", value=C_nom
        )
    )
    graph.add_element(
        Element(
            id="R_load", kind=ElementKind.RESISTOR, node_pos="n_in", node_neg="gnd", value=R_load
        )
    )

    z_ref = 50.0

    # Resonance is around 503 MHz, so at 150 MHz it will have some gamma.
    # Set gamma_max somewhat high so nominal passes but tolerance might fail.
    gamma_max = 0.99

    match_c = MatchConstraints(
        gamma_max=gamma_max,
        resistance_max_ohm=1e6,
        max_abs_reactance_ohm=1e6,
    )
    stress_c = StressConstraints(
        source_current_rms_max_a=10.0,
        off_target_eom_peak_rms_v=100.0,
    )

    hard_layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=TARGET,
        evaluation_frequencies_hz=BAND,
        target_indices=(0,),
        off_target_indices=(0, 1),
        severity_filter=ConstraintSeverity.HARD,
        n_cells_b1=1,
        n_cells_b2=0,
        z_ref_ohm=z_ref,
    )
    soft_layout = compile_constraint_layout(
        match_constraints=match_c,
        stress_constraints=stress_c,
        extra_records=[],
        target_frequencies_hz=TARGET,
        evaluation_frequencies_hz=BAND,
        target_indices=(0,),
        off_target_indices=(0, 1),
        severity_filter=ConstraintSeverity.SOFT,
        n_cells_b1=1,
        n_cells_b2=0,
        z_ref_ohm=z_ref,
    )

    source_spec = SourceSpec(
        mode=SourceMode.THEVENIN,
        z_source_real_ohm=z_ref,
        z_ref_ohm=z_ref,
        thevenin_vrms=1.0,
    )

    obj_cfg = ObjectiveConfig(
        z_ref_ohm=z_ref,
        w_gamma=1.0,
        w_voltage=0.0,
        w_loss=0.0,
        w_complexity=0.0,
        eom_element_id="R_load",
        n_reactive=2,
    )

    domain = MagicMock()
    domain.topology.branch1_cells = 1
    domain.topology.branch2_cells = 0
    domain.pole_regions_branch1 = (None,)
    domain.pole_regions_branch2 = ()

    class DummyLimits:
        l_min_h = 1e-12
        l_max_h = 1e-3
        c_min_f = 1e-15
        c_max_f = 1e-6

    ctx = MagicMock(spec=EvaluationContext)
    ctx.evaluation_frequencies_hz = BAND
    ctx.target_indices = (0,)
    ctx.off_target_indices = (0, 1)
    ctx.hard_layout = hard_layout
    ctx.soft_layout = soft_layout
    ctx.source_spec = source_spec
    ctx.eom_model = None
    ctx.p06_sweep_band_hz = BAND
    ctx.feasibility_tolerance = 1e-6
    ctx.near_feasibility_tolerance = 0.05
    ctx.requires_coarse_for_hard_soft = True
    ctx.domain = domain
    ctx.component_limits = DummyLimits()
    ctx.match_constraints = match_c
    ctx.stress_constraints = stress_c
    ctx.objective_config = obj_cfg

    with ComponentLibrary(DB) as lib:
        slots = (
            SlotSpec(
                element_id="b1_L1",
                value_nom=L_nom,
                freq_range_hz=BAND,
                fallback_policy=FallbackPolicy.STRICT,
            ),
            SlotSpec(
                element_id="b1_C1",
                value_nom=C_nom,
                freq_range_hz=BAND,
                fallback_policy=FallbackPolicy.STRICT,
            ),
        )
        nh = build_neighborhoods(slots, lib, k_max=1)

        if not nh["b1_L1"] or not nh["b1_C1"]:
            print("Failed to find neighborhood components.")
            return

        combo = CatalogCombo(
            slot_entries={"b1_L1": nh["b1_L1"][0], "b1_C1": nh["b1_C1"][0]},
            eval_result=MagicMock(),
            deb_key=(False, 0.0, 0.0, 0.5),
            verify_passed=True,
        )

        spec = RobustnessSpec(
            n_samples=100,
            seed=42,
            method="random",
            p06_diagnostic="worst_k",
            p06_worst_k=5,
        )

        res = run_robustness(
            combo=combo,
            base_graph=graph,
            context=ctx,
            library=lib,
            spec=spec,
        )

        print("\n--- P10 Smoke Test Results ---")
        print(f"Nominal Feasible: {res.nominal_feasible}")
        stoch = [s for s in res.non_stochastic_slots]
        print(f"Non-stochastic slots: {stoch if stoch else 'None'}")
        print(f"Evaluable Yield: {res.yield_stats.yield_evaluable:.4f}")
        print(
            f"Yield Bounds: [{res.yield_stats.yield_lower_bound:.4f}, {res.yield_stats.yield_upper_bound:.4f}]"
        )
        print(f"Wilson CI: [{res.yield_stats.ci_lo:.4f}, {res.yield_stats.ci_hi:.4f}]")
        print(f"Unresolved (Model): {res.yield_stats.n_model_unresolved}")
        print(f"Unresolved (Numerical): {res.yield_stats.n_numerical_unresolved}")
        print(f"P06 Worst-K yield_p06: {res.yield_stats.yield_p06}")

        if res.oat_sensitivity:
            print(
                f"Dominant sensitivity slot: {res.oat_sensitivity[0].element_id} (J_sens={res.oat_sensitivity[0].sensitivity_J:.4f})"
            )
        else:
            print("Dominant sensitivity slot: None")


if __name__ == "__main__":
    run_smoke()
