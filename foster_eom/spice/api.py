"""P11 top-level SPICE validation API.

Pipeline:
  1. build_netlist() -> check for unsupported models
  2. detect_ngspice() -> return solver_unavailable if absent
  3. run_ngspice() -> NgspiceResult
  4. Scale SPICE outputs by vth_phasor
  5. compute_quantity_comparison() for each quantity
  6. classify_status() -> SpiceValidationReport
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from foster_eom.circuit.graph import CircuitGraph
from foster_eom.circuit.measurements import CircuitSolution
from foster_eom.domain.source import SourceSpec
from foster_eom.spice.compare import classify_status, compute_quantity_comparison
from foster_eom.spice.netlist import build_netlist
from foster_eom.spice.ngspice import NgspiceRunError, detect_ngspice, run_ngspice
from foster_eom.spice.result import (
    MeasurementPlan,
    QuantityComparison,
    SpiceValidationReport,
    ValidationThresholds,
)

# Fixed convention strings — never derived at runtime
_SOURCE_CONV = "spice=AC_1_0_unit_phasor,scale=vth_phasor_in_python"
_CURRENT_CONV = "Vsense_oriented:I(Vsense)>0_into_DUT"
_PHASE_CONV = "angle(spice_conj_mna),masked_below_mag_floor"


def validate_against_mna(
    graph: CircuitGraph,
    source_spec: SourceSpec,
    mna_solutions: list[CircuitSolution],
    catalog_combo: Any | None = None,
    measurement_plan: MeasurementPlan | None = None,
    thresholds: ValidationThresholds | None = None,
    work_dir: Path | None = None,
    timeout_s: float = 30.0,
    title: str = "foster_eom",
) -> SpiceValidationReport:
    """Run full MNA-vs-SPICE validation pipeline.

    Parameters
    ----------
    graph : CircuitGraph
    source_spec : SourceSpec
    mna_solutions : list[CircuitSolution]
        Pre-computed MNA solutions at each frequency (must match frequency order).
    catalog_combo : optional
        Passed to build_netlist for provenance comments.
    measurement_plan : MeasurementPlan | None
    thresholds : ValidationThresholds | None
        Defaults to ValidationThresholds().
    work_dir : Path | None
    timeout_s : float
    title : str

    Returns
    -------
    SpiceValidationReport
        status="unsupported" if any element has no validated SPICE representation.
        status="solver_unavailable" if ngspice not on PATH.
    """
    thr = thresholds or ValidationThresholds()
    plan = measurement_plan or MeasurementPlan()

    frequencies_hz = np.array([sol.f_hz for sol in mna_solutions])

    vth = source_spec.vth_phasor

    # -- Step 1: build netlist (always, to detect unsupported models) --------
    netlist = build_netlist(
        graph,
        source_spec,
        list(frequencies_hz),
        title=title,
        measurement_plan=plan,
        catalog_combo=catalog_combo,
    )

    def _base_report(
        status: str,
        solver_version: str | None,
        comparisons: list[QuantityComparison],
        fail_reason: str | None,
    ) -> SpiceValidationReport:
        return SpiceValidationReport(
            title=title,
            status=status,  # type: ignore[arg-type]
            solver_version=solver_version,
            netlist_sha256=netlist.sha256,
            source_vth_phasor=vth,
            source_phase_deg=source_spec.phase_deg,
            source_convention=_SOURCE_CONV,
            current_direction_convention=_CURRENT_CONV,
            phase_convention=_PHASE_CONV,
            frequencies_hz=frequencies_hz,
            comparisons=comparisons,
            unsupported_elements=netlist.unsupported_elements,
            unsupported_model_reasons=netlist.unsupported_model_reasons,
            fail_reason=fail_reason,
            thresholds=thr,
        )

    if netlist.unsupported_elements:
        return _base_report(
            "unsupported",
            None,
            [],
            f"Unsupported models: {netlist.unsupported_elements}",
        )

    # -- Step 2: detect ngspice ----------------------------------------------
    solver_version = detect_ngspice()
    if solver_version is None:
        return _base_report("solver_unavailable", None, [], None)

    # -- Step 3: run ngspice -------------------------------------------------
    try:
        ng = run_ngspice(netlist, work_dir=work_dir, timeout_s=timeout_s)
    except NgspiceRunError as exc:
        return _base_report("fail", solver_version, [], f"ngspice run error: {exc}")

    # -- Step 4: scale SPICE outputs by vth_phasor ---------------------------
    # ng.node_voltages[spice_node] are unit-source complex phasors.
    # Multiply by vth to get same scale as MNA solutions.
    def _scale_node(spice_name: str) -> np.ndarray | None:
        arr = ng.node_voltages.get(spice_name)
        if arr is None:
            return None
        return arr * vth

    def _scale_sense(sense_name: str) -> np.ndarray | None:
        arr = ng.sense_currents.get(sense_name)
        if arr is None:
            return None
        return arr * vth

    # -- Step 5: build comparison quantities ---------------------------------
    comparisons: list[QuantityComparison] = []

    # Z_in = V_dut / I(Vsense)
    n_dut_spice = netlist.node_map[graph.input_port.node_pos]
    v_dut_spice = _scale_node(n_dut_spice)
    i_sense_spice = _scale_sense("Vsense")

    # MNA Z_in and I_port (source-droop current)
    mna_z_in = np.array(
        [sol.z_in if sol.z_in is not None else complex("nan") for sol in mna_solutions]
    )
    mna_i_port = np.array(
        [sol.i_port if sol.i_port is not None else complex("nan") for sol in mna_solutions]
    )

    if v_dut_spice is not None and i_sense_spice is not None:
        # Z_in
        with np.errstate(divide="ignore", invalid="ignore"):
            spice_z_in = np.where(i_sense_spice != 0, v_dut_spice / i_sense_spice, complex("nan"))
        comparisons.append(
            compute_quantity_comparison(
                "Z_in", frequencies_hz, mna_z_in, spice_z_in, thr, compute_resonance=True
            )
        )
        # I_port (current into DUT)
        comparisons.append(
            compute_quantity_comparison("I_port", frequencies_hz, mna_i_port, i_sense_spice, thr)
        )

    # V_eom
    eom_id = plan.eom_element_id or graph.eom_element_id
    if eom_id and eom_id in graph.elements:
        eom_elem = graph.elements[eom_id]
        ep_spice = netlist.node_map[eom_elem.node_pos]
        en_spice = netlist.node_map[eom_elem.node_neg]
        v_ep = _scale_node(ep_spice)
        v_en_arr = ng.node_voltages.get(en_spice)
        if v_ep is not None:
            v_en = (v_en_arr * vth) if v_en_arr is not None else np.zeros_like(v_ep)
            spice_v_eom = v_ep - v_en
            mna_v_eom = np.array(
                [
                    sol.v_eom
                    if (hasattr(sol, "v_eom") and sol.v_eom is not None)  # type: ignore[attr-defined]
                    else complex("nan")
                    for sol in mna_solutions
                ]
            )
            comparisons.append(
                compute_quantity_comparison("V_eom", frequencies_hz, mna_v_eom, spice_v_eom, thr)
            )

    # Branch currents
    for eid in plan.branch_element_ids:
        sense_name = netlist.sense_source_map.get(eid)
        if sense_name is None:
            continue
        spice_i = _scale_sense(sense_name)
        if spice_i is None:
            continue
        mna_i = np.array(
            [
                (
                    sol.element_measurements[eid].current  # type: ignore[index]
                    if (sol.element_measurements and eid in sol.element_measurements)
                    else complex("nan")
                )
                for sol in mna_solutions
            ]
        )
        comparisons.append(
            compute_quantity_comparison(f"branch_I_{eid}", frequencies_hz, mna_i, spice_i, thr)
        )

    # -- Step 6: classify ----------------------------------------------------
    status, fail_reason = classify_status(comparisons, thr)
    return _base_report(status, solver_version, comparisons, fail_reason)
