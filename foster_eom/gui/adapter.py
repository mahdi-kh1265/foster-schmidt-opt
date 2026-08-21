"""Adapter between GUI ProjectState and backend ProjectSpec."""

from __future__ import annotations

from pathlib import Path

import yaml

from foster_eom.domain.eom import EOMModelSpec, EOMModelType, MotionalBranch
from foster_eom.domain.frequency_plan import FrequencyPlan, FrequencyTarget
from foster_eom.domain.objectives import DerivativeMode, OptimizationSpec
from foster_eom.domain.project import ProjectSpec
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import TopologySearchSpec
from foster_eom.gui.state import (
    EOMParams,
    MotionalBranchParams,
    ProjectState,
    SourceParams,
    TopologyParams,
)
from foster_eom.persistence.yaml_io import load_project, save_project


def state_to_spec(state: ProjectState) -> ProjectSpec:
    """Convert GUI ProjectState to backend ProjectSpec (Pydantic model)."""
    source = SourceSpec(
        mode=SourceMode(state.source.mode),
        thevenin_vrms=state.source.vth_rms,
        z_source_real_ohm=state.source.z_source_ohm,
        available_power_w=state.source.available_power_w,
    )

    branches = [
        MotionalBranch(rm_ohm=b.rm_ohm, lm_h=b.lm_h, cm_f=b.cm_f)
        for b in state.eom.motional_branches
    ]

    eom = EOMModelSpec(
        model_type=EOMModelType(state.eom.model_type),
        c0_f=state.eom.c0_f,
        rs_ohm=state.eom.rs_ohm,
        ls_h=state.eom.ls_h,
        g0_s=state.eom.g0_s,
        motional_branches=branches,
        data_file=state.eom.tabular_file,
        data_format=state.eom.tabular_format,
        validity_hz=state.eom.validity_hz,
    )

    frequencies = FrequencyPlan(
        targets=[FrequencyTarget(frequency_hz=f) for f in state.frequencies_hz],
        sweep_f_min_hz=state.sweep_f_min_hz,
        sweep_f_max_hz=state.sweep_f_max_hz,
    )

    topology = TopologySearchSpec(
        branch1_cells_max=state.topology.n_cells_per_branch,
        branch2_cells_max=state.topology.n_cells_per_branch if state.topology.n_branches > 1 else 0,
        branch1_cells_min=1,
        branch2_cells_min=1 if state.topology.n_branches > 1 else 0,
        # Default pole settings handled by TopologySearchSpec defaults
    )

    optimization = OptimizationSpec(
        local_derivative_mode=DerivativeMode.ANALYTICAL
    )

    return ProjectSpec(
        source=source,
        eom=eom,
        frequencies=frequencies,
        topology=topology,
        optimization=optimization,
    )


def spec_to_state(spec: ProjectSpec) -> ProjectState:
    """Reconstruct GUI ProjectState from a loaded ProjectSpec."""
    state = ProjectState()

    state.frequencies_hz = [t.frequency_hz for t in spec.frequencies.targets]
    state.sweep_f_min_hz = spec.frequencies.sweep_f_min_hz
    state.sweep_f_max_hz = spec.frequencies.sweep_f_max_hz

    state.source = SourceParams(
        mode=spec.source.mode.value,
        vth_rms=spec.source.thevenin_vrms,
        z_source_ohm=spec.source.z_source_real_ohm,
        available_power_w=spec.source.available_power_w,
    )

    branches = [
        MotionalBranchParams(rm_ohm=b.rm_ohm, lm_h=b.lm_h, cm_f=b.cm_f)
        for b in spec.eom.motional_branches
    ]
    state.eom = EOMParams(
        model_type=spec.eom.model_type.value,
        c0_f=spec.eom.c0_f,
        rs_ohm=spec.eom.rs_ohm,
        ls_h=spec.eom.ls_h,
        g0_s=spec.eom.g0_s,
        motional_branches=branches,
        tabular_file=spec.eom.data_file,
        tabular_format=spec.eom.data_format,
        validity_hz=spec.eom.validity_hz,
    )

    state.topology = TopologyParams(
        n_branches=2 if spec.topology.branch2_cells_max > 0 else 1,
        n_cells_per_branch=spec.topology.branch1_cells_max,
    )

    state.input_sha256 = state.compute_input_sha()
    return state


def _sidecar_path(base_path: str | Path) -> Path:
    p = Path(base_path)
    return p.with_name(p.name + ".gui.yaml")


def save_gui_project(state: ProjectState, path: str | Path) -> None:
    """Save ProjectSpec + GUI sidecar fields."""
    spec = state_to_spec(state)
    save_project(spec, path)

    # Write .gui.yaml sidecar
    sidecar_data = {
        "library_path": state.library_path,
        "library_sha": state.library_sha,
        "revision": state.revision,
        "input_sha256": state.input_sha256,
        "optimize_result_path": state.optimize_result_path,
        "optimize_revision": state.optimize_revision,
        "verify_result_path": state.verify_result_path,
        "verify_revision": state.verify_revision,
        "realization_result_path": state.realization_result_path,
        "realization_revision": state.realization_revision,
        "realization_library_sha": state.realization_library_sha,
        "robustness_result_path": state.robustness_result_path,
        "robustness_revision": state.robustness_revision,
        "robustness_library_sha": state.robustness_library_sha,
        "spice_result_path": state.spice_result_path,
        "spice_revision": state.spice_revision,
    }
    with open(_sidecar_path(path), "w", encoding="utf-8") as f:
        yaml.dump(sidecar_data, f)


def load_gui_project(path: str | Path) -> ProjectState:
    """Load ProjectSpec + GUI sidecar."""
    spec = load_project(path)
    state = spec_to_state(spec)

    sidecar = _sidecar_path(path)
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        state.library_path = data.get("library_path")
        state.library_sha = data.get("library_sha")
        state.revision = data.get("revision", state.revision)
        state.input_sha256 = data.get("input_sha256", state.input_sha256)
        state.optimize_result_path = data.get("optimize_result_path")
        state.optimize_revision = data.get("optimize_revision")
        state.verify_result_path = data.get("verify_result_path")
        state.verify_revision = data.get("verify_revision")
        state.realization_result_path = data.get("realization_result_path")
        state.realization_revision = data.get("realization_revision")
        state.realization_library_sha = data.get("realization_library_sha")
        state.robustness_result_path = data.get("robustness_result_path")
        state.robustness_revision = data.get("robustness_revision")
        state.robustness_library_sha = data.get("robustness_library_sha")
        state.spice_result_path = data.get("spice_result_path")
        state.spice_revision = data.get("spice_revision")

        # Invalidate results if inputs changed externally
        if state.input_sha256 != state.compute_input_sha():
            state.bump_revision()

    return state
