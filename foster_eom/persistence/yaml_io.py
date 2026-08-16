"""YAML persistence for project files (spec §29.1).

Handles loading and saving ``*.fseom.yaml`` project files with schema
version checking and migration support.

Key design decisions:
- Arrays/results stay out of the project YAML
- Schema version is checked on load; migration stubs are provided
- The YAML representation closely mirrors the example in examples/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from foster_eom.domain.component import ComponentPolicy, ContinuousLimits
from foster_eom.domain.constraints import (
    MatchConstraints,
    QBandwidthConstraints,
    RobustnessSpec,
    StressConstraints,
)
from foster_eom.domain.eom import EOMModelSpec, EOMModelType, MotionalBranch
from foster_eom.domain.frequency_plan import FrequencyPlan, FrequencyTarget
from foster_eom.domain.objectives import (
    AnalysisSpec,
    ExportSpec,
    GlobalMethod,
    LocalMethod,
    OptimizationPreset,
    OptimizationSpec,
    SpiceVerificationMode,
    TimeDomainPhaseMode,
)
from foster_eom.domain.project import ProjectMeta, ProjectSpec
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.domain.topology import (
    LOrientation,
    PoleMode,
    PoleSpec,
    TopologySearchSpec,
)
from foster_eom.errors import ProjectValidationError, SchemaVersionError

CURRENT_SCHEMA_VERSION = "0.1"
SUPPORTED_SCHEMA_VERSIONS = {"0.1"}


def save_project(spec: ProjectSpec, path: str | Path) -> None:
    """Save a ProjectSpec to a ``.fseom.yaml`` file.

    Parameters
    ----------
    spec : ProjectSpec
        Project specification to save.
    path : str | Path
        Output file path.
    """
    data = _spec_to_dict(spec)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_project(path: str | Path) -> ProjectSpec:
    """Load a ProjectSpec from a ``.fseom.yaml`` file.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    ProjectSpec
        Validated project specification.

    Raises
    ------
    SchemaVersionError
        If the schema version is unsupported.
    ProjectValidationError
        If the YAML content fails validation.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ProjectValidationError(f"Expected YAML mapping at top level, got {type(data)}")

    version = str(data.get("schema_version", ""))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"Schema version '{version}' not in supported set {SUPPORTED_SCHEMA_VERSIONS}"
        )

    # Apply migrations if needed (placeholder for future versions)
    data = _migrate(data, version)

    try:
        return _dict_to_spec(data)
    except Exception as exc:
        raise ProjectValidationError(f"Failed to parse project YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _spec_to_dict(spec: ProjectSpec) -> dict[str, Any]:
    """Convert a ProjectSpec to a YAML-friendly dict.

    Uses a structure matching the example YAML for human readability.
    """
    d: dict[str, Any] = {}
    d["schema_version"] = spec.schema_version

    # Project metadata
    d["project"] = {
        "name": spec.project.name,
        "description": spec.project.description,
    }
    if spec.project.notes:
        d["project"]["notes"] = spec.project.notes

    # Source
    src: dict[str, Any] = {"mode": spec.source.mode.value}
    if spec.source.mode == SourceMode.AVAILABLE_POWER:
        if spec.source.available_power_dbm is not None:
            src["available_power_dbm"] = spec.source.available_power_dbm
        if spec.source.available_power_w is not None:
            src["available_power_w"] = spec.source.available_power_w
    elif spec.source.mode == SourceMode.THEVENIN:
        if spec.source.thevenin_vrms is not None:
            src["thevenin_vrms"] = spec.source.thevenin_vrms
        if spec.source.thevenin_vpp is not None:
            src["thevenin_vpp"] = spec.source.thevenin_vpp
    elif spec.source.mode == SourceMode.GENERATOR_INTO_Z0:
        src["generator_display_v"] = spec.source.generator_display_v
        src["generator_display_convention"] = spec.source.generator_display_convention
    src["z_source_ohm"] = {
        "real": spec.source.z_source_real_ohm,
        "imag": spec.source.z_source_imag_ohm,
    }
    src["z_ref_ohm"] = spec.source.z_ref_ohm
    src["phase_deg"] = spec.source.phase_deg
    src["insertion_loss_db"] = spec.source.insertion_loss_db
    d["source"] = src

    # EOM
    eom: dict[str, Any] = {
        "model_type": spec.eom.model_type.value,
        "name": spec.eom.name,
    }
    if spec.eom.validity_hz is not None:
        eom["validity_hz"] = list(spec.eom.validity_hz)
    if spec.eom.rs_ohm is not None or spec.eom.ls_h is not None:
        series: dict[str, Any] = {}
        if spec.eom.rs_ohm is not None:
            series["Rs_ohm"] = spec.eom.rs_ohm
        if spec.eom.ls_h is not None:
            series["Ls_H"] = spec.eom.ls_h
        eom["series"] = series
    if spec.eom.c0_f is not None or spec.eom.g0_s is not None:
        static: dict[str, Any] = {}
        if spec.eom.c0_f is not None:
            static["C0_F"] = spec.eom.c0_f
        if spec.eom.g0_s is not None:
            static["G0_S"] = spec.eom.g0_s
        eom["static_branch"] = static
    if spec.eom.motional_branches:
        eom["motional_branches"] = [
            {"Rm_ohm": b.rm_ohm, "Lm_H": b.lm_h, "Cm_F": b.cm_f} for b in spec.eom.motional_branches
        ]
    if spec.eom.data_file is not None:
        eom["data_file"] = spec.eom.data_file
    if spec.eom.data_format is not None:
        eom["data_format"] = spec.eom.data_format
    d["eom"] = eom

    # Frequencies
    freq: dict[str, Any] = {
        "targets": [_freq_target_to_dict(t) for t in spec.frequencies.targets],
        "verification_band_hz": [spec.frequencies.sweep_f_min_hz, spec.frequencies.sweep_f_max_hz],
        "base_grid_points": spec.frequencies.base_grid_points,
        "adaptive_refinement": spec.frequencies.adaptive_sweep_enabled,
    }
    d["frequencies"] = freq

    # Matching
    d["matching"] = {
        "gamma_max_at_targets": spec.matching.gamma_max,
        "resistance_window_ohm": [
            spec.matching.resistance_min_ohm,
            spec.matching.resistance_max_ohm,
        ],
        "max_abs_reactance_ohm": spec.matching.max_abs_reactance_ohm,
    }

    # Topology
    topo: dict[str, Any] = {
        "orientations": [o.value for o in spec.topology.orientations],
        "branch1_cells": {
            "min": spec.topology.branch1_cells_min,
            "max": spec.topology.branch1_cells_max,
        },
        "branch2_cells": {
            "min": spec.topology.branch2_cells_min,
            "max": spec.topology.branch2_cells_max,
        },
        "endpoint_series_cap_allowed": spec.topology.endpoint_series_cap_branch1,
        "endpoint_series_ind_allowed": spec.topology.endpoint_series_ind_branch1,
        "max_total_reactive_components": spec.topology.max_total_reactive_components,
        "complexity_penalty": spec.topology.complexity_penalty,
    }
    d["topology"] = topo

    # Poles
    ps = spec.topology.pole_spec
    poles: dict[str, Any] = {"mode": ps.mode.value}
    if ps.min_separation_hz > 0:
        poles["min_pole_separation_hz"] = ps.min_separation_hz
    if ps.min_distance_from_target_hz > 0:
        poles["min_target_distance_hz"] = ps.min_distance_from_target_hz
    if ps.allowed_band_hz is not None:
        poles["allowed_band_hz"] = list(ps.allowed_band_hz)
    if ps.fixed_poles_hz:
        poles["fixed_poles_hz"] = ps.fixed_poles_hz
    d["poles"] = poles

    # Components
    cl = spec.components.continuous_limits
    comp: dict[str, Any] = {
        "continuous_limits": {
            "L_H": [cl.l_min_h, cl.l_max_h],
            "C_F": [cl.c_min_f, cl.c_max_f],
        },
        "capacitor_dielectrics": spec.components.capacitor_dielectrics,
        "min_inductor_srf_ratio": spec.components.min_inductor_srf_ratio,
        "voltage_derating_fraction": spec.components.voltage_derating_fraction,
        "current_derating_fraction": spec.components.current_derating_fraction,
        "catalog_realization_enabled": spec.components.catalog_realization_enabled,
        "allowed_manufacturers": spec.components.allowed_manufacturers,
    }
    d["components"] = comp

    # Q/bandwidth
    qb = spec.q_bandwidth
    d["q_bandwidth"] = {
        "q_reporting_enabled": qb.q_reporting_enabled,
        "preferred_voltage_q_range": list(qb.preferred_q_range) if qb.preferred_q_range else None,
        "q_is_hard_constraint": qb.q_is_hard_constraint,
        "min_usable_half_bandwidth_hz": qb.min_usable_half_bandwidth_hz,
        "voltage_fraction_for_usable_bandwidth": qb.voltage_fraction_for_bandwidth,
    }

    # Stress
    st = spec.stress
    d["stress"] = {
        "source_current_rms_max_a": st.source_current_rms_max_a,
        "default_capacitor_peak_voltage_max_v": st.default_cap_peak_voltage_v,
        "default_inductor_peak_current_max_a": st.default_ind_peak_current_a,
        "off_target_eom_peak_max_rms_v": st.off_target_eom_peak_rms_v,
    }

    # Robustness
    rob = spec.robustness
    d["robustness"] = {
        "enabled": rob.enabled,
        "optimization_scenarios": rob.optimization_scenarios,
        "final_monte_carlo_samples": rob.final_monte_carlo_samples,
        "default_component_tolerance_fraction": rob.default_component_tolerance,
        "eom_C0_tolerance_fraction": rob.eom_c0_tolerance,
        "random_seed": rob.random_seed,
    }

    # Optimization
    opt = spec.optimization
    d["optimization"] = {
        "preset": opt.preset.value,
        "global": {
            "method": opt.global_method.value,
            "population_size_multiplier": opt.population_size_multiplier,
            "max_evaluations": opt.max_global_evaluations,
            "workers": opt.workers,
        },
        "local": {
            "preferred_method": opt.local_method.value,
            "fallback_method": opt.local_fallback_method.value,
            "polish_top_k": opt.polish_top_k,
            "max_iterations": opt.local_max_iterations,
        },
        "feasibility_first": opt.feasibility_first,
        "checkpoint_every_evaluations": opt.checkpoint_every_evaluations,
        "random_seed": opt.random_seed,
    }

    # Analysis
    an = spec.analysis
    d["analysis"] = {
        "detect_unintended_resonances": an.detect_unintended_resonances,
        "time_domain_reconstruction": an.time_domain_reconstruction,
        "time_domain_phase_mode": an.time_domain_phase_mode.value,
        "spice_verification": an.spice_verification.value,
    }

    # Export
    ex = spec.export
    d["export"] = {
        "save_csv": ex.save_csv,
        "save_npz": ex.save_npz,
        "save_spice_netlist": ex.save_spice_netlist,
        "save_plots": ex.save_plots,
        "save_full_provenance": ex.save_full_provenance,
    }

    return d


def _freq_target_to_dict(t: FrequencyTarget) -> dict[str, Any]:
    """Convert a FrequencyTarget to a YAML-friendly dict."""
    d: dict[str, Any] = {"frequency_hz": t.frequency_hz}
    if t.label:
        d["label"] = t.label
    if t.voltage_target_rms_v is not None:
        d["voltage_target_rms_v"] = t.voltage_target_rms_v
    if t.voltage_min_rms_v is not None:
        d["voltage_min_rms_v"] = t.voltage_min_rms_v
    if t.voltage_max_rms_v is not None:
        d["voltage_max_rms_v"] = t.voltage_max_rms_v
    if t.voltage_weight != 1.0:
        d["weight"] = t.voltage_weight
    if not t.enabled:
        d["enabled"] = False
    return d


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _dict_to_spec(data: dict[str, Any]) -> ProjectSpec:
    """Convert a YAML dict to a validated ProjectSpec."""
    # Project metadata
    proj_data = data.get("project", {})
    project_meta = ProjectMeta(
        name=proj_data.get("name", ""),
        description=proj_data.get("description", ""),
        notes=proj_data.get("notes", ""),
    )

    # Source
    src_data = data.get("source", {})
    z_src = src_data.get("z_source_ohm", {})
    source = SourceSpec(
        mode=SourceMode(src_data.get("mode", "available_power")),
        z_source_real_ohm=z_src.get("real", 50.0) if isinstance(z_src, dict) else float(z_src),
        z_source_imag_ohm=z_src.get("imag", 0.0) if isinstance(z_src, dict) else 0.0,
        z_ref_ohm=src_data.get("z_ref_ohm", 50.0),
        phase_deg=src_data.get("phase_deg", 0.0),
        insertion_loss_db=src_data.get("insertion_loss_db", 0.0),
        available_power_dbm=src_data.get("available_power_dbm"),
        available_power_w=src_data.get("available_power_w"),
        thevenin_vrms=src_data.get("thevenin_vrms"),
        thevenin_vpp=src_data.get("thevenin_vpp"),
        generator_display_v=src_data.get("generator_display_v"),
        generator_display_convention=src_data.get("generator_display_convention"),
    )

    # EOM
    eom_data = data.get("eom", {})
    series_data = eom_data.get("series", {})
    static_data = eom_data.get("static_branch", {})
    motional_data = eom_data.get("motional_branches", [])
    validity = eom_data.get("validity_hz")

    eom = EOMModelSpec(
        model_type=EOMModelType(eom_data.get("model_type", "ideal_capacitor")),
        name=eom_data.get("name", ""),
        validity_hz=tuple(validity) if validity else None,
        rs_ohm=series_data.get("Rs_ohm"),
        ls_h=series_data.get("Ls_H"),
        c0_f=static_data.get("C0_F"),
        g0_s=static_data.get("G0_S"),
        motional_branches=[
            MotionalBranch(
                rm_ohm=m.get("Rm_ohm", 0.0),
                lm_h=m.get("Lm_H"),
                cm_f=m.get("Cm_F"),
            )
            for m in motional_data
        ],
        data_file=eom_data.get("data_file"),
        data_format=eom_data.get("data_format"),
    )

    # Frequencies
    freq_data = data.get("frequencies", {})
    band = freq_data.get("verification_band_hz", [1e6, 30e6])
    targets_raw = freq_data.get("targets", [])
    targets = [
        FrequencyTarget(
            label=t.get("label", ""),
            frequency_hz=t["frequency_hz"],
            enabled=t.get("enabled", True),
            voltage_target_rms_v=t.get("voltage_target_rms_v"),
            voltage_min_rms_v=t.get("voltage_min_rms_v"),
            voltage_max_rms_v=t.get("voltage_max_rms_v"),
            voltage_weight=t.get("weight", 1.0),
        )
        for t in targets_raw
    ]
    frequencies = FrequencyPlan(
        targets=targets,
        sweep_f_min_hz=band[0],
        sweep_f_max_hz=band[1],
        base_grid_points=freq_data.get("base_grid_points", 1201),
        adaptive_sweep_enabled=freq_data.get("adaptive_refinement", True),
    )

    # Matching
    match_data = data.get("matching", {})
    rw = match_data.get("resistance_window_ohm", [35.0, 70.0])
    matching = MatchConstraints(
        gamma_max=match_data.get("gamma_max_at_targets", 0.25),
        resistance_min_ohm=rw[0],
        resistance_max_ohm=rw[1],
        max_abs_reactance_ohm=match_data.get("max_abs_reactance_ohm", 20.0),
    )

    # Topology
    topo_data = data.get("topology", {})
    b1 = topo_data.get("branch1_cells", {})
    b2 = topo_data.get("branch2_cells", {})

    # Poles
    pole_data = data.get("poles", {})
    pole_band = pole_data.get("allowed_band_hz")
    pole_spec = PoleSpec(
        mode=PoleMode(pole_data.get("mode", "auto")),
        fixed_poles_hz=pole_data.get("fixed_poles_hz", []),
        min_separation_hz=pole_data.get("min_pole_separation_hz", 100e3),
        min_distance_from_target_hz=pole_data.get("min_target_distance_hz", 50e3),
        allowed_band_hz=tuple(pole_band) if pole_band else None,
    )

    topology = TopologySearchSpec(
        orientations=[
            LOrientation(o) for o in topo_data.get("orientations", ["schmidt_shunt_then_series"])
        ],
        branch1_cells_min=b1.get("min", 1),
        branch1_cells_max=b1.get("max", 3),
        branch2_cells_min=b2.get("min", 1),
        branch2_cells_max=b2.get("max", 3),
        endpoint_series_cap_branch1=topo_data.get("endpoint_series_cap_allowed", True),
        endpoint_series_ind_branch1=topo_data.get("endpoint_series_ind_allowed", True),
        endpoint_series_cap_branch2=topo_data.get("endpoint_series_cap_allowed", True),
        endpoint_series_ind_branch2=topo_data.get("endpoint_series_ind_allowed", True),
        max_total_reactive_components=topo_data.get("max_total_reactive_components", 14),
        complexity_penalty=topo_data.get("complexity_penalty", 0.02),
        pole_spec=pole_spec,
    )

    # Components
    comp_data = data.get("components", {})
    cl_data = comp_data.get("continuous_limits", {})
    l_range = cl_data.get("L_H", [10e-9, 100e-6])
    c_range = cl_data.get("C_F", [0.2e-12, 20e-9])
    components = ComponentPolicy(
        continuous_limits=ContinuousLimits(
            l_min_h=l_range[0],
            l_max_h=l_range[1],
            c_min_f=c_range[0],
            c_max_f=c_range[1],
        ),
        capacitor_dielectrics=comp_data.get("capacitor_dielectrics", ["C0G", "NP0"]),
        min_inductor_srf_ratio=comp_data.get("min_inductor_srf_ratio", 2.0),
        voltage_derating_fraction=comp_data.get("voltage_derating_fraction", 0.60),
        current_derating_fraction=comp_data.get("current_derating_fraction", 0.60),
        catalog_realization_enabled=comp_data.get("catalog_realization_enabled", False),
        allowed_manufacturers=comp_data.get("allowed_manufacturers", []),
    )

    # Q/bandwidth
    qb_data = data.get("q_bandwidth", {})
    pref_q = qb_data.get("preferred_voltage_q_range")
    q_bandwidth = QBandwidthConstraints(
        q_reporting_enabled=qb_data.get("q_reporting_enabled", True),
        preferred_q_range=tuple(pref_q) if pref_q else None,
        q_is_hard_constraint=qb_data.get("q_is_hard_constraint", False),
        min_usable_half_bandwidth_hz=qb_data.get("min_usable_half_bandwidth_hz", 50e3),
        voltage_fraction_for_bandwidth=qb_data.get("voltage_fraction_for_usable_bandwidth", 0.90),
    )

    # Stress
    st_data = data.get("stress", {})
    stress = StressConstraints(
        source_current_rms_max_a=st_data.get("source_current_rms_max_a", 0.5),
        default_cap_peak_voltage_v=st_data.get("default_capacitor_peak_voltage_max_v", 100.0),
        default_ind_peak_current_a=st_data.get("default_inductor_peak_current_max_a", 1.0),
        off_target_eom_peak_rms_v=st_data.get("off_target_eom_peak_max_rms_v", 50.0),
    )

    # Robustness
    rob_data = data.get("robustness", {})
    robustness = RobustnessSpec(
        enabled=rob_data.get("enabled", True),
        optimization_scenarios=rob_data.get("optimization_scenarios", 32),
        final_monte_carlo_samples=rob_data.get("final_monte_carlo_samples", 2000),
        default_component_tolerance=rob_data.get("default_component_tolerance_fraction", 0.02),
        eom_c0_tolerance=rob_data.get("eom_C0_tolerance_fraction", 0.03),
        random_seed=rob_data.get("random_seed", 20260815),
    )

    # Optimization
    opt_data = data.get("optimization", {})
    glob_data = opt_data.get("global", {})
    loc_data = opt_data.get("local", {})
    optimization = OptimizationSpec(
        preset=OptimizationPreset(opt_data.get("preset", "balanced")),
        random_seed=opt_data.get("random_seed", 20260815),
        global_method=GlobalMethod(glob_data.get("method", "differential_evolution")),
        population_size_multiplier=glob_data.get("population_size_multiplier", 12),
        max_global_evaluations=glob_data.get("max_evaluations", 50000),
        workers=glob_data.get("workers", "auto"),
        polish_top_k=loc_data.get("polish_top_k", 8),
        local_method=LocalMethod(loc_data.get("preferred_method", "ipopt")),
        local_fallback_method=LocalMethod(loc_data.get("fallback_method", "trust-constr")),
        local_max_iterations=loc_data.get("max_iterations", 1500),
        feasibility_first=opt_data.get("feasibility_first", True),
        checkpoint_every_evaluations=opt_data.get("checkpoint_every_evaluations", 5000),
    )

    # Analysis
    an_data = data.get("analysis", {})
    analysis = AnalysisSpec(
        detect_unintended_resonances=an_data.get("detect_unintended_resonances", True),
        time_domain_reconstruction=an_data.get("time_domain_reconstruction", True),
        time_domain_phase_mode=TimeDomainPhaseMode(
            an_data.get("time_domain_phase_mode", "specified")
        ),
        spice_verification=SpiceVerificationMode(an_data.get("spice_verification", "optional")),
    )

    # Export
    ex_data = data.get("export", {})
    export = ExportSpec(
        save_csv=ex_data.get("save_csv", True),
        save_npz=ex_data.get("save_npz", True),
        save_spice_netlist=ex_data.get("save_spice_netlist", True),
        save_plots=ex_data.get("save_plots", True),
        save_full_provenance=ex_data.get("save_full_provenance", True),
    )

    return ProjectSpec(
        schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
        project=project_meta,
        source=source,
        eom=eom,
        frequencies=frequencies,
        matching=matching,
        topology=topology,
        components=components,
        q_bandwidth=q_bandwidth,
        stress=stress,
        robustness=robustness,
        optimization=optimization,
        analysis=analysis,
        export=export,
    )


def _migrate(data: dict[str, Any], version: str) -> dict[str, Any]:
    """Apply schema migrations.

    Parameters
    ----------
    data : dict
        Raw YAML data.
    version : str
        Current schema version in the file.

    Returns
    -------
    dict
        Migrated data (currently identity for v0.1).
    """
    # No migrations needed yet
    return data
