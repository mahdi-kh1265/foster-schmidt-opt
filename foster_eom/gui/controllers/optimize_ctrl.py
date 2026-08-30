"""Controller for optimization."""

from __future__ import annotations

import threading

from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState
from foster_eom.optimize.engine import run_optimization
from foster_eom.optimize.progress import ProgressCallback


class OptimizeCtrl:
    @staticmethod
    def run(
        state: ProjectState,
        cancel_event: threading.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> object:
        """Run optimization and return the result."""
        if not state.frequencies_hz:
            raise ValueError("At least one target frequency must be specified.")

        spec = state_to_spec(state)

        import numpy as np

        from foster_eom.foster.seed import generate_seeds
        from foster_eom.models.factory import build_eom_model

        eom_model = build_eom_model(spec.eom)
        f_targets_hz = np.array([t.frequency_hz for t in spec.frequencies.enabled_targets])
        voltage_targets_rms_v = tuple(t.voltage_target_rms_v for t in spec.frequencies.enabled_targets)

        seed_result = generate_seeds(
            r_match_ohm=spec.source.z_source_real_ohm,
            source_spec=spec.source,
            eom_model=eom_model,
            f_targets_hz=f_targets_hz,
            topo_spec=spec.topology,
            component_limits=spec.components.continuous_limits,
        )

        return run_optimization(
            seed_result=seed_result,
            opt_spec=spec.optimization,
            source_spec=spec.source,
            eom_model=eom_model,
            component_limits=spec.components.continuous_limits,
            match_constraints=spec.matching,
            stress_constraints=spec.stress,
            target_frequencies_hz=tuple(f_targets_hz),
            sweep_f_min_hz=spec.frequencies.sweep_f_min_hz,
            sweep_f_max_hz=spec.frequencies.sweep_f_max_hz,
            voltage_targets_rms_v=voltage_targets_rms_v,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
