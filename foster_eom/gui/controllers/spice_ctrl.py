"""Controller for SPICE verification."""

from __future__ import annotations

from foster_eom.circuit.solve import solve_circuit
from foster_eom.gui.adapter import state_to_spec
from foster_eom.gui.state import ProjectState
from foster_eom.realization.substitute import build_substituted_graph
from foster_eom.spice.api import validate_against_mna
from foster_eom.spice.netlist import build_netlist


class SpiceCtrl:
    @staticmethod
    def export_netlist(state: ProjectState, realization_result: object) -> object:
        """Generate SPICE netlist string."""
        from foster_eom.realization.result import RealizationResult

        if not isinstance(realization_result, RealizationResult):
            raise TypeError("Expected RealizationResult")

        if realization_result.best is None:
            raise ValueError("No catalog realization available for SPICE export.")

        spec = state_to_spec(state)
        graph = build_substituted_graph(realization_result.best)
        freqs = [t.freq_hz for t in spec.frequencies.targets]
        return build_netlist(graph, spec.source, freqs)

    @staticmethod
    def validate(state: ProjectState, realization_result: object) -> object:
        """Run ngspice validation against MNA."""
        from foster_eom.realization.result import RealizationResult

        if not isinstance(realization_result, RealizationResult):
            raise TypeError("Expected RealizationResult")

        if realization_result.best is None:
            raise ValueError("No catalog realization available for SPICE validation.")

        spec = state_to_spec(state)
        graph = build_substituted_graph(realization_result.best)
        freqs = [t.freq_hz for t in spec.frequencies.targets]

        # MNA baseline solutions
        sols = solve_circuit(graph, spec.source, freqs)

        return validate_against_mna(graph, spec.source, sols)
