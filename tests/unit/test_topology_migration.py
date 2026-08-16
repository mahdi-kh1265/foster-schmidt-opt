"""Tests for schema v0.1 → v0.2 migration (Prompt 04B, poles migration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foster_eom.errors import SchemaVersionError
from foster_eom.persistence.yaml_io import load_project

# The approach: use raw YAML with the exact fields yaml_io expects, extracted
# from the example YAML structure. We only need to test the _migrate() path.


# Fixture: minimal v0.1 YAML with legacy poles block
_V01_WITH_POLES = """\
schema_version: "0.1"
project:
  name: "migration test"

source:
  mode: available_power
  available_power_dbm: 20.0
  z_source_ohm: {real: 50.0, imag: 0.0}

eom:
  model_type: mbvd
  name: "test"
  validity_hz: [1.0e6, 30.0e6]
  series:
    Rs_ohm: 0.5
    Ls_H: 15.0e-9
  static_branch:
    C0_F: 12.0e-12
    G0_S: 2.0e-5
  motional_branches:
    - Rm_ohm: 8.0
      Lm_H: 50.0e-6
      Cm_F: 9.0e-12

frequencies:
  targets:
    - {label: f1, frequency_hz: 10.0e6, voltage_target_rms_v: 20.0}

topology:
  orientations: [schmidt_shunt_then_series]
  branch1_cells: {min: 1, max: 2}
  branch2_cells: {min: 1, max: 2}

poles:
  mode: auto
  min_pole_separation_hz: 200.0e3
  min_target_distance_hz: 75.0e3
  allowed_band_hz: [5.0e6, 15.0e6]

components:
  continuous_limits:
    L_H: [10.0e-9, 100.0e-6]
    C_F: [0.2e-12, 20.0e-9]
"""


_V01_WITHOUT_POLES = """\
schema_version: "0.1"
project:
  name: "no poles test"

source:
  mode: available_power
  available_power_dbm: 20.0
  z_source_ohm: {real: 50.0, imag: 0.0}

eom:
  model_type: mbvd
  name: "test"
  validity_hz: [1.0e6, 30.0e6]
  series:
    Rs_ohm: 0.5
    Ls_H: 15.0e-9
  static_branch:
    C0_F: 12.0e-12
    G0_S: 2.0e-5
  motional_branches:
    - Rm_ohm: 8.0
      Lm_H: 50.0e-6
      Cm_F: 9.0e-12

frequencies:
  targets:
    - {label: f1, frequency_hz: 10.0e6, voltage_target_rms_v: 20.0}

components:
  continuous_limits:
    L_H: [10.0e-9, 100.0e-6]
    C_F: [0.2e-12, 20.0e-9]
"""


_V02_CONFLICTING = """\
schema_version: "0.2"
project:
  name: "conflicting test"

source:
  mode: available_power
  available_power_dbm: 20.0
  z_source_ohm: {real: 50.0, imag: 0.0}

eom:
  model_type: mbvd
  name: "test"
  validity_hz: [1.0e6, 30.0e6]
  series:
    Rs_ohm: 0.5
    Ls_H: 15.0e-9
  static_branch:
    C0_F: 12.0e-12
    G0_S: 2.0e-5
  motional_branches:
    - Rm_ohm: 8.0
      Lm_H: 50.0e-6
      Cm_F: 9.0e-12

frequencies:
  targets:
    - {label: f1, frequency_hz: 10.0e6, voltage_target_rms_v: 20.0}

poles:
  mode: auto

poles_branch1:
  mode: auto

poles_branch2:
  mode: auto

components:
  continuous_limits:
    L_H: [10.0e-9, 100.0e-6]
    C_F: [0.2e-12, 20.0e-9]
"""


class TestPolesMigration:
    """Test v0.1 → v0.2 migration of the poles field."""

    def _write(self, tmp_path: Path, content: str) -> Path:
        path = tmp_path / "test_project.yaml"
        path.write_text(content)
        return path

    def test_v01_with_poles_migrates_to_branch_specific(self, tmp_path: Path) -> None:
        """v0.1 with legacy 'poles:' → branch-specific pole specs."""
        path = self._write(tmp_path, _V01_WITH_POLES)
        spec = load_project(path)
        assert spec.schema_version == "0.2"
        ps1 = spec.topology.pole_spec_branch1
        ps2 = spec.topology.pole_spec_branch2
        assert ps1.min_separation_hz == pytest.approx(200.0e3)
        assert ps1.min_distance_from_target_hz == pytest.approx(75.0e3)
        assert ps1.allowed_band_hz == (5.0e6, 15.0e6)
        # Both branches same
        assert ps2.min_separation_hz == ps1.min_separation_hz
        assert ps2.min_distance_from_target_hz == ps1.min_distance_from_target_hz

    def test_v01_without_poles_uses_defaults(self, tmp_path: Path) -> None:
        """v0.1 without 'poles:' → defaults for both branches."""
        path = self._write(tmp_path, _V01_WITHOUT_POLES)
        spec = load_project(path)
        assert spec.schema_version == "0.2"
        assert spec.topology.pole_spec_branch1.mode.value == "auto"
        assert spec.topology.pole_spec_branch2.mode.value == "auto"

    def test_v02_with_legacy_poles_raises(self, tmp_path: Path) -> None:
        """v0.2 with legacy 'poles:' key raises SchemaVersionError."""
        path = self._write(tmp_path, _V02_CONFLICTING)
        with pytest.raises(SchemaVersionError, match="Conflicting pole specifications"):
            load_project(path)

    def test_v02_branch_specific_round_trip(self):
        """v0.2 with branch specific fields can be round-tripped."""
        from foster_eom.domain.topology import PoleMode, PoleSpec, TopologySearchSpec

        spec = TopologySearchSpec(
            orientations=["schmidt_shunt_then_series"],
            pole_spec_branch1=PoleSpec(mode=PoleMode.FIXED, fixed_poles_hz=[10e6]),
            pole_spec_branch2=PoleSpec(
                mode=PoleMode.INTERVALS, intervals=[{"min_hz": 1e6, "max_hz": 2e6}]
            ),
        )
        data = spec.model_dump(exclude_unset=True)
        assert "pole_spec" not in data
        assert "pole_spec_branch1" in data
        assert "pole_spec_branch2" in data

        spec_loaded = TopologySearchSpec.model_validate(data)
        assert spec_loaded.pole_spec_branch1.mode == PoleMode.FIXED
        assert spec_loaded.pole_spec_branch2.mode == PoleMode.INTERVALS
