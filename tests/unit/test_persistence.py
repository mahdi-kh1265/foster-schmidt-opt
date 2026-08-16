"""Tests for YAML persistence and provenance.

Covers project round-trip serialization, loading the example YAML,
project hash stability, and provenance manifest creation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from foster_eom.domain.eom import EOMModelSpec, EOMModelType, MotionalBranch
from foster_eom.domain.frequency_plan import FrequencyPlan, FrequencyTarget
from foster_eom.domain.project import ProjectSpec
from foster_eom.domain.provenance import (
    RunManifest,
    collect_dependency_versions,
    hash_dict,
    hash_file,
    hash_string,
)
from foster_eom.domain.source import SourceMode, SourceSpec
from foster_eom.errors import ProjectValidationError, SchemaVersionError
from foster_eom.persistence.yaml_io import load_project, save_project

# Path to example YAML in the fs-theo handoff package
EXAMPLE_YAML = Path(__file__).parent.parent.parent / "fs-theo" / "examples" / "design_spec.example.yaml"


def _make_test_project() -> ProjectSpec:
    """Create a minimal valid ProjectSpec for round-trip tests."""
    return ProjectSpec(
        source=SourceSpec(
            mode=SourceMode.AVAILABLE_POWER,
            available_power_dbm=20.0,
        ),
        eom=EOMModelSpec(
            model_type=EOMModelType.MBVD,
            name="SYNTHETIC_TEST_ONLY",
            c0_f=12e-12,
            g0_s=2e-5,
            rs_ohm=0.5,
            ls_h=15e-9,
            motional_branches=[
                MotionalBranch(rm_ohm=8.0, lm_h=50e-6, cm_f=9e-12),
            ],
            validity_hz=(1e6, 30e6),
        ),
        frequencies=FrequencyPlan(
            targets=[
                FrequencyTarget(label="f1", frequency_hz=9e6, voltage_target_rms_v=20.0),
                FrequencyTarget(label="f2", frequency_hz=10e6, voltage_target_rms_v=20.0),
                FrequencyTarget(label="f3", frequency_hz=11e6, voltage_target_rms_v=20.0),
            ],
            sweep_f_min_hz=5e6,
            sweep_f_max_hz=15e6,
        ),
    )


# ---------------------------------------------------------------------------
# YAML round-trip tests
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        """Save a project, reload it, and compare key fields."""
        spec = _make_test_project()
        path = tmp_path / "test_project.fseom.yaml"
        save_project(spec, path)

        loaded = load_project(path)

        # Source
        assert loaded.source.mode == spec.source.mode
        assert loaded.source.available_power_dbm == spec.source.available_power_dbm
        assert loaded.source.z_source_real_ohm == spec.source.z_source_real_ohm

        # EOM
        assert loaded.eom.model_type == spec.eom.model_type
        assert loaded.eom.c0_f == pytest.approx(spec.eom.c0_f)
        assert len(loaded.eom.motional_branches) == len(spec.eom.motional_branches)

        # Frequencies
        assert len(loaded.frequencies.targets) == len(spec.frequencies.targets)
        for orig, loaded_t in zip(
            spec.frequencies.enabled_targets, loaded.frequencies.enabled_targets, strict=False
        ):
            assert loaded_t.frequency_hz == pytest.approx(orig.frequency_hz)
            assert loaded_t.voltage_target_rms_v == pytest.approx(orig.voltage_target_rms_v)

        # Matching
        assert loaded.matching.gamma_max == pytest.approx(spec.matching.gamma_max)

        # Schema version preserved
        assert loaded.schema_version == spec.schema_version

    def test_round_trip_thevenin_source(self, tmp_path: Path) -> None:
        """Round-trip a Thévenin source."""
        spec = ProjectSpec(
            source=SourceSpec(mode=SourceMode.THEVENIN, thevenin_vrms=7.07),
            eom=EOMModelSpec(model_type=EOMModelType.IDEAL_CAPACITOR, c0_f=10e-12),
            frequencies=FrequencyPlan(
                targets=[FrequencyTarget(frequency_hz=10e6)],
                sweep_f_min_hz=5e6,
                sweep_f_max_hz=15e6,
            ),
        )
        path = tmp_path / "thevenin.fseom.yaml"
        save_project(spec, path)
        loaded = load_project(path)
        assert loaded.source.mode == SourceMode.THEVENIN
        assert loaded.source.thevenin_vrms == pytest.approx(7.07)


# ---------------------------------------------------------------------------
# Example YAML loading
# ---------------------------------------------------------------------------

class TestExampleYAML:
    @pytest.mark.skipif(
        not EXAMPLE_YAML.exists(),
        reason="Example YAML not found at expected location",
    )
    def test_load_example(self) -> None:
        """Load the supplied example YAML and verify key fields."""
        spec = load_project(EXAMPLE_YAML)

        assert spec.schema_version == "0.1"
        assert spec.project.name == "Synthetic 9-10-11 MHz regression example"
        assert spec.source.mode == SourceMode.AVAILABLE_POWER
        assert spec.source.available_power_dbm == pytest.approx(20.0)
        assert spec.eom.model_type == EOMModelType.MBVD
        assert spec.eom.name == "SYNTHETIC_TEST_ONLY"
        assert len(spec.frequencies.targets) == 3
        freqs = spec.frequencies.target_frequencies_hz
        assert freqs[0] == pytest.approx(9e6)
        assert freqs[1] == pytest.approx(10e6)
        assert freqs[2] == pytest.approx(11e6)

    @pytest.mark.skipif(
        not EXAMPLE_YAML.exists(),
        reason="Example YAML not found at expected location",
    )
    def test_example_round_trip(self, tmp_path: Path) -> None:
        """Load the example, save it, reload, and compare."""
        spec = load_project(EXAMPLE_YAML)
        path = tmp_path / "re_saved.fseom.yaml"
        save_project(spec, path)
        reloaded = load_project(path)
        assert reloaded.schema_version == spec.schema_version
        assert len(reloaded.frequencies.targets) == len(spec.frequencies.targets)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestYAMLErrors:
    def test_unsupported_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_version.fseom.yaml"
        path.write_text("schema_version: '99.0'\n")
        with pytest.raises(SchemaVersionError):
            load_project(path)

    def test_invalid_yaml_content(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_content.fseom.yaml"
        path.write_text("not_a_mapping\n")
        with pytest.raises(ProjectValidationError):
            load_project(path)


# ---------------------------------------------------------------------------
# Provenance and hashing
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_run_manifest_creation(self) -> None:
        m = RunManifest(random_seed=42)
        assert m.run_id  # UUID generated
        assert m.random_seed == 42
        assert m.software_version
        assert m.python_version
        assert m.start_time

    def test_run_manifest_serialization(self) -> None:
        m = RunManifest(random_seed=42)
        d = m.model_dump()
        assert d["random_seed"] == 42
        # Round-trip through dict
        m2 = RunManifest(**d)
        assert m2.run_id == m.run_id

    def test_dependency_versions(self) -> None:
        v = collect_dependency_versions()
        assert "numpy" in v
        assert "scipy" in v

    def test_hash_string_deterministic(self) -> None:
        h1 = hash_string("hello world")
        h2 = hash_string("hello world")
        assert h1 == h2

    def test_hash_string_different(self) -> None:
        h1 = hash_string("hello")
        h2 = hash_string("world")
        assert h1 != h2

    def test_hash_dict_key_order_independent(self) -> None:
        """Dict hash must be stable regardless of insertion order."""
        d1 = {"b": 2, "a": 1, "c": 3}
        d2 = {"a": 1, "c": 3, "b": 2}
        assert hash_dict(d1) == hash_dict(d2)

    def test_hash_dict_different_values(self) -> None:
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert hash_dict(d1) != hash_dict(d2)

    def test_hash_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("test content")
        h1 = hash_file(str(f))
        h2 = hash_file(str(f))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_project_hash_stability(self) -> None:
        """Same project must produce same hash across serializations."""
        spec = _make_test_project()
        d1 = spec.model_dump()
        d2 = spec.model_dump()
        assert hash_dict(d1) == hash_dict(d2)

    @given(st.text(min_size=1, max_size=100))
    def test_hash_string_hypothesis(self, text: str) -> None:
        """Property: same input always produces same hash."""
        assert hash_string(text) == hash_string(text)
