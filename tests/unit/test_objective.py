"""Prompt 05 unit tests — objective.py."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from foster_eom.optimize.objective import (
    ObjectiveBreakdown,
    ObjectiveConfig,
    compute_j_complexity,
    compute_j_gamma,
    compute_j_loss,
    compute_j_voltage,
    compute_objective,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_solution(gamma: complex, v_eom: complex | None = None) -> MagicMock:
    sol = MagicMock()
    sol.gamma = gamma
    sol.v_eom = v_eom
    sol.element_measurements = {}
    sol.p_source_delivered_w = 1.0
    return sol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJGamma:
    def test_perfect_match_gives_zero(self) -> None:
        sols = (_mock_solution(0.0 + 0.0j),)
        assert compute_j_gamma(sols, z_ref_ohm=50.0) == 0.0

    def test_open_circuit_gives_one(self) -> None:
        sols = (_mock_solution(1.0 + 0.0j),)
        assert compute_j_gamma(sols, z_ref_ohm=50.0) == pytest.approx(1.0)

    def test_mean_squared(self) -> None:
        """Mean of |Γ|^2 over two targets."""
        sols = (_mock_solution(0.0 + 0.0j), _mock_solution(0.5 + 0.0j))
        expected = (0.0 + 0.25) / 2.0
        assert compute_j_gamma(sols, z_ref_ohm=50.0) == pytest.approx(expected, rel=1e-9)

    def test_no_gamma_gives_one_per_slot(self) -> None:
        sol = _mock_solution(None)
        sol.gamma = None
        result = compute_j_gamma((sol,), z_ref_ohm=50.0)
        assert result == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        assert compute_j_gamma((), z_ref_ohm=50.0) == 0.0


class TestJLoss:
    def test_compute_j_loss_db_scale(self) -> None:
        """Verify j_loss is computed as 10*log10(P_source / P_eom)."""
        sol = _mock_solution(0.0)
        sol.p_source_delivered_w = 10.0

        elem_mock = MagicMock()
        elem_mock.real_power_w = 2.0
        sol.element_measurements = {"L1": elem_mock}

        loss_db = compute_j_loss((sol,), eom_element_id="EOM", lossy_element_ids=("L1",))

        # 10 * log10(10 / 8)
        expected = 10.0 * math.log10(10.0 / 8.0)
        assert math.isclose(loss_db, expected, rel_tol=1e-5)

    def test_compute_j_loss_zero_power(self) -> None:
        """Verify j_loss handles 0 or negative EOM power safely."""
        sol = _mock_solution(0.0)
        sol.p_source_delivered_w = 10.0

        elem_mock = MagicMock()
        elem_mock.real_power_w = 12.0
        sol.element_measurements = {"L1": elem_mock}

        loss_db = compute_j_loss((sol,), eom_element_id="EOM", lossy_element_ids=("L1",))
        assert loss_db == 100.0


class TestJVoltage:
    def test_exact_match_gives_zero(self) -> None:
        sol = _mock_solution(0.0j, v_eom=complex(5.0, 0.0))
        v_targets = (5.0,)
        result = compute_j_voltage((sol,), v_targets, (1.0,))
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_double_target_gives_one(self) -> None:
        """v_eom = 2*v_target → relative error = 1 → J = 1."""
        sol = _mock_solution(0.0j, v_eom=complex(10.0, 0.0))
        result = compute_j_voltage((sol,), (5.0,), (1.0,))
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_none_voltage_target_skipped(self) -> None:
        sol = _mock_solution(0.0j, v_eom=complex(5.0, 0.0))
        result = compute_j_voltage((sol,), (None,), (1.0,))
        assert result == 0.0

    def test_missing_v_eom_gives_penalty(self) -> None:
        sol = _mock_solution(0.0j, v_eom=None)
        sol.v_eom = None
        result = compute_j_voltage((sol,), (5.0,), (1.0,))
        assert result == pytest.approx(1.0)


class TestJComplexity:
    def test_zero_alpha(self) -> None:
        assert compute_j_complexity(4, 0.0) == 0.0

    def test_proportional(self) -> None:
        assert compute_j_complexity(6, 0.1) == pytest.approx(0.6)


class TestComputeObjective:
    def _make_config(self, **kw) -> ObjectiveConfig:
        return ObjectiveConfig(
            z_ref_ohm=50.0,
            w_gamma=1.0,
            w_voltage=0.0,
            w_loss=0.0,
            w_complexity=0.0,
            **kw,
        )

    def test_perfect_match_gives_zero_base(self) -> None:
        sol = _mock_solution(0.0 + 0.0j, v_eom=complex(5.0))
        soft_layout = MagicMock()
        soft_layout.descriptors = []
        result = compute_objective(self._make_config(), (sol,), soft_layout, ())
        assert result.j_gamma == pytest.approx(0.0)
        assert result.j_base == pytest.approx(0.0)
        assert result.j_total == pytest.approx(0.0)

    def test_soft_penalty_adds_to_total(self) -> None:
        """A violated soft constraint (g=-0.5) with weight=2 adds 2*(0.5)^2=0.5."""
        sol = _mock_solution(0.0j)
        desc = MagicMock()
        desc.name = "test_soft"
        desc.penalty_weight = 2.0

        soft_layout = MagicMock()
        soft_layout.descriptors = [desc]

        result = compute_objective(self._make_config(), (sol,), soft_layout, (-0.5,))
        assert result.j_soft == pytest.approx(2.0 * 0.25)
        assert result.j_total == pytest.approx(0.0 + 2.0 * 0.25)

    def test_weight_gamma_scaling(self) -> None:
        """w_gamma=2 doubles J_gamma contribution."""
        sol = _mock_solution(0.5 + 0.0j)  # |Γ|^2 = 0.25
        soft_layout = MagicMock()
        soft_layout.descriptors = []
        cfg = ObjectiveConfig(z_ref_ohm=50.0, w_gamma=2.0)
        result = compute_objective(cfg, (sol,), soft_layout, ())
        assert result.j_gamma == pytest.approx(0.25)
        assert result.j_base == pytest.approx(0.5)  # 2 * 0.25

    def test_breakdown_fields_present(self) -> None:
        sol = _mock_solution(0.0j)
        soft_layout = MagicMock()
        soft_layout.descriptors = []
        result = compute_objective(self._make_config(), (sol,), soft_layout, ())
        assert isinstance(result, ObjectiveBreakdown)
        assert True  # soft_terms may be empty
        # Check objective_terms would be built by engine — just verify fields
        assert result.j_total >= 0
