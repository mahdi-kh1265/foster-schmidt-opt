"""Prompt 05 unit tests — dedup.py."""

from __future__ import annotations

import pytest

from foster_eom.optimize.dedup import (
    deb_better,
    deduplicate_basins,
    rms_distance,
)
from foster_eom.optimize.evaluator import EvaluationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    *,
    x: tuple[float, ...] = (0.5,),
    feasible: bool = True,
    v_max: float = 0.0,
    v_sum: float = 0.0,
    objective_value: float = 0.5,
    near_feasible: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        x=x,
        objective_value=objective_value,
        base_objective_value=objective_value,
        soft_penalty_total=0.0,
        objective_terms={"total": objective_value},
        hard_margins=(),
        soft_penalties={},
        v_max=v_max,
        v_sum=v_sum,
        feasible=feasible,
        near_feasible=near_feasible,
        numerical_status="ok",
        numerical_failure_reason=None,
        failed_frequency_hz=None,
        failed_stage=None,
        all_solutions=(),
        target_solutions=(),
        coarse_evaluated=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDebKey:
    def test_feasible_beats_infeasible(self) -> None:
        feas = _result(feasible=True, objective_value=100.0)
        infeas = _result(feasible=False, v_max=0.01, v_sum=0.01, objective_value=0.001)
        assert deb_better(feas, infeas)

    def test_lower_vmax_wins_among_infeasible(self) -> None:
        a = _result(feasible=False, v_max=0.1, v_sum=0.1, objective_value=1.0)
        b = _result(feasible=False, v_max=0.5, v_sum=0.5, objective_value=0.1)
        assert deb_better(a, b)

    def test_lower_vsum_wins_when_vmax_equal(self) -> None:
        a = _result(feasible=False, v_max=0.1, v_sum=0.1, objective_value=1.0)
        b = _result(feasible=False, v_max=0.1, v_sum=0.5, objective_value=0.1)
        assert deb_better(a, b)

    def test_lower_objective_wins_among_feasible(self) -> None:
        a = _result(feasible=True, objective_value=0.1)
        b = _result(feasible=True, objective_value=0.5)
        assert deb_better(a, b)
        assert not deb_better(b, a)

    def test_identical_returns_false(self) -> None:
        a = _result(feasible=True, objective_value=0.3)
        b = _result(feasible=True, objective_value=0.3)
        assert not deb_better(a, b)
        assert not deb_better(b, a)


class TestRmsDistance:
    def test_identical_is_zero(self) -> None:
        assert rms_distance((0.5, 0.5), (0.5, 0.5)) == 0.0

    def test_unit_distance_in_1d(self) -> None:
        assert rms_distance((0.0,), (1.0,)) == pytest.approx(1.0)

    def test_unit_diagonal_in_2d(self) -> None:
        # sqrt(mean([1,1])) = 1.0
        assert rms_distance((0.0, 0.0), (1.0, 1.0)) == pytest.approx(1.0)

    def test_half_step_in_2d(self) -> None:
        # sqrt(mean([0.25, 0.25])) = 0.5
        assert rms_distance((0.0, 0.0), (0.5, 0.5)) == pytest.approx(0.5)


class TestDeduplicateBasins:
    def test_empty_returns_empty(self) -> None:
        assert deduplicate_basins([], 0.1) == []

    def test_single_candidate_one_basin(self) -> None:
        r = _result(x=(0.5,))
        basins = deduplicate_basins([r], 0.1)
        assert len(basins) == 1
        assert basins[0].representative is r

    def test_two_close_candidates_merge_into_one_basin(self) -> None:
        r1 = _result(x=(0.5,), objective_value=0.3)
        r2 = _result(x=(0.51,), objective_value=0.4)
        basins = deduplicate_basins([r1, r2], radius=0.1)
        assert len(basins) == 1
        # Best Deb member should be representative
        assert basins[0].representative.objective_value == 0.3

    def test_two_far_candidates_two_basins(self) -> None:
        r1 = _result(x=(0.1,), objective_value=0.3)
        r2 = _result(x=(0.9,), objective_value=0.4)
        basins = deduplicate_basins([r1, r2], radius=0.05)
        assert len(basins) == 2

    def test_infeasible_into_basin_after_feasible(self) -> None:
        """Infeasible candidates cluster into basins of nearby feasible."""
        feas = _result(x=(0.5,), feasible=True, objective_value=0.3)
        infeas = _result(x=(0.52,), feasible=False, v_max=0.1, objective_value=0.0)
        basins = deduplicate_basins([feas, infeas], radius=0.1)
        # Should be 1 basin; feasible should be representative
        assert len(basins) == 1
        assert basins[0].representative is feas

    def test_basins_sorted_best_first(self) -> None:
        r1 = _result(x=(0.1,), objective_value=0.5)
        r2 = _result(x=(0.9,), objective_value=0.2)
        basins = deduplicate_basins([r1, r2], radius=0.05)
        assert len(basins) == 2
        # Best objective should be first
        assert basins[0].representative.objective_value < basins[1].representative.objective_value

    def test_representative_updates_to_better_member(self) -> None:
        """When a Deb-better candidate is added to a basin, it becomes rep."""
        r1 = _result(x=(0.5,), objective_value=0.6)
        r2 = _result(x=(0.5,), objective_value=0.4)  # same x, better obj
        basins = deduplicate_basins([r1, r2], radius=0.0)  # radius=0: exact match
        assert len(basins) == 1
        assert basins[0].representative.objective_value == 0.4
