"""Erlang tests."""

import heapq
import math

import numpy as np
import pytest

from src.erlang import erlang_b, required_agents, service_level_a, service_level_c

SERVICE = 242.0
ANSWER = 25.0


def _factorial_wait_probability(load: float, agents: int) -> float:
    """Textbook Erlang C, written independently of the module's recurrence."""
    top = load**agents / math.factorial(agents) * agents / (agents - load)
    bottom = sum(load**k / math.factorial(k) for k in range(agents)) + top
    return top / bottom


def _factorial_service_level(load: float, agents: int) -> float:
    wait = _factorial_wait_probability(load, agents)
    return 1.0 - wait * math.exp(-(agents - load) * ANSWER / SERVICE)


def _simulate_service_level(
    load: float, agents: int, patience: float, arrivals: int, seed: int
) -> float:
    """FIFO M/M/N+M: a caller abandons if the wait it would face exceeds its patience."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(SERVICE / load, arrivals)
    service = rng.exponential(SERVICE, arrivals)
    patience_draws = rng.exponential(patience, arrivals)
    free_at = [0.0] * agents
    heapq.heapify(free_at)
    clock = 0.0
    answered_in_time = 0
    for gap, svc, pat in zip(gaps, service, patience_draws):
        clock += gap
        wait = max(0.0, free_at[0] - clock)
        if wait <= pat:
            heapq.heapreplace(free_at, clock + wait + svc)
            if wait <= ANSWER:
                answered_in_time += 1
    return answered_in_time / arrivals


def test_erlang_b_matches_published_values():
    assert erlang_b(5, 5) == pytest.approx(0.28487, abs=5e-5)
    assert erlang_b(10, 10) == pytest.approx(0.21459, abs=5e-5)


@pytest.mark.parametrize("load, agents", [(5.0, 7), (8.0, 10), (20.0, 23)])
def test_service_level_c_matches_factorial_formula(load, agents):
    got = service_level_c(load, agents, service_seconds=SERVICE, answer_seconds=ANSWER)
    assert got == pytest.approx(_factorial_service_level(load, agents), abs=1e-12)


def test_service_level_c_is_zero_without_a_steady_state():
    assert service_level_c(8.0, 8, service_seconds=SERVICE, answer_seconds=ANSWER) == 0.0


def test_service_level_a_approaches_erlang_c_as_patience_grows():
    a = service_level_a(
        8.0, 10, service_seconds=SERVICE, answer_seconds=ANSWER, patience_seconds=1e7
    )
    c = service_level_c(8.0, 10, service_seconds=SERVICE, answer_seconds=ANSWER)
    assert a == pytest.approx(c, abs=1e-4)


def test_service_level_a_matches_simulation_under_heavy_queueing():
    load, agents, patience = 8.0, 8, 60.0
    got = service_level_a(
        load,
        agents,
        service_seconds=SERVICE,
        answer_seconds=ANSWER,
        patience_seconds=patience,
    )
    simulated = _simulate_service_level(load, agents, patience, 300_000, seed=7)
    assert got == pytest.approx(simulated, abs=0.01)


def test_service_level_a_is_monotone_in_agents_and_load():
    def sl(load, agents):
        return service_level_a(
            load,
            agents,
            service_seconds=SERVICE,
            answer_seconds=ANSWER,
            patience_seconds=120.0,
        )

    by_agents = [sl(8.0, n) for n in range(4, 14)]
    by_load = [sl(load, 10) for load in (4.0, 6.0, 8.0, 10.0, 12.0)]
    assert by_agents == sorted(by_agents)
    assert by_load == sorted(by_load, reverse=True)


@pytest.mark.parametrize("calls", [10, 45, 120])
def test_required_agents_c_is_smallest_count_meeting_target(calls):
    target = 0.8
    load = calls * SERVICE / 1800.0
    expected = next(
        n
        for n in range(int(load) + 1, 500)
        if _factorial_service_level(load, n) >= target
    )
    got = required_agents(
        calls,
        model="c",
        target=target,
        answer_seconds=ANSWER,
        service_seconds=SERVICE,
    )
    assert got == expected


def test_required_agents_a_sits_on_the_target_boundary():
    kwargs = dict(
        service_seconds=SERVICE, answer_seconds=ANSWER, patience_seconds=120.0
    )
    calls = 60
    load = calls * SERVICE / 1800.0
    n = required_agents(calls, model="a", target=0.8, **kwargs)
    assert service_level_a(load, n, **kwargs) >= 0.8
    assert service_level_a(load, n - 1, **kwargs) < 0.8


def test_required_agents_is_zero_for_zero_calls():
    assert (
        required_agents(
            0,
            model="a",
            target=0.8,
            answer_seconds=ANSWER,
            service_seconds=SERVICE,
            patience_seconds=120.0,
        )
        == 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(model="c", target=1.0),
        dict(model="c", target=0.0),
        dict(model="a", target=0.8),
        dict(model="c", target=0.8, patience_seconds=120.0),
    ],
)
def test_required_agents_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        required_agents(
            10, answer_seconds=ANSWER, service_seconds=SERVICE, **kwargs
        )
