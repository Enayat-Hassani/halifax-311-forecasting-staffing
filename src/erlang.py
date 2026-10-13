"""Erlang staffing functions: required agents for a call volume and service-level target.

Each interval is treated as an independent steady-state queue with exponential service
and, for Erlang A, exponential patience. Service level is the share of arrivals answered
within `answer_seconds`; abandoned calls count as misses, so Erlang C and Erlang A report
the same quantity.

`load` is offered load in erlangs (arrival rate x service_seconds). All times are seconds.
The functions know nothing about Halifax; the staffing requirement they return is
counterfactual, not a roster.
"""

import math
from typing import Literal

import numpy as np

# Relative tail mass at which the stationary queue distribution is cut off.
_QUEUE_TAIL_TOL = 1e-10


def erlang_b(load: float, agents: int) -> float:
    """Blocking probability for `agents` servers and no queue.

    Uses the recurrence B(k) = a*B(k-1) / (k + a*B(k-1)), which stays stable at
    agent counts where the factorial form overflows.
    """
    b = 1.0
    for k in range(1, agents + 1):
        b = load * b / (k + load * b)
    return b


def _wait_probability_c(load: float, agents: int) -> float:
    b = erlang_b(load, agents)
    return agents * b / (agents - load * (1.0 - b))


def service_level_c(
    load: float, agents: int, *, service_seconds: float, answer_seconds: float
) -> float:
    """Share of arrivals answered within `answer_seconds` under Erlang C.

    Returns 0.0 when `agents <= load`: the queue has no steady state.
    """
    if load == 0:
        return 1.0
    if agents <= load:
        return 0.0
    wait = _wait_probability_c(load, agents)
    return 1.0 - wait * math.exp(-(agents - load) * answer_seconds / service_seconds)


def service_level_a(
    load: float,
    agents: int,
    *,
    service_seconds: float,
    answer_seconds: float,
    patience_seconds: float,
) -> float:
    """Share of arrivals answered within `answer_seconds` under Erlang A (M/M/N+M).

    An arrival finding a free agent is answered at once. One that finds j callers
    queued takes position j+1; that position advances at rate N*mu + (position-1)*theta
    and the caller abandons at rate theta. The probability of reaching the front within
    `answer_seconds` comes from uniformising that chain, weighted by the stationary
    queue-length distribution. Abandoned calls count as misses.
    """
    mu = 1.0 / service_seconds
    theta = 1.0 / patience_seconds
    lam = load * mu

    # Stationary weights relative to the empty system: idle[n] for n < agents,
    # queue[j] for j callers waiting with all agents busy.
    idle = [1.0]
    for n in range(1, agents):
        idle.append(idle[-1] * load / n)
    queue = [idle[-1] * load / agents]
    j = 0
    while True:
        j += 1
        ratio = lam / (agents * mu + j * theta)
        if ratio < 1.0:
            # Ratios fall with j, so the remaining mass is bounded by a geometric tail.
            total = sum(idle) + sum(queue)
            if queue[-1] * ratio / (1.0 - ratio) < _QUEUE_TAIL_TOL * total:
                break
        queue.append(queue[-1] * ratio)
    total = sum(idle) + sum(queue)

    positions = len(queue)
    # answered[k] = P(a caller at queue position k reaches an agent within t).
    # Uniformise the chain on positions 0..K; position 0 is "answered", abandonment
    # is an absorbing miss with value 0.
    k = np.arange(positions + 1)
    advance = agents * mu + (k - 1) * theta
    leave = agents * mu + k * theta
    rate = agents * mu + positions * theta
    stay = 1.0 - leave / rate
    move = advance / rate

    x = rate * answer_seconds
    steps = int(x + 12.0 * math.sqrt(x) + 40)
    m = np.arange(steps + 1)
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(m[1:]))))
    weights = np.exp(-x + m * math.log(x) - log_fact) if x > 0 else np.eye(1, steps + 1)[0]

    v = np.zeros(positions + 1)
    v[0] = 1.0
    answered = np.zeros(positions + 1)
    for w in weights:
        answered += w * v
        nxt = stay * v
        nxt[1:] += move[1:] * v[:-1]
        nxt[0] = 1.0
        v = nxt

    p_queue = np.array(queue) / total
    return sum(idle) / total + float(p_queue @ answered[1:])


def service_level(
    model: Literal["a", "c"],
    load: float,
    agents: int,
    *,
    service_seconds: float,
    answer_seconds: float,
    patience_seconds: float | None = None,
) -> float:
    """Service level under Erlang A (`model="a"`) or Erlang C (`model="c"`)."""
    if model == "c":
        return service_level_c(
            load, agents, service_seconds=service_seconds, answer_seconds=answer_seconds
        )
    return service_level_a(
        load,
        agents,
        service_seconds=service_seconds,
        answer_seconds=answer_seconds,
        patience_seconds=patience_seconds,
    )


def required_agents(
    calls: float,
    *,
    model: Literal["a", "c"],
    target: float,
    answer_seconds: float,
    service_seconds: float,
    patience_seconds: float | None = None,
    interval_seconds: float = 1800.0,
) -> int:
    """Smallest integer agent count whose service level meets `target`.

    `calls` is the arrival count in one interval and may be fractional, since forecasts
    are. Zero calls need zero agents. `patience_seconds` is required for Erlang A
    (`model="a"`) and rejected for Erlang C.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target}")
    if model == "a" and patience_seconds is None:
        raise ValueError("model 'a' needs patience_seconds")
    if model == "c" and patience_seconds is not None:
        raise ValueError("model 'c' takes no patience_seconds")
    if calls < 0:
        raise ValueError(f"calls must be non-negative, got {calls}")
    if calls == 0:
        return 0

    load = calls * service_seconds / interval_seconds
    # Erlang C has no steady state at or below the offered load.
    agents = int(load) + 1 if model == "c" else 1
    while (
        service_level(
            model,
            load,
            agents,
            service_seconds=service_seconds,
            answer_seconds=answer_seconds,
            patience_seconds=patience_seconds,
        )
        < target
    ):
        agents += 1
    return agents
