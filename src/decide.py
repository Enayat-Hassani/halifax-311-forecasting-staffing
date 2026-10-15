"""Decision layer: staff each forecast with Erlang and cost the gap to the actual requirement.

Each run is staffed three ways: from its point forecast; from the point forecast times one
uplift; and from the demand quantile by slot type. The uplift and the quantile are both set
by the cost ratio r = c_u / c_o at q = r / (1 + r), from forecast errors on 26 weekly origins
inside training, so the test window is only scored. Costs are agent-hours in units of the
overstaffing cost. A requirement is productive agents per half-hour, not a roster.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.erlang import requirement_lookup, service_level_a
from src.evaluate import RUNS, backtest, peak_keys, slot_key, splits
from src.probe import PRIMARY, ROBUSTNESS

CALIBRATION = pd.date_range("2025-02-24", "2025-08-18", freq="W-MON")
RATIOS = 2.0 ** np.arange(-2, 4.01, 0.25)
TABLE_RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SCENARIOS = {"primary": PRIMARY, **ROBUSTNESS}
METHODS = ("point", "uplift", "quantile")
WEEKS = 52


def slot_types(ts: pd.DatetimeIndex, peak: set) -> pd.MultiIndex:
    keys = slot_key(ts)
    return pd.MultiIndex.from_arrays(
        [[w for w, _ in keys], [k in peak for k in keys]], names=["is_weekend", "is_peak"]
    )


def _ratios(calib: pd.DataFrame, peak: set) -> pd.Series:
    fit = calib[calib["is_open"] & (calib["forecast"] > 0)]
    return pd.Series(
        fit["actual"].to_numpy() / fit["forecast"].to_numpy(),
        index=slot_types(pd.DatetimeIndex(fit["ts"]), peak),
    )


def ratio_quantiles(calib: pd.DataFrame, peak: set, levels: np.ndarray) -> pd.DataFrame:
    """Quantiles of actual / forecast on open slots, by slot type; one column per level."""
    return _ratios(calib, peak).groupby(level=[0, 1]).quantile(list(levels)).unstack()


def _slot_cost(gap: np.ndarray, ratio: float) -> np.ndarray:
    return 0.5 * (ratio * np.clip(-gap, 0, None) + np.clip(gap, 0, None))


def staffing_cost(actual_agents, staffed_agents, ratio: float) -> float:
    """r x short + over, in agent-hours. One agent for one half-hour is 0.5 agent-hours."""
    gap = np.asarray(staffed_agents) - np.asarray(actual_agents)
    return float(_slot_cost(gap, ratio).sum())


def block_bootstrap(weekly: np.ndarray, block: int = 4, draws: int = 4000) -> tuple[float, float]:
    """95% interval of the mean of a weekly series, resampled in blocks of `block` weeks.

    Blocks keep the week-to-week correlation of errors that an iid resample would drop.
    """
    rng = np.random.default_rng(0)
    n = len(weekly)
    starts = rng.integers(0, n - block + 1, size=(draws, -(-n // block)))
    idx = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = weekly[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@lru_cache(maxsize=None)
def _primary_level(calls: float, agents: int) -> float:
    if agents == 0:
        return 0.0
    s = PRIMARY
    return service_level_a(
        calls * s["service_seconds"] / 1800.0,
        agents,
        service_seconds=s["service_seconds"],
        answer_seconds=s["answer_seconds"],
        patience_seconds=s["patience_seconds"],
    )


def implied_service_level(actual: np.ndarray, agents: np.ndarray) -> float:
    """Erlang A service level of `agents` under actual demand, weighted by actual calls."""
    levels = np.array([_primary_level(float(c), int(a)) for c, a in zip(actual, agents)])
    return float((levels * actual).sum() / actual.sum())


def _lookup(scenario: dict):
    return requirement_lookup(**scenario, interval_seconds=1800.0)


def poisson_floor(test: pd.DataFrame, draws: int = 20) -> dict:
    """Error of a forecast that knows the true rate, when demand is Poisson around it.

    The rate is actual demand itself, so the floor has the test year's volume. Real call
    arrivals are overdispersed, so the true floor is higher.
    """
    rate = test.loc[test["is_open"], "actual"].to_numpy()
    lookup = _lookup(PRIMARY)
    staffed = lookup(rate)
    rng = np.random.default_rng(0)
    wape, short, over = [], [], []
    for _ in range(draws):
        calls = rng.poisson(rate).astype(float)
        gap = staffed - lookup(calls)
        wape.append(np.abs(rate - calls).sum() / calls.sum())
        short.append(0.5 * np.clip(-gap, 0, None).sum() / WEEKS)
        over.append(0.5 * np.clip(gap, 0, None).sum() / WEEKS)
    return {"wape": np.mean(wape), "short": np.mean(short), "over": np.mean(over)}


def evaluate_run(test: pd.DataFrame, calib: pd.DataFrame, peak: set) -> dict:
    """Point, uplift and quantile staffing of one run's open test slots, every scenario.

    Costs are weekly arrays with shape (len(RATIOS), WEEKS).
    """
    # Post-closure days are found from the closed days, so split before dropping them.
    is_open = test["is_open"].to_numpy()
    masks = {k: m[is_open] for k, m in splits(test, peak).items()}
    test = test[is_open]
    actual, point = test["actual"].to_numpy(), test["forecast"].to_numpy()
    week = pd.factorize(test["origin"], sort=True)[0]
    levels = RATIOS / (1 + RATIOS)
    ts = pd.DatetimeIndex(test["ts"])
    by_type = ratio_quantiles(calib, peak, levels).reindex(slot_types(ts, peak)).to_numpy()
    if np.isnan(by_type).any():
        raise ValueError("a test slot type has no calibration slots")
    uplift = np.quantile(_ratios(calib, peak).to_numpy(), levels)
    staffing = {
        "point": np.repeat(point[:, None], len(RATIOS), axis=1),
        "uplift": point[:, None] * uplift,
        "quantile": point[:, None] * by_type,
    }
    covered = pd.DataFrame(actual[:, None] <= staffing["quantile"], columns=levels)
    out = {
        "wape": np.abs(point - actual).sum() / actual.sum(),
        "bias": (point - actual).sum() / actual.sum(),
        "uplift": uplift,
        # Share of actual demand at or below each quantile forecast: 0.5 means calibrated at q = 0.5.
        "coverage": covered.mean().to_numpy(),
        "coverage_quarter": covered.groupby(ts.to_period("Q").astype(str)).mean(),
    }
    for name, scenario in SCENARIOS.items():
        lookup = _lookup(scenario)
        need = lookup(actual)
        gap = lookup(point) - need
        out[name] = {
            "error": {k: 0.5 * np.abs(gap[m]).sum() / WEEKS for k, m in masks.items()},
            "short": 0.5 * np.clip(-gap, 0, None).sum() / WEEKS,
            "over": 0.5 * np.clip(gap, 0, None).sum() / WEEKS,
            "understaffed": {k: (gap[m] < 0).mean() for k, m in masks.items()},
            **{
                method: np.array([
                    np.bincount(week, _slot_cost(lookup(f[:, j]) - need, r), minlength=WEEKS)
                    for j, r in enumerate(RATIOS)
                ])
                for method, f in staffing.items()
            },
        }
        if name == "primary":
            out[name]["service_level"] = implied_service_level(actual, lookup(point))
    return out


def calibrate(series: pd.DataFrame, out_dir: Path) -> None:
    """Write each run's backtest on the calibration origins, once."""
    for model, train_window in RUNS:
        path = out_dir / "calibration" / f"{model.name}_{train_window}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            backtest(model, series, train_window, at=CALIBRATION).to_parquet(path)


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="min").astype(int)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def main(
    series_path: Path = Path("data/processed/series.parquet"),
    out_dir: Path = Path("results"),
) -> Path:
    series = pd.read_parquet(series_path)
    peak = peak_keys(series)
    calibrate(series, out_dir)

    results, primary_runs = {}, []
    for model, train_window in RUNS:
        label = "any" if getattr(model, "context", None) else train_window
        run = f"{model.name} ({label})"
        file = f"{model.name}_{train_window}.parquet"
        test = pd.read_parquet(out_dir / "forecasts" / file)
        results[run] = evaluate_run(test, pd.read_parquet(out_dir / "calibration" / file), peak)
        if train_window == "primary":
            primary_runs.append(run)
    floor = poisson_floor(test)
    runs = list(results)
    primary = {run: r["primary"] for run, r in results.items()}
    wape = pd.Series({run: r["wape"] for run, r in results.items()})
    error = pd.Series({run: p["error"]["all"] for run, p in primary.items()})
    j1, j4 = list(RATIOS).index(1.0), list(RATIOS).index(4.0)
    columns = [list(RATIOS).index(r) for r in TABLE_RATIOS]
    cost = {
        m: pd.DataFrame({run: primary[run][m].sum(axis=1) / WEEKS for run in runs}, index=RATIOS)
        for m in METHODS
    }
    s = PRIMARY

    lines = [
        "# Decision layer",
        "",
        "Generated by `src/decide.py` from the forecasts of `results/baselines.md`: "
        "52 weekly origins, open intervals only. Each forecast is turned into the integer "
        "agent requirement per half-hour and compared with the requirement of actual demand. "
        "Agent-hours are per week; one agent for one half-hour is 0.5 agent-hours. These are "
        "counterfactual requirements, not rosters and not HRM costs.",
        "",
        "| Input | Value | Source |",
        "|---|---|---|",
        f"| Service time | {s['service_seconds']:.0f} s | Mean talk time, primary training (data) |",
        f"| Service level | {s['target']:.0%} in {s['answer_seconds']:.0f} s | HRM documents; no current target published (assumption) |",
        f"| Patience | {s['patience_seconds']:.0f} s | Not identifiable from the data (assumption) |",
        "| Queue model | Erlang A | Abandonment is recorded (choice) |",
        "| Cost ratio r = c_u / c_o | 0.25 to 16 | Swept, not estimated (assumption) |",
        "",
        "## Accuracy and staffing, point forecast",
        "",
        "Staffed from the point forecast, primary scenario. Service level is the share of "
        "actual calls answered within 25 s by the staffed agents, under Erlang A. The Poisson "
        "oracle floor is a forecast that knows the true rate when demand is Poisson around it; real "
        "arrivals are overdispersed, so the true floor is higher.",
        "",
        "| Run | WAPE | Bias | WAPE rank | Agent-hours error | Short | Over | Error rank | Understaffed | Service level |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wape_rank, error_rank = _rank(wape), _rank(error)
    for run in runs:
        r, p = results[run], primary[run]
        lines.append(
            f"| {run} | {r['wape']:.1%} | {r['bias']:+.1%} | {wape_rank[run]} | "
            f"{p['error']['all']:.1f} | {p['short']:.1f} | {p['over']:.1f} | {error_rank[run]} | "
            f"{p['understaffed']['all']:.1%} | {p['service_level']:.1%} |"
        )
    lines.append(
        f"| Poisson oracle floor | {floor['wape']:.1%} | | | {floor['short'] + floor['over']:.1f} | "
        f"{floor['short']:.1f} | {floor['over']:.1f} | | | |"
    )

    best = error.idxmin()
    lines += [
        "",
        "### Difference from the best run",
        "",
        f"Mean weekly difference from {best}, with a 95% moving-block bootstrap interval "
        "(4-week blocks, 4,000 draws). An interval that contains 0 does not separate the runs.",
        "",
        "| Run | Agent-hours error, point | Cost at r = 4, quantile |",
        "|---|---:|---:|",
    ]
    for run in runs:
        if run == best:
            continue
        cells = []
        for m, j in (("point", j1), ("quantile", j4)):
            diff = primary[run][m][j] - primary[best][m][j]
            lo, hi = block_bootstrap(diff)
            cells.append(f"{diff.mean():+.1f} ({lo:+.1f} to {hi:+.1f})")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### By split",
        "",
        "Agent-hours error per week, and the share of understaffed intervals in brackets.",
        "",
        "| Run | Peak | Off-peak | Post-closure |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        p = primary[run]
        cells = [f"{p['error'][k]:.1f} ({p['understaffed'][k]:.0%})" for k in ("peak", "off_peak", "post_closure")]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")

    relative = cost["quantile"].div(cost["quantile"].min(axis=1), axis=0)
    lines += [
        "",
        "## Cost-ratio sweep",
        "",
        "Each run is staffed three ways at each ratio. Point: the point forecast. Uplift: the "
        "point forecast times the q-quantile of actual / forecast over all open slots, "
        "q = r / (1 + r). Quantile: the same, by slot type (weekday or weekend, peak or "
        "off-peak). Both are fitted on 26 weekly origins from 2025-02-24 to 2025-08-18, inside "
        "training.",
        "",
        "Lowest weekly cost at each ratio, in units of one overstaffed agent-hour.",
        "",
        "| r | q | Uplift (xgboost secondary) | Point | Uplift | Quantile |",
        "|---:|---:|---:|---|---|---|",
    ]
    for j in columns:
        r = RATIOS[j]
        cells = [f"{cost[m].iloc[j].min():.1f} {cost[m].iloc[j].idxmin()}" for m in METHODS]
        k = results["xgboost (secondary)"]["uplift"][j]
        lines.append(f"| {r:g} | {r / (1 + r):.2f} | {k:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Quantile staffing cost relative to the best run at that ratio (1.00 = best).",
        "",
        "| Run | " + " | ".join(f"r = {r:g}" for r in TABLE_RATIOS) + " |",
        "|---|" + "---:|" * len(TABLE_RATIOS),
    ]
    for run in runs:
        lines.append(f"| {run} | " + " | ".join(f"{relative[run].iloc[j]:.2f}" for j in columns) + " |")

    lines += [
        "",
        "### Does accuracy predict cost?",
        "",
        "Spearman rank correlation between WAPE and cost, over all eight runs and over the "
        f"{len(primary_runs)} models on primary training.",
        "",
        "| r | Point, 8 runs | Quantile, 8 runs | Point, 5 models | Quantile, 5 models |",
        "|---:|---:|---:|---:|---:|",
    ]
    for j in columns:
        cells = [
            f"{_spearman(wape[subset], cost[m].iloc[j][subset]):.2f}"
            for subset in (runs, primary_runs)
            for m in ("point", "quantile")
        ]
        lines.append(f"| {RATIOS[j]:g} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    q80 = RATIOS[j4] / (1 + RATIOS[j4])
    lines += [
        "",
        "### Calibration of the quantiles on the test window",
        "",
        "Share of actual demand at or below the quantile forecast. A calibrated quantile "
        "matches its level.",
        "",
        "| Run | " + " | ".join(f"q = {RATIOS[j] / (1 + RATIOS[j]):.2f}" for j in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for run in runs:
        lines.append(f"| {run} | " + " | ".join(f"{results[run]['coverage'][j]:.2f}" for j in columns) + " |")
    quarters = list(results[runs[0]]["coverage_quarter"].index)
    lines += [
        "",
        f"By quarter, q = {q80:.2f}. 2025Q3 is September only and 2026Q3 is July and August.",
        "",
        "| Run | " + " | ".join(quarters) + " |",
        "|---|" + "---:|" * len(quarters),
    ]
    for run in runs:
        by_quarter = results[run]["coverage_quarter"][q80]
        lines.append(f"| {run} | " + " | ".join(f"{v:.2f}" for v in by_quarter) + " |")

    lines += [
        "",
        "## Scenarios",
        "",
        "One input changed at a time. Rank correlation is Spearman between WAPE and agent-hours "
        "error over the eight runs, point staffing.",
        "",
        "| Scenario | Best by WAPE | Best by agent-hours error | Rank correlation | Best, quantile, r = 4 |",
        "|---|---|---|---:|---|",
    ]
    for name in SCENARIOS:
        err = pd.Series({run: results[run][name]["error"]["all"] for run in runs})
        c4 = pd.Series({run: results[run][name]["quantile"][j4].sum() for run in runs})
        lines.append(
            f"| {name} | {wape.idxmin()} | {err.idxmin()} | {_spearman(wape, err):.2f} | {c4.idxmin()} |"
        )
    lines.append("")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in runs:
        ax.plot(RATIOS, relative[run], label=run)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Cost ratio r = understaffing / overstaffing cost")
    ax.set_ylabel("Cost relative to best run")
    ax.set_title("Quantile staffing cost by cost ratio, primary scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_ratio.png", dpi=150)
    plt.close(fig)

    out = out_dir / "decision.md"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
