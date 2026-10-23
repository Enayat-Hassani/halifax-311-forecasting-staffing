"""Staffing sensitivity probe.

Perturbs actual agent-queue volume to a fixed WAPE and measures how often the integer staffing
requirement changes. No model is fitted. The probe reads open intervals of the primary
training window, so the test window stays untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import window
from src.erlang import requirement_lookup

WAPE_LEVELS = (0.05, 0.10, 0.20)
SEEDS = range(20)

PRIMARY = dict(
    model="a",
    target=0.80,
    answer_seconds=25.0,
    service_seconds=237.0,
    patience_seconds=120.0,
)
ROBUSTNESS = {
    "target 75%": dict(PRIMARY, target=0.75),
    "patience 60 s": dict(PRIMARY, patience_seconds=60.0),
    "patience 300 s": dict(PRIMARY, patience_seconds=300.0),
    "service 360 s": dict(PRIMARY, service_seconds=360.0),
    "Erlang C": dict(PRIMARY, model="c", patience_seconds=None),
}


def perturb(actual: np.ndarray, wape: float, rng: np.random.Generator) -> np.ndarray:
    """Return actual * (1 + k*z) with z standard normal, clipped to +-3.5, and k set so
    WAPE equals `wape`. Without the clip a 20% WAPE (k ~ 0.25) goes negative on a 4-sigma draw."""
    z = np.clip(rng.standard_normal(len(actual)), -3.5, 3.5)
    k = wape * actual.sum() / (actual * np.abs(z)).sum()
    forecast = actual * (1.0 + k * z)
    if (forecast < 0).any():
        raise ValueError("perturbation produced a negative forecast")
    return forecast


def compare_staffing(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    model: str,
    target: float,
    answer_seconds: float,
    service_seconds: float,
    patience_seconds: float | None = None,
    interval_seconds: float = 1800.0,
) -> dict:
    """Compare required agents from `forecast` with those from `actual`, interval by interval."""
    lookup = requirement_lookup(
        model, target, answer_seconds, service_seconds, patience_seconds, interval_seconds
    )
    diff = lookup(forecast) - lookup(actual)
    return {
        "change_rate": float((diff != 0).mean()),
        "within_one_rate": float((np.abs(diff) <= 1).mean()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "mean_signed_diff": float(diff.mean()),
    }


def to_hourly(series: pd.DataFrame) -> np.ndarray:
    """Sum the two half-hours of each open hour; drop hours with only one half open."""
    hours = series["demand"].groupby(series.index.floor("h")).agg(["sum", "count"])
    return hours.loc[hours["count"] == 2, "sum"].to_numpy(dtype=float)


def load_probe_series(path: Path) -> pd.DataFrame:
    """Open intervals of the primary training window."""
    primary = window(pd.read_parquet(path), "primary")
    return primary[primary["is_open"]]


def _sweep(actual: np.ndarray, interval_seconds: float, staffing: dict) -> dict:
    """Per WAPE level, the comparison fields for every seed."""
    out = {}
    for wape in WAPE_LEVELS:
        runs = [
            compare_staffing(
                actual,
                perturb(actual, wape, np.random.default_rng(seed)),
                interval_seconds=interval_seconds,
                **staffing,
            )
            for seed in SEEDS
        ]
        out[wape] = runs
    return out


def _stat(runs: list[dict], key: str) -> str:
    values = [r[key] for r in runs]
    return f"{np.mean(values):.3f} ({min(values):.3f}–{max(values):.3f})"


def _table(sweep: dict) -> list[str]:
    lines = [
        "| WAPE | change rate | within one agent | mean abs diff (agents) | mean signed diff |",
        "|---|---|---|---|---|",
    ]
    for wape, runs in sweep.items():
        cells = [_stat(runs, k) for k in
                 ("change_rate", "within_one_rate", "mean_abs_diff", "mean_signed_diff")]
        lines.append(f"| {wape:.0%} | " + " | ".join(cells) + " |")
    return lines


def run_probe(
    series_path: Path = Path("data/processed/series.parquet"),
    out: Path = Path("results/staffing_probe.md"),
) -> dict:
    series = load_probe_series(series_path)
    half_hourly = series["demand"].to_numpy(dtype=float)
    grains = {
        "half-hourly": (half_hourly, 1800.0),
        "ten-fold": (half_hourly * 10, 1800.0),
        "hourly": (to_hourly(series), 3600.0),
    }

    results = {
        name: _sweep(actual, interval, PRIMARY) for name, (actual, interval) in grains.items()
    }
    robustness = {
        name: _sweep(half_hourly, 1800.0, staffing) for name, staffing in ROBUSTNESS.items()
    }
    lines = [
        "# Staffing sensitivity probe",
        "",
        "Synthetic sensitivity test on open intervals from primary training "
        "(2024-07-17 to 2025-08-31). Each cell is the mean across 20 seeds, "
        "with the min–max in brackets.",
        "Perturbations use independent, clipped normal errors scaled to each WAPE level. "
        "They show how staffing thresholds respond to volume errors; they do not represent "
        "forecast errors or validate a staffing plan.",
        "",
        "Primary configuration: Erlang A, 80% answered within 25 s, patience 120 s, "
        "service time 237 s (mean talk time in the window). These are scenario values, "
        "not Halifax facts.",
        "",
    ]
    for name, (actual, interval) in grains.items():
        req = requirement_lookup(
            PRIMARY["model"], PRIMARY["target"], PRIMARY["answer_seconds"],
            PRIMARY["service_seconds"], PRIMARY["patience_seconds"], interval,
        )(actual)
        lines += [
            f"## {name.capitalize()}",
            "",
            f"{len(actual):,} staffed intervals. Actual-implied requirement: "
            f"mean {req.mean():.1f}, peak {req.max()} agents.",
            "",
            *_table(results[name]),
            "",
        ]
    lines += ["## Robustness (half-hourly, one parameter changed)", ""]
    for name, sweep in robustness.items():
        lines += [f"### {name}", "", *_table(sweep), ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return {"results": results, "robustness": robustness}


if __name__ == "__main__":
    run_probe()
