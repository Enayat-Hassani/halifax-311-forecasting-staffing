"""Pull the Halifax 311 Call Volumes series from the ArcGIS FeatureServer.

Pagination follows exceededTransferLimit, not an assumed page size, because the
service can change how many rows it returns per page. The pull stays raw: no
timezone conversion, no gap filling, no type coercion beyond what parquet
requires. Cleaning is in src/clean.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

SERVICE_URL = (
    "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/"
    "311_Call_Volumes/FeatureServer/0/query"
)
DATASET_URL = "https://data-hrm.hub.arcgis.com/datasets/HRM::311-call-volumes"
LICENSE_URL = "https://data-hrm.hub.arcgis.com/pages/open-data-licence"

QUERY_PARAMS = {"where": "1=1", "outFields": "*", "orderByFields": "ObjectId"}

# The service held 148,269 rows on 2026-09-13. The floor sits below that to
# leave room for revision; it is here to catch a truncated pull, not to assert
# an exact size.
MIN_ROWS = 140_000

# The spring and autumn changeover dates checked by audit_findings. 2026's
# autumn changeover falls after the series ends, so 2025's is used instead.
DST_CHANGEOVERS = {
    "2026-03-08": ("2026-03-07", "2026-03-09"),
    "2025-11-02": ("2025-11-01", "2025-11-03"),
}

# The fields audit_findings profiles for range, nulls and zeros. HANDLED is a
# call count rather than a duration, and is in the list because it is the
# denominator when AVERAGE_TALK_TIME is checked against TOTAL_TALK_TIME.
TALK_TIME_FIELDS = ["HANDLED", "PROCESSED_IN_IVR", "TOTAL_TALK_TIME"]


def _parse_esri_response(response: httpx.Response) -> dict:
    """Parse an ArcGIS JSON response, raising on an embedded error payload.

    Esri FeatureServer endpoints can return HTTP 200 with an
    ``{"error": {...}}`` body for throttling, malformed queries, or
    transient server errors, so a 2xx status doesn't mean the query
    succeeded.
    """
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS service returned an error: {payload['error']}")
    return payload


def service_count(client: httpx.Client) -> int:
    """Row count the service reports for the full table.

    Raises httpx.HTTPStatusError on a non-2xx response, RuntimeError if the
    service reports an error in an otherwise-2xx body.
    """
    response = client.get(
        SERVICE_URL, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}
    )
    response.raise_for_status()
    return _parse_esri_response(response)["count"]


def fetch_pages(client: httpx.Client, page_size: int = 32000) -> Iterator[list[dict]]:
    """Yield pages of attribute dicts, ordered by ObjectId.

    Continues while the response sets exceededTransferLimit, advancing
    resultOffset by the number of features actually returned.

    The default page_size is the standardMaxRecordCount the service
    advertises, which turns the pull into 5 requests instead of 75. Honouring
    it is server-side policy, so the loop reads the flag rather than assuming
    the request was granted. Raises httpx.HTTPStatusError on a non-2xx
    response, RuntimeError if the service reports an error in an
    otherwise-2xx body, or if it sets exceededTransferLimit while returning
    zero features (which would otherwise spin forever at the same offset).
    """
    offset = 0
    while True:
        params = {
            **QUERY_PARAMS,
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        response = client.get(SERVICE_URL, params=params)
        response.raise_for_status()
        payload = _parse_esri_response(response)
        rows = [feature["attributes"] for feature in payload["features"]]
        yield rows
        if not payload.get("exceededTransferLimit", False):
            break
        if not rows:
            raise RuntimeError(
                "service reported exceededTransferLimit with zero features "
                f"returned at resultOffset={offset}; aborting to avoid an "
                "infinite loop"
            )
        offset += len(rows)


def _call_date_to_date(call_date: pd.Series) -> pd.Series:
    # CALL_DATE is epoch ms, offset a few hours from UTC midnight by a
    # server-side timezone quirk unrelated to Halifax. Flooring to UTC date
    # still recovers the intended calendar date, since that offset is always
    # smaller than a day.
    return pd.to_datetime(call_date, unit="ms", utc=True).dt.date


def audit_findings(df: pd.DataFrame) -> dict:
    """Record interval, daylight-saving, and talk-time findings for the audit.

    These findings describe the raw table and do not resolve the remaining
    field-semantics questions in ``data/raw/data_audit.md``.
    """
    day = _call_date_to_date(df["CALL_DATE"])
    null_call_date_count = int(day.isna().sum())

    distinct_intervals = sorted(df["INTERVAL"].dropna().unique().tolist())
    pairs_per_day = (
        df.assign(_day=day)
        .drop_duplicates(subset=["_day", "MILITARY_HOUR", "INTERVAL"])
        .groupby("_day")
        .size()
    )
    interval = {
        "distinct_values": distinct_intervals,
        "n_distinct_values": len(distinct_intervals),
        "max_pairs_per_day": int(pairs_per_day.max()) if len(pairs_per_day) else 0,
        "days_at_max": int((pairs_per_day == pairs_per_day.max()).sum())
        if len(pairs_per_day)
        else 0,
        "n_days": int(len(pairs_per_day)),
        "null_call_date_count": null_call_date_count,
    }

    def rows_on(date_str: str) -> int:
        target = pd.Timestamp(date_str).date()
        return int((day == target).sum())

    dst = {
        changeover: {
            "row_count": rows_on(changeover),
            "day_before": rows_on(before),
            "day_after": rows_on(after),
        }
        for changeover, (before, after) in DST_CHANGEOVERS.items()
    }

    talk_time = {}
    for field in TALK_TIME_FIELDS:
        series = df[field]
        non_null = series.dropna()
        talk_time[field] = {
            "min": None if non_null.empty else int(non_null.min()),
            "max": None if non_null.empty else int(non_null.max()),
            "null_count": int(series.isna().sum()),
            "zero_count": int((series == 0).sum()),
        }

    handled_positive = df["HANDLED"] > 0
    computed_avg = (
        df.loc[handled_positive, "TOTAL_TALK_TIME"] / df.loc[handled_positive, "HANDLED"]
    )
    mismatch = (df.loc[handled_positive, "AVERAGE_TALK_TIME"] - computed_avg).abs() > 1
    talk_time["avg_talk_time_checked_count"] = int(handled_positive.sum())
    talk_time["avg_talk_time_mismatch_count"] = int(mismatch.sum())

    return {"interval": interval, "dst": dst, "talk_time": talk_time}


def _render_audit_section(findings: dict, pulled_date: str) -> str:
    interval = findings["interval"]
    dst = findings["dst"]
    talk_time = findings["talk_time"]

    lines = [f"\n## {pulled_date} — pull audit\n"]

    lines.append("**What `INTERVAL` encodes.**\n")
    lines.append(f"Distinct values: {interval['n_distinct_values']}.")
    lines.append(f"{interval['distinct_values']}\n")
    lines.append(
        f"Max distinct (`MILITARY_HOUR`, `INTERVAL`) pairs on one `CALL_DATE`: "
        f"{interval['max_pairs_per_day']}, reached on {interval['days_at_max']} of "
        f"{interval['n_days']} days.\n"
    )
    if interval["null_call_date_count"]:
        lines.append(
            f"**{interval['null_call_date_count']} rows have a null or unparseable "
            f"`CALL_DATE`** and are excluded from `n_days`, `max_pairs_per_day`, and "
            f"the DST row counts below.\n"
        )

    lines.append("**Is `MILITARY_HOUR` UTC.**\n")
    lines.append("| Changeover | Day before | Changeover day | Day after |")
    lines.append("|---|---:|---:|---:|")
    for changeover, counts in dst.items():
        lines.append(
            f"| {changeover} | {counts['day_before']} | {counts['row_count']} | "
            f"{counts['day_after']} |"
        )
    lines.append("")

    lines.append("**Do the talk-time fields agree.**\n")
    lines.append("| Field | Min | Max | Nulls | Zeros |")
    lines.append("|---|---:|---:|---:|---:|")
    for field in TALK_TIME_FIELDS:
        stats = talk_time[field]
        lines.append(
            f"| {field} | {stats['min']} | {stats['max']} | {stats['null_count']} | "
            f"{stats['zero_count']} |"
        )
    lines.append(
        f"\n`AVERAGE_TALK_TIME` differs from `TOTAL_TALK_TIME / HANDLED` by more than a "
        f"second on {talk_time['avg_talk_time_mismatch_count']} of "
        f"{talk_time['avg_talk_time_checked_count']} rows with `HANDLED` > 0.\n"
    )

    return "\n".join(lines)


def pull(raw_dir: Path = Path("data/raw")) -> Path:
    """Fetch, verify against service_count, write both files, append the audit.

    Creates raw_dir if absent, writes call_volumes_<today>.parquet and
    pull_metadata.json, and appends a dated section to data_audit.md. That
    file is only ever appended to, so it gains a section per successful call.

    Returns the parquet path. Raises FileExistsError if a snapshot for today
    exists or a pull for today is already in progress, and ValueError if the
    row count disagrees with service_count or falls under MIN_ROWS. Network
    and decoding errors from the request are propagated, and nothing is
    written when one is raised.
    """
    raw_dir = Path(raw_dir)
    today = datetime.now(timezone.utc).date().isoformat()
    parquet_path = raw_dir / f"call_volumes_{today}.parquet"
    metadata_path = raw_dir / "pull_metadata.json"
    audit_path = raw_dir / "data_audit.md"
    lock_path = raw_dir / f".call_volumes_{today}.lock"

    if parquet_path.exists():
        raise FileExistsError(f"snapshot for {today} already exists at {parquet_path}")

    # Reserve the snapshot slot atomically before the network fetch, so two
    # concurrent pulls for the same day can't both pass the exists() check
    # above and race to write parquet_path.
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise FileExistsError(f"pull for {today} is already in progress ({lock_path})")

    try:
        with httpx.Client(timeout=30.0) as client:
            reported_count = service_count(client)
            rows: list[dict] = []
            for page in fetch_pages(client):
                rows.extend(page)

        df = pd.DataFrame(rows)

        if len(df) != reported_count:
            raise ValueError(f"pulled {len(df)} rows but service reports {reported_count}")
        if len(df) < MIN_ROWS:
            raise ValueError(f"pulled {len(df)} rows, fewer than the {MIN_ROWS} floor")

        df.to_parquet(parquet_path, index=False)

        metadata = {
            "service_url": SERVICE_URL,
            "dataset_url": DATASET_URL,
            "license": "Halifax Regional Municipality Open Data Licence",
            "license_url": LICENSE_URL,
            "query_params": QUERY_PARAMS,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "snapshot_file": parquet_path.name,
            "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "service_reported_count": reported_count,
            "rows_written": len(df),
            "field_names": df.columns.tolist(),
            "object_id_min": int(df["ObjectId"].min()),
            "object_id_max": int(df["ObjectId"].max()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        findings = audit_findings(df)
        with audit_path.open("a") as f:
            f.write(_render_audit_section(findings, today))
    finally:
        lock_path.unlink(missing_ok=True)

    return parquet_path


if __name__ == "__main__":
    print(pull())
