"""Check the duplicate repair day by day against HRM's per-call records.

311 Call Details holds one row per call record. For each day from a week before the
duplicated period to a week after it, the check counts the handled and abandoned agent
records and their distinct CALL_IDs, and sets them beside the published agent calls and the
repaired series. Writes `results/duplicate_check.csv`.
"""

from pathlib import Path

import httpx
import pandas as pd

from src.clean import DUPLICATED, local_time
from src.ingest import _parse_esri_response

DETAILS_URL = (
    "https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/"
    "311_Call_Details/FeatureServer/0/query"
)
AGENT_CALLS = "QUEUE_NAME <> 'EMail' AND (OUTCOME LIKE 'Handled%' OR OUTCOME LIKE 'Abandoned%')"


def _count(client: httpx.Client, day: pd.Timestamp, distinct: bool) -> int:
    """Agent records on `day`, or their distinct CALL_IDs."""
    end = day + pd.Timedelta(days=1)
    params = {
        "where": f"ARRIVAL_DATETIME >= TIMESTAMP '{day:%Y-%m-%d} 00:00:00' AND "
        f"ARRIVAL_DATETIME < TIMESTAMP '{end:%Y-%m-%d} 00:00:00' AND {AGENT_CALLS}",
        "returnCountOnly": "true",
        "f": "json",
    }
    if distinct:
        params |= {"returnDistinctValues": "true", "outFields": "CALL_ID"}
    response = client.get(DETAILS_URL, params=params)
    response.raise_for_status()
    return _parse_esri_response(response)["count"]


def main(
    raw_dir: Path = Path("data/raw"),
    series_path: Path = Path("data/processed/series.parquet"),
    out: Path = Path("results/duplicate_check.csv"),
) -> pd.DataFrame:
    raw = pd.read_parquet(sorted(raw_dir.glob("call_volumes_*.parquet"))[-1])
    published = (raw["HANDLED"] + raw["ABANDONED"]).groupby(local_time(raw).dt.normalize()).sum()
    repaired = pd.read_parquet(series_path)["demand"].resample("D").sum()

    days = pd.date_range(DUPLICATED[0] - pd.Timedelta(days=7), DUPLICATED[1] + pd.Timedelta(days=7))
    with httpx.Client(timeout=60.0) as client:
        counts = [(_count(client, d, False), _count(client, d, True)) for d in days]
    check = pd.DataFrame(counts, index=days.rename("date"), columns=["records", "call_ids"])
    check["published"] = published.reindex(days, fill_value=0).to_numpy()
    check["repaired"] = repaired.reindex(days).to_numpy()
    check["halved"] = check["repaired"] < check["published"]
    out.parent.mkdir(parents=True, exist_ok=True)
    check.to_csv(out)
    return check


if __name__ == "__main__":
    main()
