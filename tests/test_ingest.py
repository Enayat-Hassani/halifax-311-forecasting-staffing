"""Ingest tests. None of these hit the network."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingest import audit_findings, fetch_pages, pull

RAW_DIR = Path("data/raw")

METADATA_KEYS = {
    "service_url",
    "dataset_url",
    "license",
    "license_url",
    "query_params",
    "pulled_at_utc",
    "snapshot_file",
    "sha256",
    "service_reported_count",
    "rows_written",
    "field_names",
    "object_id_min",
    "object_id_max",
}


def _snapshot_path() -> Path:
    matches = sorted(RAW_DIR.glob("call_volumes_*.parquet"))
    if not matches:
        pytest.fail("no frozen snapshot in data/raw; run src.ingest.pull() first")
    return matches[-1]


@pytest.fixture(scope="module")
def snapshot_df() -> pd.DataFrame:
    return pd.read_parquet(_snapshot_path())


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((RAW_DIR / "pull_metadata.json").read_text())


def test_rows_written_matches_service_reported_count(metadata):
    assert metadata["rows_written"] == metadata["service_reported_count"]


def test_rows_written_matches_snapshot_row_count(snapshot_df, metadata):
    assert len(snapshot_df) == metadata["rows_written"]


def test_snapshot_matches_recorded_hash(metadata):
    snapshot = RAW_DIR / metadata["snapshot_file"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == metadata["sha256"]


def test_no_duplicate_object_id(snapshot_df):
    assert snapshot_df["ObjectId"].duplicated().sum() == 0


def test_no_duplicate_call_date_hour_interval(snapshot_df):
    subset = ["CALL_DATE", "MILITARY_HOUR", "INTERVAL"]
    assert snapshot_df.duplicated(subset=subset).sum() == 0


@pytest.mark.parametrize("column", ["OFFERED", "HANDLED", "ABANDONED"])
def test_volume_columns_non_negative_and_non_null(snapshot_df, column):
    assert snapshot_df[column].isna().sum() == 0
    assert (snapshot_df[column] < 0).sum() == 0


def test_metadata_carries_every_documented_key(metadata):
    assert METADATA_KEYS <= metadata.keys()


def test_pull_raises_when_snapshot_for_today_exists(tmp_path, monkeypatch):
    import src.ingest as ingest_module
    from datetime import datetime, timezone

    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(ingest_module, "datetime", FrozenDateTime)
    existing = tmp_path / "call_volumes_2026-01-01.parquet"
    existing.write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        pull(raw_dir=tmp_path)


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _StubClient:
    """Returns two pages: the first sets exceededTransferLimit, the second does not."""

    def __init__(self):
        self.requests = []

    def get(self, url, params=None):
        self.requests.append(dict(params))
        offset = params["resultOffset"]
        if offset == 0:
            return _StubResponse(
                {
                    "exceededTransferLimit": True,
                    "features": [
                        {"attributes": {"ObjectId": 1}},
                        {"attributes": {"ObjectId": 2}},
                    ],
                }
            )
        return _StubResponse(
            {
                "exceededTransferLimit": False,
                "features": [{"attributes": {"ObjectId": 3}}],
            }
        )


def test_fetch_pages_stops_when_transfer_limit_not_exceeded():
    client = _StubClient()
    pages = list(fetch_pages(client, page_size=2))

    assert len(pages) == 2
    assert pages[0] == [{"ObjectId": 1}, {"ObjectId": 2}]
    assert pages[1] == [{"ObjectId": 3}]
    assert [r["resultOffset"] for r in client.requests] == [0, 2]


class _ErrorPayloadClient:
    """Returns an HTTP 200 whose body is an ArcGIS error payload."""

    def get(self, url, params=None):
        return _StubResponse({"error": {"code": 400, "message": "bad query"}})


def test_fetch_pages_raises_on_esri_error_payload():
    client = _ErrorPayloadClient()

    with pytest.raises(RuntimeError, match="bad query"):
        list(fetch_pages(client, page_size=2))


class _StuckClient:
    """Always reports exceededTransferLimit with zero features."""

    def get(self, url, params=None):
        return _StubResponse({"exceededTransferLimit": True, "features": []})


def test_fetch_pages_raises_instead_of_looping_forever_on_zero_progress():
    client = _StuckClient()

    with pytest.raises(RuntimeError, match="infinite loop"):
        list(fetch_pages(client, page_size=2))


def _ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").value / 1_000_000)


def test_audit_findings_on_known_fixture():
    # day_A carries two distinct (MILITARY_HOUR, INTERVAL) pairs across three
    # rows, one of them a same-day duplicate pair that drop_duplicates must
    # absorb. day_B carries one pair, so day_A alone reaches the per-day max.
    day_a, day_b = _ms("2019-01-01"), _ms("2019-01-02")
    rows = [
        {
            "CALL_DATE": day_a,
            "MILITARY_HOUR": 9,
            "INTERVAL": "09:00 AM - 09:30 AM",
            "HANDLED": 5,
            "PROCESSED_IN_IVR": 0,
            "TOTAL_TALK_TIME": 500,
            "AVERAGE_TALK_TIME": 100,  # 500/5 = 100, matches
        },
        {
            "CALL_DATE": day_a,
            "MILITARY_HOUR": 9,
            "INTERVAL": "09:00 AM - 09:30 AM",  # duplicate pair, same day
            "HANDLED": 5,
            "PROCESSED_IN_IVR": 0,
            "TOTAL_TALK_TIME": 500,
            "AVERAGE_TALK_TIME": 100,
        },
        {
            "CALL_DATE": day_a,
            "MILITARY_HOUR": 9,
            "INTERVAL": "09:30 AM - 10:00 AM",
            "HANDLED": 0,  # excluded from the AVERAGE_TALK_TIME check
            "PROCESSED_IN_IVR": 2,
            "TOTAL_TALK_TIME": 0,
            "AVERAGE_TALK_TIME": 0,
        },
        {
            "CALL_DATE": day_b,
            "MILITARY_HOUR": 10,
            "INTERVAL": "10:00 AM - 10:30 AM",
            "HANDLED": 10,
            "PROCESSED_IN_IVR": 1,
            "TOTAL_TALK_TIME": 1000,
            "AVERAGE_TALK_TIME": 95,  # 1000/10 = 100, off by 5: the one mismatch
        },
    ]

    # DST fixture rows: one row on each changeover date and its neighbours,
    # all with identical HANDLED/talk-time values so they never register as
    # a mismatch. 2026-03-08 gets two rows sharing the same pair, so its
    # distinct-pair count is still 1 despite a row_count of 2.
    for date_str in [
        "2026-03-07",
        "2026-03-08",
        "2026-03-08",
        "2026-03-09",
        "2025-11-01",
        "2025-11-02",
        "2025-11-03",
    ]:
        rows.append(
            {
                "CALL_DATE": _ms(date_str),
                "MILITARY_HOUR": 0,
                "INTERVAL": "12:00 AM - 12:30 AM",
                "HANDLED": 1,
                "PROCESSED_IN_IVR": 0,
                "TOTAL_TALK_TIME": 60,
                "AVERAGE_TALK_TIME": 60,
            }
        )

    df = pd.DataFrame(rows)
    findings = audit_findings(df)

    assert findings["interval"]["n_distinct_values"] == 4
    assert findings["interval"]["max_pairs_per_day"] == 2
    assert findings["interval"]["days_at_max"] == 1
    assert findings["interval"]["n_days"] == 8

    assert findings["dst"]["2026-03-08"] == {
        "row_count": 2,
        "day_before": 1,
        "day_after": 1,
    }
    assert findings["dst"]["2025-11-02"] == {
        "row_count": 1,
        "day_before": 1,
        "day_after": 1,
    }

    assert findings["talk_time"]["HANDLED"] == {
        "min": 0,
        "max": 10,
        "null_count": 0,
        "zero_count": 1,
    }
    assert findings["talk_time"]["PROCESSED_IN_IVR"]["zero_count"] == 9
    assert findings["talk_time"]["TOTAL_TALK_TIME"]["zero_count"] == 1
    assert findings["talk_time"]["avg_talk_time_checked_count"] == 10
    assert findings["talk_time"]["avg_talk_time_mismatch_count"] == 1
    assert findings["interval"]["null_call_date_count"] == 0


def test_audit_findings_counts_null_call_date_rows_instead_of_dropping_silently():
    rows = [
        {
            "CALL_DATE": _ms("2019-01-01"),
            "MILITARY_HOUR": 9,
            "INTERVAL": "09:00 AM - 09:30 AM",
            "HANDLED": 1,
            "PROCESSED_IN_IVR": 0,
            "TOTAL_TALK_TIME": 60,
            "AVERAGE_TALK_TIME": 60,
        },
        {
            "CALL_DATE": None,
            "MILITARY_HOUR": 9,
            "INTERVAL": "09:00 AM - 09:30 AM",
            "HANDLED": 1,
            "PROCESSED_IN_IVR": 0,
            "TOTAL_TALK_TIME": 60,
            "AVERAGE_TALK_TIME": 60,
        },
    ]
    df = pd.DataFrame(rows)
    findings = audit_findings(df)

    assert findings["interval"]["null_call_date_count"] == 1
    assert findings["interval"]["n_days"] == 1
