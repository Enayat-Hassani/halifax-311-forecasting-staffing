# Data audit

This file records what is in the data snapshot and which problems in the source affect
cleaning. Figures come from `call_volumes_2026-09-13.parquet`, in Halifax local time, after
the repair in `src/clean.py`, unless stated otherwise.

In short:

- From 17 July 2024, every timestamp is four hours off (a fixed UTC+4 offset).
- From 23 May 2023 to 16 July 2024, most days are counted twice.
- `OFFERED` also counts calls that ended in the phone menu. Agent demand is
  `HANDLED + ABANDONED`.
- Talk time is not full handle time. Handle time is between 237 s and 353 s.

Service:
`https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/311_Call_Volumes/FeatureServer/0`

Fields: `CALL_DATE`, `MILITARY_HOUR`, `INTERVAL`, `OFFERED`, `HANDLED`, `ABANDONED`,
`PROCESSED_IN_IVR`, `TOTAL_TALK_TIME`, `AVERAGE_TALK_TIME`, `ObjectId`.

148,269 rows, 2017-01-01 to 2026-09-11 local. 4,057,797 calls offered as published,
3,776,255 after the duplicate repair.

## Grain and absent intervals

`MILITARY_HOUR` and `INTERVAL` agree on every row, so the grain is 30 minutes across a
48-slot day. `CALL_DATE` is midnight US Pacific in epoch milliseconds.

Half-hours with no calls are usually left out of the table. They are missing mostly in
quiet hours, and the table also has 477 rows with an explicit zero, so missing half-hours
are read as zero. Inside
opening hours in the test window, 0.01% of intervals are absent.

But a missing row does not always mean no calls. Since 2017-07-03, three or more open
half-hours in a row are missing on two days:

| Date | Absent | Reading |
|---|---|---|
| 2019-05-26 | Whole day, 17 intervals | Outage |
| 2024-02-06 | 08:30–11:30, 7 intervals | Outage. Demand returns at 12:00 with 6 calls, then at the usual level. A normal Tuesday morning has 32–45 calls per interval. |

Both are listed as `outage` in `data/special_days.csv`, and are replaced in the training history.
Neither is in the primary or test window.

Since 2020, Christmas Eve has no agent calls after about 14:00. These half-hours are
real zeros, not missing data. The opening-hours rule does not include this early close, so
the test scores 11 of them on 2025-12-24, from 14:30. They add about 0.1 percentage
points of WAPE to every run and do not change the ranking.

## Time-base break, 2024-07-17

From the row labelled 2024-07-17 00:00, every timestamp is in fixed UTC+4: 7 hours ahead
of Halifax in summer and 8 in winter. The date moves with the hour. So after this date,
Saturday 00:00–02:30 holds Friday-evening calls, and Sunday and Monday at that time hold
almost none.

When the timestamps are read as UTC+4 and converted to Halifax time, the weekday profile
matches the pre-2023 profile with a correlation of 0.9997. Shifting by 30 minutes either
way gives only 0.96–0.97. Shifting the hour while keeping the date does not fit.

HRM documents the migration from Telus Elements to an Enghouse-hosted cloud service (CAO
contract amendment, 2024-04-05) and the Enghouse Cloud launch in July 2024. HRM has
published nothing about the timestamp problem, so its cause is a guess.

## Duplicate counts, 2023-05-23 → 2024-07-16

On 395 of the 407 days with calls in this period, every handled and abandoned call is
counted twice. Talk time is doubled too, and `OFFERED` includes the extra copy.
`PROCESSED_IN_IVR` is not doubled.

HRM's 311 Call Details service holds one row per call:
`https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/arcgis/rest/services/311_Call_Details/FeatureServer/0`

- Inside the period, Call Details itself repeats each call: handled calls about four times
  and abandoned calls twice, with the same ID, times and wrap-up code.
- On a doubled day, every count in this table is even apart from a stray interval. The
  12 days counted once have about half their counts odd. No day outside the period is
  doubled. 2023-05-23 is the first open day after a closure; 2024-07-16 is the last day
  before the Enghouse cutover.

`src/clean.py` uses this even-number rule: inside the period, a day is halved when fewer
than one in five of its non-zero handled and abandoned counts are odd. It halves handled, abandoned and talk
time, and removes the second copy from `OFFERED`. This takes out 281,543 agent calls. A
stray odd interval on a doubled day becomes a half call. The snapshot is not changed.

`src/duplicates.py` checks the rule day by day against distinct `CALL_ID`s in Call Details,
from a week before the period to a week after it, and writes the daily counts to
`results/duplicate_check.csv`:

| Days | Published / distinct IDs | Repaired / distinct IDs |
|---|---:|---:|
| 395 halved, inside the period | 2.000–2.026 | 1.000–1.013 |
| 12 not halved, inside the period | 1.000 | 1.000 |
| 6 open days before the period | 1.000 | 1.000 |

The rule and Call Details agree on which days are doubled. Repaired counts equal the distinct IDs on 379 of the 395 halved days, and are within 1.3% on the rest. In the week after the period, totals agree
within 4%. Single days there differ by up to 18%, probably because the two services place
the post-cutover time base differently.

| Period | Dates | Weekday agent demand | IVR share | Abandoned / agent demand |
|---|---|---:|---:|---:|
| Telus, before | → 2023-05-22 | 828 | 25–34% | 8.3% |
| Telus, repaired | 2023-05-23 → 2024-07-16 | 885 | 26–41% | 7.4% |
| Enghouse | 2024-07-17 → | 899 | 24–37% | 4.8% |

Weekday means per day, holidays excluded, over 2022-05 → 2023-04, 2023-06-12 → 2024-07-16
and 2024-07-22 → 2025-07-21. IVR share is the monthly range.

After the repair, call volume has no jump. Abandonment still drops at the 2024-07-17
platform change. This may be a change in how abandoned calls are counted.

HRM publishes 374,000 calls for FY2023/24. After the repair, `OFFERED` for that year is
390,698, within 5%; FY2022/23 agrees within 3%.

## Opening hours

Agent demand follows HRM's published 311 hours to the half-hour: weekdays 08:00–20:00,
weekends 09:00–17:30. Until 2017-07-03 the line was staffed 07:00–23:00. Weekend closing
was later in 2018–2020 and has been 17:30 since 2021.

From 2023-05-22 the line takes no agent calls on statutory holidays. Before that, only
25 December and 1 January were closed (from 2020-12-25), and other holidays were open with
low volume. Practice is not consistent: Easter Monday was closed in 2026 and open in 2024
and 2025. `data/special_days.csv` lists each closed day.

## The line runs 24 hours; agents do not

`OFFERED` counts every call to the line, and is 47–49% higher than `HANDLED + ABANDONED`
in training and test. `HANDLED + ABANDONED` counts the calls that reached or waited for an
agent, which is what staffing needs.

The three outcome fields do not add up to `OFFERED`. On 10,419 rows
`HANDLED + ABANDONED + PROCESSED_IN_IVR` is more than `OFFERED`, so a call can go through
the phone menu and still reach an agent. A further 146,004 calls, 3.9% of offered, fall into none of the three.
So `OFFERED − PROCESSED_IN_IVR` is not a safe way to get agent demand.

## Talk time and abandonment

Mean talk time is total `TOTAL_TALK_TIME` divided by total `HANDLED`. It changes over time: 182 s in 2017,
288 s in 2023, 237 s in primary training (2024-07-17 → 2025-08-31) and 248 s in the test
window. Averaging the `AVERAGE_TALK_TIME` column gives a lower figure, because the column is
0 on the 55,247 rows where `HANDLED` is 0.

This is talk time, not handle time. HRM's metadata for this table defines it as "the total
time in seconds the customer spoke with an Agent" (ODMT_311_Call_Volumes, 2018-10-31).

The per-call 311 Call Details table also has `DURATION_IN_SECONDS`, defined as the time "from
when the call is initiated, time in the queue, talking with an agent, and wrap up"
(ODMT_311_Call_Details, 2023-11-27). Handled phone calls in primary training (263,428, email
queue excluded):

| Measure | Mean | Reading |
|---|---:|---|
| Talk time | 239 s | Lower bound on handle time |
| Duration | 353 s | Upper bound: adds queue time and wrap-up to talk |

Handle time is between the two. The table does not split queue time from wrap-up. On a
sample week (2025-03-03), the median of duration minus talk is 72 s in every hour, busy or
quiet, so most of the gap does not grow with congestion. The 237 s from this table matches
the per-call talk time. The "service 360 s" scenario covers the upper bound.

Abandonment is 4.9% of agent demand in primary training and 5.8% in the test window. It
depends on both caller patience and queue length. The table has no waiting times or
staffing levels, so patience cannot be measured from it.

## Named days

Hurricane Dorian (2019-09-07/08), Hurricane Fiona (2022-09-24/25), the Tantallon wildfire
(2023-05-29/30, 2,579 agent calls on the first day), the July 2023 flood (2023-07-22), the
February 2024 snowstorms (2024-02-03/04 and 2024-02-14), and Canada Day evenings in
2017–2019, when calls offered spike after 21:00. No 2020 shock shows in agent demand.

## Open questions

What `OFFERED` counts under HRM's configuration. Vendor conventions differ: Genesys
documents flow-outs that break offered = answered + abandoned, Cisco can exclude short
abandons, Avaya treats offered as the denominator for both outcomes.

What the 3.9% residual between `OFFERED` and the three outcome fields represents.

Why Telus wrote duplicate counts, and why 12 days escaped them.
