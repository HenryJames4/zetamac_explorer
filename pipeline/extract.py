"""Shards -> per-shard trial and session parquet files in work/.

Per session: drop the forced last trial, drop trials with a
missing or negative timeMs, drop trials that are not "a op b" with operands
of at most 15 digits. Everything else is decided in build.py.
"""

import glob
import sys
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from common import OPS, SHARD_GLOB, THRESHOLD_MS, WORK

RANGE_COLS = [
    f"{f}_{s}_{b}" for f in ("add", "mul") for s in ("left", "right") for b in ("min", "max")
]

SCHEMA = {
    "completed_at": pl.String,
    "game_options": pl.Struct(
        {"add": pl.Boolean, "sub": pl.Boolean, "mul": pl.Boolean, "div": pl.Boolean,
         "duration": pl.Float64, **{c: pl.Float64 for c in RANGE_COLS}}  # f64: overflowing junk gets nulled below
    ),
    "problem_log": pl.List(
        pl.Struct({"problem": pl.String, "timeMs": pl.Float64, "entry": pl.List(pl.String)})
    ),
    "score": pl.String,
    "rotating_ip_hash": pl.String,
}

PROBLEM_RE = r"^\s*(?P<left>\d{1,15})\s*(?P<op>[+\-*/])\s*(?P<right>\d{1,15})\s*$"


def extract(index: int, path: str) -> tuple[int, int, int]:
    raw = pl.read_ndjson(path, schema=SCHEMA)
    n = pl.col("problem_log").list.len()
    raw = raw.with_row_index("row").with_columns(
        (pl.lit(index, dtype=pl.UInt32) * 1_000_000 + pl.col("row")).alias("sid"),
        pl.when(n > 1).then(pl.col("problem_log").list.slice(0, n - 1))
        .otherwise(pl.col("problem_log").list.slice(0, 0)).alias("problem_log"),
    )

    trials = (
        raw.select("sid", "problem_log")
        .explode("problem_log", empty_as_null=False)
        .unnest("problem_log")
        .filter(pl.col("timeMs").is_not_null() & (pl.col("timeMs") >= 0))
        .with_columns(pl.col("problem").str.extract_groups(PROBLEM_RE).alias("g"))
        .unnest("g")
        .filter(pl.col("left").is_not_null())
        .select(
            "sid",
            pl.col("left").cast(pl.Int64),
            pl.col("op").replace_strict(OPS, list(range(4)), return_dtype=pl.UInt8),
            pl.col("right").cast(pl.Int64),
            pl.col("timeMs").round().cast(pl.Int32).alias("ms"),
            pl.col("entry").list.len().cast(pl.Int16).fill_null(-1).alias("n_keys"),
        )
    )

    per_session = trials.group_by("sid").agg(
        pl.len().cast(pl.Int32).alias("n_trials"),
        pl.min_horizontal(pl.col("left").min(), pl.col("right").min()).alias("min_operand"),
        (pl.col("ms") <= THRESHOLD_MS).any().alias("has_fast"),
    )
    sessions = (
        raw.select("sid", "rotating_ip_hash", "score", "game_options")
        .unnest("game_options")
        .with_columns(
            pl.col("rotating_ip_hash").is_not_null().alias("has_ip"),
            pl.col("score").cast(pl.Int32, strict=False),
            pl.col("duration").fill_null(120).cast(pl.Int32, strict=False),
            *[pl.col(c).cast(pl.Int64, strict=False) for c in RANGE_COLS],
        )
        .drop("rotating_ip_hash")
        .join(per_session, on="sid", how="left")
        .with_columns(pl.col("n_trials").fill_null(0), pl.col("has_fast").fill_null(False))
    )

    trials.write_parquet(WORK / "trials" / f"{index:04d}.parquet")
    sessions.write_parquet(WORK / "sessions" / f"{index:04d}.parquet")
    return index, raw.height, trials.height


if __name__ == "__main__":
    (WORK / "trials").mkdir(parents=True, exist_ok=True)
    (WORK / "sessions").mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(SHARD_GLOB))
    if len(sys.argv) > 1:
        paths = paths[: int(sys.argv[1])]
    total_s = total_t = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, ns, nt in ex.map(extract, range(len(paths)), paths, chunksize=4):
            total_s += ns
            total_t += nt
            if i % 50 == 0:
                print(f"shard {i}/{len(paths)}  sessions={total_s:,}  trials={total_t:,}", flush=True)
    print(f"done: sessions={total_s:,} trials={total_t:,}")
