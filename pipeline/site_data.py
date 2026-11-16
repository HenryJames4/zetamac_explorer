"""Data for the single-page site (web/index.html) -> web/data/*.js.

Per problem on the stock grids, from clean sessions (build.sessions_table) and
trials with 100 ms < RT <= 90 s: trial count, median RT, first-try rate
(keystrokes == answer digits, trials without an entry log excluded) and a
32 x 250 ms RT histogram plus an overflow bin.

Two replay sessions: clean, 120 s, all four operators, stock ranges, picked at
the median and the 95th percentile of score, with their real keystroke logs.
"""

import glob
import json
import re
import sys

import numpy as np
import polars as pl

from build import sessions_table
from common import MAX_TRIAL_MS, ROOT, SHARD_GLOB, THRESHOLD_MS, WORK
from extract import SCHEMA

OUT = ROOT / "web/data"
COUNTS = WORK / "site_counts"
BIN_MS, BINS = 250, 32
GRIDS = {0: ("add", 2, 100), 1: ("sub", 2, 100), 2: ("mul", 2, 12), 3: ("div", 2, 12)}  # rows; cols are 2..100
COLS = (2, 100)


def ms_counts(paths: list[str], clean: pl.DataFrame) -> pl.DataFrame:
    """Trial counts per (problem, ms) for a batch of trial files."""
    left, right, op = pl.col("left"), pl.col("right"), pl.col("op")
    # Grid coordinates: sub is (r+c) - r and div is (r*c) / r.
    r = pl.when(op.is_in([0, 2])).then(left).otherwise(right)
    c = (pl.when(op.is_in([0, 2])).then(right)
         .when(op == 1).then(left - right)
         .otherwise(left // pl.when(right == 0).then(1).otherwise(right)))
    exact_div = (op != 3) | ((right != 0) & (left % pl.when(right == 0).then(1).otherwise(right) == 0))
    ans = pl.when(op == 0).then(left + right).when(op == 1).then(left - right) \
        .when(op == 2).then(left * right).otherwise(c)
    first = pl.col("n_keys") == ans.cast(pl.Utf8).str.len_chars()
    return (
        pl.scan_parquet(paths)
        .join(clean.lazy(), on="sid", how="semi")
        .filter((pl.col("ms") > THRESHOLD_MS) & (pl.col("ms") <= MAX_TRIAL_MS) & exact_div)
        .with_columns(r.cast(pl.Int32).alias("r"), c.cast(pl.Int32).alias("c"))
        .filter(pl.col("r").is_between(2, 100) & pl.col("c").is_between(*COLS))
        .group_by("op", "r", "c", "ms")
        .agg(pl.len().cast(pl.UInt32).alias("n"),
             (pl.col("n_keys") >= 0).sum().cast(pl.UInt32).alias("keyed"),
             first.sum().cast(pl.UInt32).alias("first"))
        .collect()
    )


def write_counts(sessions: pl.DataFrame) -> None:
    # Batches are cached on disk so a rerun resumes where it stopped.
    COUNTS.mkdir(exist_ok=True)
    clean = sessions.filter("clean").select("sid")
    paths = sorted(glob.glob(str(WORK / "trials/*.parquet")))
    for i in range(0, len(paths), 40):
        out = COUNTS / f"{i:04d}.parquet"
        if not out.exists():
            ms_counts(paths[i:i + 40], clean).write_parquet(out)
        print(f"trial files {min(i + 40, len(paths))}/{len(paths)}", flush=True)


def problem_table(op: int) -> pl.DataFrame:
    m = (pl.scan_parquet(str(COUNTS / "*.parquet")).filter(pl.col("op") == op)
         .group_by("op", "r", "c", "ms").agg(pl.col("n", "keyed", "first").sum())
         .sort("op", "r", "c", "ms").collect())
    # Exact median from per-ms counts: lower and upper middle values averaged.
    cum = pl.col("n").cum_sum().over("op", "r", "c")
    tot = pl.col("n").sum().over("op", "r", "c")
    m = m.with_columns(cum.alias("cum"), tot.alias("tot"), (pl.col("ms") // BIN_MS).clip(0, BINS).alias("bin"))
    lo_mid = pl.col("ms").filter(pl.col("cum") >= (pl.col("tot") + 1) // 2).first()
    hi_mid = pl.col("ms").filter(pl.col("cum") >= pl.col("tot") // 2 + 1).first()
    stats = m.group_by("op", "r", "c").agg(
        pl.col("n").sum(), ((lo_mid + hi_mid) / 2).alias("med"),
        (pl.col("first").sum() / pl.col("keyed").sum()).alias("first"),
    )
    hist = (m.group_by("op", "r", "c", "bin").agg(pl.col("n").sum())
            .pivot("bin", index=["op", "r", "c"], values="n").fill_null(0))
    hist = hist.select("op", "r", "c", *[pl.col(str(b)).alias(f"b{b}") if str(b) in hist.columns
                                          else pl.lit(0).alias(f"b{b}") for b in range(BINS + 1)])
    return stats.join(hist, on=["op", "r", "c"])


def export_ops() -> None:
    for op, (name, lo, hi) in GRIDS.items():
        d = problem_table(op).filter(pl.col("r").is_between(lo, hi) & pl.col("c").is_between(*COLS))
        grid = {(row["r"], row["c"]): row for row in d.iter_rows(named=True)}
        cells = []
        for rr in range(lo, hi + 1):
            for cc in range(COLS[0], COLS[1] + 1):
                p = grid.get((rr, cc))
                cells.append(None if p is None else
                             [round(p["med"]), p["n"], round(100 * p["first"]), [p[f"b{b}"] for b in range(BINS + 1)]])
        missing = sum(x is None for x in cells)
        print(f"{name}: {len(cells) - missing} problems, {missing} missing, {d['n'].sum():,} trials", flush=True)
        js = json.dumps({"rows": [lo, hi], "cols": list(COLS), "cells": cells}, separators=(",", ":"))
        (OUT / f"{name}.js").write_text(f"ZDATA.{name}={js};\n")


def pick_sessions(sessions: pl.DataFrame) -> list[tuple[str, int]]:
    s = pl.read_parquet(str(WORK / "sessions/*.parquet")).join(sessions.select("sid", "clean"), on="sid")
    pool = s.filter(
        pl.col("clean") & (pl.col("duration") == 120) & pl.col("score").is_not_null()
        & pl.all_horizontal([pl.col(f).fill_null(False) for f in ("add", "sub", "mul", "div")])
    )
    rng = np.random.default_rng(0)
    out = []
    for label, q in (("Fast player", 0.95), ("Typical player", 0.5)):
        score = int(pool["score"].quantile(q, "nearest"))
        sids = pool.filter(pl.col("score") == score)["sid"].to_list()
        out.append((label, int(rng.choice(sids))))
        print(f"{label}: q={q} score={score} from {len(sids):,} sessions")
    return out


def encode_keys(entry: list[str]) -> str:
    # One char per key press: digit typed or "<" for backspace, from successive input states.
    out, prev = [], ""
    for s in entry:
        p = 0
        while p < min(len(prev), len(s)) and prev[p] == s[p]:
            p += 1
        out += ["<"] * (len(prev) - p) + list(s[p:])
        prev = s
    return "".join(out)


def export_sessions(picks: list[tuple[str, int]]) -> None:
    paths = sorted(glob.glob(SHARD_GLOB))
    sym = {"+": 0, "-": 1, "*": 2, "/": 3}
    res = []
    for label, sid in picks:
        raw = pl.read_ndjson(paths[sid // 1_000_000], schema=SCHEMA)
        row = raw.row(sid % 1_000_000, named=True)
        trials = []
        for i, t in enumerate(row["problem_log"]):
            a, o, b = t["problem"].split()
            a, b, op = int(a), int(b), sym[o]
            R, C = (a, b) if op in (0, 2) else (b, a - b if op == 1 else a // b)
            last = i == len(row["problem_log"]) - 1
            trials.append([op, R, C, None if last else round(t["timeMs"]), encode_keys(t["entry"] or [])])
        done = sum(x[3] for x in trials[:-1])
        print(f"{label}: sid={sid} score={row['score']} trials={len(trials) - 1} sum_rt={done} ms")
        res.append({"label": label, "score": int(row["score"]), "trials": trials})
    js = json.dumps(res, separators=(",", ":"))
    (OUT / "sessions.js").write_text(f"ZDATA.sessions={js};\n")


def export_scale(n_shards: int = 60, per_shard: int = 250, n_games: int = 10_000) -> None:
    """Random games from random shards, any settings, as 'rt:problem:keys' per trial.

    rt is in 100 ms units; keys are the typed characters with '<' for backspace.
    """
    total = (pl.scan_parquet(str(WORK / "sessions/*.parquet"))
             .select((pl.col("n_trials") > 0).sum()).collect().item())
    paths = sorted(glob.glob(SHARD_GLOB))
    rng = np.random.default_rng(1)
    games = []
    for k, i in enumerate(sorted(rng.choice(len(paths), n_shards, replace=False))):
        raw = pl.read_ndjson(paths[i], schema={"problem_log": pl.List(pl.Struct(
            {"problem": pl.String, "timeMs": pl.Float64, "entry": pl.List(pl.String)}))})
        rows = raw["problem_log"].to_list()
        picked = 0
        for j in rng.permutation(len(rows)):
            trials = []
            for t in (rows[j] or [])[:-1]:   # last trial is the unfinished one at time-out
                ms, prob = t["timeMs"], re.fullmatch(r"\s*(\d{1,4})\s*([-+*/])\s*(\d{1,4})\s*", t["problem"] or "")
                if prob and 0 in (int(prob[1]), int(prob[3])):   # custom ranges with 0 (x0, +0): skip the whole game
                    trials = []
                    break
                if ms is None or not THRESHOLD_MS < ms <= MAX_TRIAL_MS or prob is None or not t["entry"]:
                    continue
                keys = encode_keys(t["entry"])
                if not re.fullmatch(r"[0-9<]+", keys):
                    continue
                trials.append(f"{max(1, round(ms / 100))}:{''.join(prob.groups())}:{keys}")
            if trials:
                games.append(",".join(trials))
                picked += 1
                if picked == per_shard:
                    break
        print(f"scale shard {k + 1}/{n_shards}", flush=True)
    games = [games[i] for i in rng.permutation(len(games))[:n_games]]
    js = json.dumps({"total": total, "games": games}, separators=(",", ":"))
    (OUT / "scale.js").write_text(f"ZDATA.scale={js};\n")
    print(f"scale: {len(games):,} games, {total:,} sessions with trials, {len(js) / 1e6:.1f} MB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--scale" in sys.argv:
        export_scale()
        sys.exit()
    sessions, diag = sessions_table()
    print(diag, flush=True)
    export_sessions(pick_sessions(sessions))
    write_counts(sessions)
    export_ops()
