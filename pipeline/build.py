"""Skill tiers and the binned cube the explorer reads.

Every trial with 100 ms < RT <= 90 s goes into the cube. Mode 0 holds the
trials that pass the default cleaning (filter_data.py): stock ranges, IP hash
present, not a single-question drill, more than MIN_TRIALS trials, no giant
log, no trial at or under 100 ms in the session, keystroke count equal to the
answer's digit count. Mode 1 holds every other trial, so mode 0 + mode 1 is
all of the data.

Skill tier = quintile of the session's score among all sessions with identical
game settings and duration. Groups under MIN_GROUP_SESSIONS are unranked.
"""

import json
import shutil
import sys

import numpy as np
import polars as pl

from common import (
    COARSE_EDGES, DEFAULT_RANGES, FINE_EDGES, FINE_PER_COARSE, MAX_LOG_TRIALS,
    MAX_TRIAL_MS, MIN_GROUP_SESSIONS, MIN_PROBLEM_N, MIN_TRIALS, N_FINE,
    N_MODES, N_TIERS, OPS, OUT, THRESHOLD_MS, WORK,
)

SETTINGS_COLS = ["add", "sub", "mul", "div", "duration", *DEFAULT_RANGES]
LOG_CENTRE = np.log(1500.0)  # centring keeps float32 power sums precise


def sessions_table() -> tuple[pl.DataFrame, dict]:
    s = pl.read_parquet(str(WORK / "sessions/*.parquet"))
    single = pl.lit(False)
    for f in ("add", "mul"):
        single = single | (
            pl.col(f).fill_null(False)
            & (pl.col(f"{f}_left_min") == pl.col(f"{f}_left_max"))
            & (pl.col(f"{f}_right_min") == pl.col(f"{f}_right_max"))
        )
    stock = pl.all_horizontal([pl.col(c) == v for c, v in DEFAULT_RANGES.items()]).fill_null(False)
    clean = (
        pl.col("has_ip") & ~single.fill_null(False)
        & (pl.col("n_trials") > MIN_TRIALS) & (pl.col("n_trials") <= MAX_LOG_TRIALS)
        & stock & ~pl.col("has_fast")
    )

    # Score quintile within identical settings; small groups stay unranked.
    n_group = pl.len().over(SETTINGS_COLS)
    pct = (pl.col("score").rank("average").over(SETTINGS_COLS) - 0.5) / n_group
    tier = (pct * 5).floor().clip(0, 4)
    s = s.with_columns(
        clean.alias("clean"),
        pl.when((n_group >= MIN_GROUP_SESSIONS) & pl.col("score").is_not_null())
        .then(tier).otherwise(5).cast(pl.UInt8).alias("tier"),
    )
    diag = {
        "sessions": s.height,
        "sessions_clean": int(s["clean"].sum()),
        "sessions_unranked": int((s["tier"] == 5).sum()),
    }
    return s.select("sid", "clean", "tier"), diag


def build_cube(sessions: pl.DataFrame) -> pl.DataFrame:
    ms = np.arange(THRESHOLD_MS + 1, MAX_TRIAL_MS + 1)
    bins = pl.DataFrame({
        "ms": ms.astype(np.int32),
        "fbin": (np.searchsorted(FINE_EDGES, ms, side="left") - 1).astype(np.uint8),
    })
    left, right = pl.col("left"), pl.col("right")
    answer = (
        pl.when(pl.col("op") == 0).then(left + right)
        .when(pl.col("op") == 1).then(left - right)
        .when(pl.col("op") == 2).then(left * right)
        .otherwise(left // pl.when(right == 0).then(1).otherwise(right))
    )
    keys_ok = (pl.col("n_keys") == -1) | (
        pl.col("n_keys") == answer.cast(pl.Utf8).str.len_chars()
    )
    return (
        pl.scan_parquet(str(WORK / "trials/*.parquet"))
        .join(sessions.lazy(), on="sid", how="inner")
        .join(bins.lazy(), on="ms", how="inner")  # enforces 100 < ms <= 90 s
        .with_columns(pl.when(pl.col("clean") & keys_ok).then(0).otherwise(1).cast(pl.UInt8).alias("mode"))
        .group_by("op", "left", "right", "mode", "tier", "fbin")
        .agg(pl.len().cast(pl.UInt32).alias("n"))
        .collect(engine="streaming")
    )


def export(cube: pl.DataFrame, diag: dict) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "overview").mkdir(parents=True)
    (OUT / "detail").mkdir()
    (OUT / "moments").mkdir()

    problems = (
        cube.group_by("op", "left", "right")
        .agg(
            pl.col("n").sum().alias("n"),
            pl.col("n").filter(pl.col("mode") == 0).sum().alias("n_clean"),
        )
        .filter((pl.col("n") >= MIN_PROBLEM_N) & (pl.col("left") < 2**31) & (pl.col("right") < 2**31))
        .sort("op", "left", "right")
        .with_row_index("pid")
    )
    P = problems.height
    np.stack([
        problems["op"].to_numpy(), problems["left"].to_numpy(), problems["right"].to_numpy(),
        problems["n_clean"].to_numpy(), (problems["n"] - problems["n_clean"]).to_numpy(),
    ], axis=1).astype("<i4").tofile(OUT / "problems.bin")

    # Operator-level histograms over every trial, indexed or not.
    g = np.zeros((4, N_MODES, N_TIERS, N_FINE), dtype="<u4")
    agg = cube.group_by("op", "mode", "tier", "fbin").agg(pl.col("n").sum())
    g[agg["op"], agg["mode"], agg["tier"], agg["fbin"]] = agg["n"]
    g.tofile(OUT / "global.bin")

    # Dense per-problem cube: (pid, mode, tier, fine bin).
    c = cube.join(problems.select("pid", "op", "left", "right"), on=["op", "left", "right"], how="inner")
    dense = np.zeros((P, N_MODES, N_TIERS, N_FINE), dtype="<u4")
    dense[c["pid"], c["mode"], c["tier"], c["fbin"]] = c["n"]
    for pid in range(P):
        dense[pid].tofile(OUT / "detail" / f"{pid}.bin")

    # Overview files carry the marginals too: mode 2 = both modes, tier 6 = all tiers.
    n_coarse = len(COARSE_EDGES) - 1
    full = np.concatenate([dense, dense.sum(axis=1, keepdims=True, dtype="<u4")], axis=1)
    full = np.concatenate([full, full.sum(axis=2, keepdims=True, dtype="<u4")], axis=2)
    coarse = full.reshape(P, N_MODES + 1, N_TIERS + 1, n_coarse, FINE_PER_COARSE).sum(axis=-1, dtype="<u4")
    op_col = problems["op"].to_numpy()
    for op in range(4):
        sel = coarse[op_col == op]
        for mode in range(N_MODES + 1):
            for tier in range(N_TIERS + 1):
                sel[:, mode, tier].astype("<u4").tofile(OUT / "overview" / f"{op}_{mode}_{tier}.bin")

    # Power sums of centred log-time per coarse bin, so shape statistics (skew,
    # kurtosis, bimodality) can be computed for any floor/max at fine-bin precision.
    x = np.log(np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])) - LOG_CENTRE
    xk = np.stack([x, x**2, x**3, x**4], axis=-1).reshape(n_coarse, FINE_PER_COARSE, 4)
    for op in range(4):
        sub = full[op_col == op].reshape(-1, N_MODES + 1, N_TIERS + 1, n_coarse, FINE_PER_COARSE)
        for mode in (0, 2):
            for tier in (*range(5), 6):
                m = np.einsum("pcf,cfk->pck", sub[:, mode, tier].astype(np.float64), xk)
                m.astype("<f4").tofile(OUT / "moments" / f"{op}_{mode}_{tier}.bin")

    by_mode = cube.group_by("mode").agg(pl.col("n").sum()).sort("mode")["n"].to_list()
    meta = {
        "ops": OPS,
        "coarse_edges": COARSE_EDGES,
        "fine_edges": FINE_EDGES.tolist(),
        "fine_per_coarse": FINE_PER_COARSE,
        "n_modes": N_MODES,
        "n_tiers": N_TIERS,
        "n_problems": P,
        "problems_per_op": [int((op_col == op).sum()) for op in range(4)],
        "trials": int(cube["n"].sum()),
        "trials_clean": by_mode[0],
        "trials_indexed": int(c["n"].sum()),
        "min_problem_n": MIN_PROBLEM_N,
        "log_centre": LOG_CENTRE,
        **diag,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print({k: v for k, v in meta.items() if k not in ("fine_edges", "coarse_edges")})


if __name__ == "__main__":
    sessions, diag = sessions_table()
    print(diag, flush=True)
    if (WORK / "cube.parquet").exists() and "--reuse-cube" in sys.argv:
        cube = pl.read_parquet(WORK / "cube.parquet")
    else:
        cube = build_cube(sessions)
        cube.write_parquet(WORK / "cube.parquet")
    print("cube rows", cube.height, flush=True)
    export(cube, diag)
