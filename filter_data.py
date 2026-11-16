"""Filters for the September Zetamac release.

Rules match arithmetic_zetamac/zetamac_pipeline.py, including the
non-trivial / tie split used by classical_determinants.ipynb.

Order of operations
-------------------
Per session, while reading a shard:
  1. Drop the session if problem_log is longer than 100k characters (bot / giant).
  2. Drop the last trial. The timer cuts the game off mid-problem, so that
     trial is forced and incomplete. A log with only one trial is dropped
     entirely — that single trial is the forced one.
  3. Drop trials with a missing or negative timeMs.
  4. Drop trials whose problem string is not "a op b", or whose operands
     are longer than 15 digits (corrupted / overflow).

Then, on the session table:
  5. Drop sessions with no IP hash.
  6. Drop single-question drills (one operator, both operands fixed).
  7. Require more than 3 remaining trials.
  8. Settings. mode="default" keeps the stock Zetamac ranges only.
     mode="any" drops trivially easy ranges (any bound of 0 or 1) and, as a
     backstop, sessions that still contain a trial with an operand <= 1.
  9. Drop the whole session if it contains any trial at or under 100 ms
     (bot guard). A trial at exactly 100 ms counts.

Then, on the trials that remain:
  10. Drop trials slower than 90 s. The session stays.
  11. Drop trials whose keystroke count does not equal the number of digits
      in the correct answer. A missing entry log is kept, because there is
      nothing to judge. Backspace-and-retype fails this check on purpose.
  12. Optional split used by the classical-determinants notebook: zero
      answers and doubles (a op a) leave the main set; addition, multiplication,
      division, and subtraction ties are kept in their own tables.
"""

import glob
from pathlib import Path

import polars as pl

THRESHOLD_MS = 100
RT_FLOOR_MS = 100
MAX_TRIAL_MS = 90_000
MAX_LOG_CHARS = 100_000
MIN_TRIALS = 3

PROBLEM_RE = r"^\s*(?P<left>\d+)\s*(?P<op>[+\-*/])\s*(?P<right>\d+)\s*$"

# Stock Zetamac. Subtraction reuses the addition bounds; division reuses
# the multiplication bounds, so only these eight columns exist.
DEFAULT_RANGES = {
    "add_left_min": 2, "add_left_max": 100,
    "add_right_min": 2, "add_right_max": 100,
    "mul_left_min": 2, "mul_left_max": 12,
    "mul_right_min": 2, "mul_right_max": 100,
}

SHARD_GLOB = "data/sept-release/export-*.jsonl.gz"

_RAW_SCHEMA = {
    "completed_at": pl.String,
    "game_options": pl.String,
    "problem_log": pl.String,
    "score": pl.String,
    "rotating_ip_hash": pl.String,
}


def shard_paths(pattern: str = SHARD_GLOB) -> list[str]:
    return sorted(glob.glob(pattern))


def read_shard(path: str) -> pl.DataFrame:
    return pl.read_ndjson(path, schema=_RAW_SCHEMA)


def _decode_json(series: pl.Series) -> pl.Series:
    """The logs are Python reprs, not JSON. Also null integers with 16+ digits.

    A row that still will not parse is nulled rather than failing the shard.
    """
    import json

    fixed = (
        series
        .str.replace_all(r"\\x([0-9a-fA-F]{2})", r"\\u00$1")
        .str.replace_all("True", "true", literal=True)
        .str.replace_all("False", "false", literal=True)
        .str.replace_all("None", "null", literal=True)
        .str.replace_all("'", '"', literal=True)
        .str.replace_all(r":\s*-?\d{16,}(\s*[,}\]])", r": null$1")
    )
    try:
        return fixed.str.json_decode(infer_schema_length=None)
    except pl.exceptions.ComputeError:
        def _ok(s):
            if s is None:
                return True
            try:
                json.loads(s)
                return True
            except Exception:
                return False

        ok = fixed.map_elements(_ok, return_dtype=pl.Boolean)
        cleaned = (
            pl.DataFrame({"s": fixed, "ok": ok})
            .select(pl.when(pl.col("ok")).then(pl.col("s")).otherwise(None))
            .to_series()
        )
        return cleaned.str.json_decode(infer_schema_length=None)


def _drop_last_trial(log: pl.Expr) -> pl.Expr:
    """Drop the forced final trial. A one-trial log contributes nothing."""
    n = log.list.len()
    return (
        pl.when(n > 1)
        .then(log.list.slice(0, n - 1))
        .otherwise(log.list.slice(0, 0))
    )


def _parse_operands(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    extracted = df["problem"].str.extract_groups(PROBLEM_RE)
    df = df.with_columns(extracted.alias("g")).unnest("g")
    ok = (
        df["left"].is_not_null()
        & (df["left"].str.len_chars() <= 15)
        & (df["right"].str.len_chars() <= 15)
    )
    n_bad = int((~ok).sum())
    df = df.filter(ok).with_columns(
        pl.col("left").cast(pl.Int64),
        pl.col("right").cast(pl.Int64),
    )
    cols = ["session", "rotating_ip_hash", "problem", "left", "op", "right", "timeMs"]
    if "entry" in df.columns:
        cols.append("entry")
    return df.select(cols), n_bad


def explode_shard(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """One shard -> (trials with operands parsed, session rows, counts).

    Trials here still include sub-100 ms and >90 s rows. Session cleaning
    and the 90 s cutoff happen later, because both need the whole release.
    """
    n_raw = raw.height
    raw = raw.filter(pl.col("problem_log").str.len_chars() <= MAX_LOG_CHARS)
    n_giant = n_raw - raw.height

    log = _decode_json(raw["problem_log"])
    opts = _decode_json(raw["game_options"])

    trials = (
        raw.with_columns(log.alias("log"))
        .with_columns(_drop_last_trial(pl.col("log")).alias("log"))
        .select(["completed_at", "rotating_ip_hash", "log"])
        .rename({"completed_at": "session"})
        .explode("log", empty_as_null=True)
        .unnest("log")
        .with_columns(pl.col("timeMs").cast(pl.Float64, strict=False).fill_null(-1))
    )
    valid = trials.filter(pl.col("timeMs") >= 0)
    parsed, n_bad = _parse_operands(valid)

    n_trials = parsed.group_by("session").len().rename({"len": "n_trials"})
    sessions = (
        raw.select(["completed_at", "rotating_ip_hash"])
        .rename({"completed_at": "session"})
        .with_columns(opts.alias("opts"))
        .unnest("opts")
        .join(n_trials, on="session", how="left")
        .with_columns(pl.col("n_trials").fill_null(0))
    )
    stats = {
        "n_sessions_raw": n_raw,
        "n_giant_dropped": n_giant,
        "n_valid_trials": valid.height,
        "n_malformed_operands": n_bad,
    }
    return parsed, sessions, stats


def _flag(df: pl.DataFrame, column: str) -> pl.Expr:
    if column not in df.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(False)


def _single_question(columns) -> pl.Expr:
    """An operator is on and both of its operands are pinned to one value."""
    parts = []
    for prefix in ("add", "sub", "mul"):
        names = [
            f"{prefix}_left_min", f"{prefix}_left_max",
            f"{prefix}_right_min", f"{prefix}_right_max",
        ]
        if prefix not in columns or not all(c in columns for c in names):
            continue
        lo_min, lo_max, hi_min, hi_max = names
        parts.append(
            pl.col(prefix).fill_null(False)
            & pl.col(lo_min).is_not_null()
            & (pl.col(lo_min) == pl.col(lo_max))
            & pl.col(hi_min).is_not_null()
            & (pl.col(hi_min) == pl.col(hi_max))
        )
    return pl.any_horizontal(*parts) if parts else pl.lit(False)


def _trivial_range(df: pl.DataFrame, prefix: str) -> pl.Expr:
    cols = [
        c for side in ("left", "right") for bound in ("min", "max")
        if (c := f"{prefix}_{side}_{bound}") in df.columns
    ]
    if not cols:
        return pl.lit(False)
    return pl.any_horizontal([pl.col(c) <= 1 for c in cols])


def clean_sessions(
    sessions: pl.DataFrame,
    trials: pl.DataFrame,
    mode: str = "default",
    min_trials: int = MIN_TRIALS,
) -> tuple[pl.DataFrame, dict]:
    """Return (clean session ids, diagnostics).

    mode="default" keeps stock ranges. mode="any" keeps every non-trivial range.
    """
    if mode not in ("default", "any"):
        raise ValueError(f"mode must be 'default' or 'any', got {mode!r}")

    has_ip = pl.col("rotating_ip_hash").is_not_null()
    pre = sessions.filter(
        has_ip
        & ~_single_question(sessions.columns)
        & (pl.col("n_trials") > min_trials)
    )

    diag = {
        "mode": mode,
        "total_sessions": sessions.height,
        "n_null_ip": int(sessions["rotating_ip_hash"].null_count()),
        "pre_settings_eligible": pre.height,
    }

    if mode == "default":
        default = pl.lit(True)
        for col, val in DEFAULT_RANGES.items():
            if col in sessions.columns:
                default = default & (pl.col(col) == val)
        diag["dropped_non_default"] = pre.filter(~default).height
        eligible = pre.filter(default)
    else:
        trivial = (
            ((_flag(sessions, "add") | _flag(sessions, "sub")) & _trivial_range(sessions, "add"))
            | ((_flag(sessions, "mul") | _flag(sessions, "div")) & _trivial_range(sessions, "mul"))
        ).fill_null(False)
        diag["dropped_trivial"] = pre.filter(trivial).height
        eligible = pre.filter(~trivial)
        # Corrupted range columns can hide a 0/1 operand. Drop those sessions too.
        if trials.height:
            trivial_trial_sessions = (
                trials.filter(pl.min_horizontal("left", "right") <= 1)
                .select("session").unique()
            )
            before = eligible.height
            eligible = eligible.join(trivial_trial_sessions, on="session", how="anti")
            diag["dropped_trivial_trial_backstop"] = before - eligible.height

    # Bot guard: any trial at or under 100 ms drops the session.
    fast_sessions = (
        trials.filter(pl.col("timeMs") <= THRESHOLD_MS)
        .select("session").unique()
    )
    clean = eligible.join(fast_sessions, on="session", how="anti").select("session")
    diag["eligible"] = eligible.height
    diag["dropped_sub100"] = eligible.height - clean.height
    diag["clean_final"] = clean.height
    return clean, diag


def reference_trials(trials: pl.DataFrame, clean_sids: pl.DataFrame) -> pl.DataFrame:
    """Clean-session trials at or above 100 ms, with the >90 s rows removed.

    The session is kept. correct_ans is filled in, and trials whose keystroke
    count disagrees with the answer's digit count are dropped.
    """
    n_keystrokes_ok = (
        pl.col("entry").list.len()
        == pl.col("correct_ans").cast(pl.Utf8).str.len_chars()
    )
    lf = trials.lazy().join(clean_sids.lazy(), on="session", how="semi")
    return (
        lf.filter(pl.col("timeMs").is_between(RT_FLOOR_MS, MAX_TRIAL_MS))
        .with_columns(
            pl.when(pl.col("op") == "+").then(pl.col("left") + pl.col("right"))
            .when(pl.col("op") == "-").then(pl.col("left") - pl.col("right"))
            .when(pl.col("op") == "*").then(pl.col("left") * pl.col("right"))
            .when(pl.col("op") == "/").then(
                pl.col("left") // pl.when(pl.col("right") == 0).then(1).otherwise(pl.col("right"))
            )
            .alias("correct_ans")
        )
        .filter(pl.col("entry").is_null() | n_keystrokes_ok)
        .drop("entry")
        .collect(engine="streaming")
    )


def split_nontrivial(ref_clean: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Main set used by classical determinants, plus the tie tables it sets aside.

    Non-trivial drops zero answers and doubles (a op a). Division ties
    (49/7 = 7) are removed from that set and kept on their own. Addition,
    multiplication, and subtraction ties are taken from the full clean set,
    because doubles were just excluded.
    """
    not_zero_or_double = ~((pl.col("correct_ans") == 0) | (pl.col("left") == pl.col("right")))
    nontrivial = ref_clean.filter(not_zero_or_double)
    div_tie = (pl.col("op") == "/") & (pl.col("correct_ans") == pl.col("right"))
    return {
        "nontrivial": nontrivial.filter(~div_tie),
        "div_ties": nontrivial.filter(div_tie),
        "add_ties": ref_clean.filter(
            (pl.col("op") == "+") & (pl.col("left") == pl.col("right")) & (pl.col("correct_ans") != 0)
        ),
        "mul_ties": ref_clean.filter(
            (pl.col("op") == "*") & (pl.col("left") == pl.col("right")) & (pl.col("correct_ans") != 0)
        ),
        "sub_ties": ref_clean.filter(
            (pl.col("op") == "-") & (pl.col("correct_ans") == pl.col("right")) & (pl.col("correct_ans") != 0)
        ),
    }
