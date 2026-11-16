"""Shared constants for the explorer pipeline. Rules follow filter_data.py."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHARD_GLOB = str(ROOT / "data/sept-release/export-*.jsonl.gz")
WORK = ROOT / "work"
OUT = ROOT / "web/public/data"

THRESHOLD_MS = 100        # any trial at or under this drops the whole session
MAX_TRIAL_MS = 90_000     # slower trials are dropped, session stays
MAX_LOG_TRIALS = 1500     # default-cleaning stand-in for the 100k-character giant-log rule
MIN_TRIALS = 3            # sessions need more than this many trials
MIN_GROUP_SESSIONS = 1000 # settings groups smaller than this get no skill tiers
MIN_PROBLEM_N = 100       # problems with fewer clean trials are not indexed

OPS = ["+", "-", "*", "/"]

DEFAULT_RANGES = {
    "add_left_min": 2, "add_left_max": 100,
    "add_right_min": 2, "add_right_max": 100,
    "mul_left_min": 2, "mul_left_max": 12,
    "mul_right_min": 2, "mul_right_max": 100,
}

# Slider stops. Every user-facing floor / max is one of these.
COARSE_EDGES = [
    100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000,
    3000, 5000, 7500, 10000, 15000, 20000, 30000, 45000, 60000, 90000,
]
FINE_PER_COARSE = 8


def fine_edges() -> np.ndarray:
    """Log-uniform subdivision of each coarse bin, so coarse edges stay exact."""
    edges = []
    for lo, hi in zip(COARSE_EDGES[:-1], COARSE_EDGES[1:]):
        edges.extend(np.geomspace(lo, hi, FINE_PER_COARSE, endpoint=False))
    edges.append(COARSE_EDGES[-1])
    return np.array(edges, dtype=np.float64)


FINE_EDGES = fine_edges()
N_FINE = len(FINE_EDGES) - 1
N_COARSE = len(COARSE_EDGES) - 1
N_TIERS = 6  # quintiles 0..4, 5 = unranked
N_MODES = 2  # 0 = stock ranges, 1 = custom ranges
