# Zetamac explorer

A single-page site for the Zetamac dataset. Open `index.html` in a browser; it needs no build step or server. Data lives in `data/*.js`, built by:

```
cd pipeline && python3.12 site_data.py
```

That needs `work/` from `extract.py`.

## Data

All numbers come from clean sessions (default cleaning in `build.sessions_table`, which includes stock ranges) and trials with 100 ms < RT ≤ 90 s.

- `data/{add,sub,mul,div}.js`: one cell per problem on the stock grid, `[median ms, trials, first-try %, hist]`. First-try means keystrokes equal the answer's digit count; trials without an entry log are left out of that rate. `hist` is 32 bins of 250 ms over 0–8 s plus one overflow bin.
- `data/sessions.js`: two real 120 s games with all four operators on default settings, picked at random among sessions scoring exactly the median (typical) and the 95th percentile (fast). Keys are rebuilt from the logged input-box states, with one character per press and `<` for backspace.

Problem coordinates `(op, r, c)`:

| op | r | c | Shown as |
|---|---|---|---|
| `add` | first operand (2–100) | second operand (2–100) | `r + c` |
| `sub` | subtrahend (2–100) | difference (2–100) | `(r+c) − r` |
| `mul` | small factor (2–12) | large factor (2–100) | `r × c` |
| `div` | divisor (2–12) | quotient (2–100) | `(r·c) ÷ r` |

## Approximations on the page

- Keystrokes are spread evenly across each trial's RT, because the logs have trial RTs, not per-key timestamps.
- The falling dots are 420 random draws from the problem's histogram. The counter scales landed dots to the trial count, so it ends below the total by the share over 8 s.
- "Faster than X% of attempts" interpolates linearly within the 250 ms bin.
- Heatmap cells pool 3×3 problems (3×1 for × and ÷) and show the unweighted mean of their medians.
