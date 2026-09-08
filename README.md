# nflprops

Prediction and probability modeling for NFL player props: Anytime TD, Passing
Yards, Rushing Yards, Receiving Yards. Built for a new season, where Week 1
leans on last year and later weeks fold in new data on a per-metric decay
schedule.

Full design writeup, formulas, and feature reference: `docs/DESIGN.md`.

## Quickstart

```bash
pip install -r requirements.txt

# generate synthetic data so you can see the whole thing run
python scripts/make_sample_data.py --root data --season 2026 --week 1

# Week 1: no current-season data exists yet
python scripts/run_week.py --root data --season 2026 --week 1 \
    --out out/week1.json

# one player, full JSON
python scripts/run_week.py --root data --season 2026 --week 1 \
    --player KC_WR1 --pretty

# mid-season, ranked by edge against the posted lines
python scripts/make_sample_data.py --root data --season 2026 --week 5
python scripts/run_week.py --root data --season 2026 --week 5 --edges-only

# backtest and recalibrate on everything scored so far
python scripts/calibrate.py --root data --season 2026 --through-week 4 --start-week 2
```

The sample generator produces structurally correct but randomly generated data.
Its numbers are for exercising the plumbing, not for evaluating the model. The
calibration report on synthetic data will look poorly calibrated by design,
since the generator's touchdown process is unrelated to the model's structural
assumptions. Swap in real feeds (nflverse / nfl_data_py, PFF or equivalent
grades, your odds provider) and the column names already line up with
`schema.py`.

## Real data (nflverse)

`scripts/fetch_nflverse_data.py` pulls real NFL data, free, no API key, reading
directly from [nflverse-data](https://github.com/nflverse/nflverse-data)'s
GitHub-hosted releases (play-by-play, weekly stats, snap counts, rosters) and
[nfldata](https://github.com/nflverse/nfldata)'s schedule file (which includes
real spread/total lines, updated as sportsbooks post them).

```bash
# once, after a season ends (builds data/prior/*.csv from real 2025 data)
python scripts/fetch_nflverse_data.py prior --season 2025 --root data

# once, at the start of a new season (real team changes + rookies)
python scripts/fetch_nflverse_data.py roster --season 2026 --prior-season 2025 --root data

# every week, once games start (real game logs + that week's real schedule/lines)
python scripts/fetch_nflverse_data.py current --season 2026 --through-week 5 --root data
```

**What's real vs. what still needs your input:**

| Covered by nflverse (real, free) | Not available anywhere free |
|---|---|
| Player usage: targets, carries, snaps, red zone shares | OL/WR-corps/QB unit grades (`config.GRADE_PLACEHOLDER` fills in 65 until you supply real ones, e.g. from PFF) |
| Team offense/defense: EPA, pace proxy, red zone rate | Coordinator names, scheme labels, "new OC" flags |
| Team changes and rookie status (real, from roster files) | Depth chart ranks, camp-battle volatility |
| Game-level spread/total (real sportsbook lines) | Player prop lines (over/unders on yards, anytime-TD odds) |

The right-hand column is in `roster_changes.csv` and the team offense/defense
tables specifically so you can hand-edit it — those facts change by press
release and beat-reporter tweet, not by a clean weekly data feed. Nobody
publishes it in a scrapable format. Treat `roster_changes.csv` as a
spreadsheet you touch up once in August and lightly during the season, not
something to regenerate.

Player prop lines are deliberately out of scope for now (see the "Web page"
section below) — the pipeline runs fine with `prop_lines.csv` empty; it just
shows projections without market/edge columns.

**A note on the data source's own stability:** nflverse restructured its
player-stats file in August 2025 and the popular `nfl_data_py` wrapper
package was archived (unmaintained) around the same time. This script does
not depend on `nfl_data_py` at all -- it reads the nflverse-data release URLs
directly -- but that also means if nflverse changes a file layout again, this
script needs a matching update rather than a wrapper library's own bugfix.
The URLs and column names it depends on are all in one place at the top of
`fetch_nflverse_data.py` if that day comes.

## Web page (GitHub Pages, auto-updates Wednesdays)

The `docs/` folder is a self-contained static site: `index.html` + `app.js` +
`style.css`, reading `docs/data/props.json` and `docs/data/meta.json`. No
server, no build step.

**One-time setup:**

1. Push this repo to GitHub.
2. In the repo, go to Settings > Pages. Under "Build and deployment," set
   Source to "Deploy from a branch," Branch to `main`, folder to `/docs`.
   Save. GitHub gives you a URL like
   `https://<username>.github.io/<repo-name>/` within a minute or two.
3. That's it. The page is live now, showing whatever is currently in
   `docs/data/`.

**Automated weekly refresh:**

`.github/workflows/weekly-update.yml` runs every Wednesday (cron `0 18 * * 3`,
which is 2pm Eastern during daylight saving, 1pm once clocks fall back --
edit the cron hour if you want it pinned to a specific Eastern time year
round). It figures out the current season/week, rebuilds the data, and pushes
`docs/data/props.json` + `meta.json` back to the repo, which GitHub Pages
picks up automatically.

You can also trigger it by hand: repo > Actions tab > "Weekly NFL Props
Update" > Run workflow.

The workflow runs `fetch_nflverse_data.py current` every week automatically
(prior-season and roster tables are built once and skipped on later runs, so
it isn't re-downloading a full season of play-by-play every Wednesday). Prop
lines are still not wired up -- see the "Real data" section above.

**Local site preview**, without waiting for GitHub Pages:

```bash
python scripts/build_site_data.py --root data --season 2026 --week 1 --out docs/data
cd docs && python -m http.server 8000
# open http://localhost:8000
```



```python
from nflprops import DataStore, ProjectionEngine

store  = DataStore("data", season=2026, week=5)
engine = ProjectionEngine(store)

proj = engine.project_player("KC_WR1")
proj["markets"]["receiving_yards"]["projection"]
proj["markets"]["receiving_yards"]["lines"][0]["p_over"]
proj["markets"]["anytime_td"]["probability"]
proj["key_drivers"]

# probability against an arbitrary line, not just the posted one
p = engine.project_player("KC_WR1", lines={"receiving_yards": 62.5})

slate = engine.project_slate()
```

## Layout

```
nflprops/
  config.py            every tunable: K constants, rho multipliers, betas,
                       league baselines, dispersion, Kelly settings
  schema.py            all 7 table definitions; `python -m nflprops.schema`
                       prints the data dictionary
  features.py          feature engineering and the multiplier logic
  probability.py       predictive distributions, line probabilities,
                       intervals, de-vig, edge, Kelly
  projections.py       orchestrator: blend -> roster -> features -> models
  output.py            JSON Schema, driver ablation, assembly
  data/
    loaders.py         DataStore, recency-weighted current-season aggregation
    blend.py           three-way weighted decay blend
    roster.py          continuity rho, usage multipliers, renormalization
  models/
    anytime_td.py      structural Poisson + optional logistic calibration
    yardage.py         passing / rushing / receiving, opportunity x efficiency
scripts/
  make_sample_data.py  synthetic dataset generator
  run_week.py          weekly projection CLI
  calibrate.py         backtest, reliability, refit
docs/DESIGN.md         the writeup
```

## Design notes worth knowing before you tune anything

**Opportunity and efficiency are modeled separately.** They stabilize at
different rates (`rush_share` at K=3, `ypc` at K=12) and respond to different
roster changes. Collapsing them loses that.

**Roster changes act through three channels, not one.** Continuity `rho` moves
how much history is trusted. Usage multipliers move opportunity. Unit grade
deltas move efficiency. A trade can hit all three with different signs.

**Shares are renormalized per team.** Tag several players on one team as
promoted and the naive result is an offense with 130% of its own targets.

**Edge is computed against de-vigged prices.** Comparing a model probability to
a vig-inclusive implied probability invents 2 to 3 points of phantom edge on
every bet.

**Uncertainty widens intervals rather than shifting means.** A contested camp
battle should not make the projection lower; it should make it less confident,
which is what `role_uncertainty` does through the dispersion term and the Kelly
stake.

**The market is an input, not an opponent.** Implied team total anchors the
expected TD pool and the spread drives game script. The edge comes from
allocating that total to players better than the book does, not from
disagreeing with the total.
