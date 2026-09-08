# NFL Player Prop Model: Design

Markets covered: Anytime TD, Passing Yards, Rushing Yards, Receiving Yards.

The organizing principle is that every projection splits into **opportunity**
and **efficiency**, and those two halves have different stabilization rates,
different roster-change sensitivities, and different noise levels. Target share
is knowable by Week 4. Yards per target is not knowable by Week 12. Modeling
them as one blob throws away that structure.

---

## 1. Data Pipeline

### 1.1 Required inputs from last season

**Player season table** (`data/prior/player_season.csv`)

Usage: `snap_share`, `route_participation`, `target_share`, `air_yards_share`,
`rush_share`, `carries_per_game`, `targets_per_game`.
Red zone: `rz_target_share`, `rz_rush_share`, `inside5_rush_share`,
`inside10_target_share`.
Efficiency: `ypc`, `yptarget`, `adot`, `catch_rate`, `yac_per_rec`, `ypa`,
`sack_rate`, `pass_att_share`.
Production: `rush_td`, `rec_td`, `pass_td`, and the three yardage totals.

**Team offense** (`data/prior/team_season_offense.csv`)

Pace and volume: `plays_per_game`, `pace_sec_per_play`, `pass_rate`, `proe`.
Red zone: `rz_pass_rate`, `rz_trips_per_game`, `inside5_plays_pg`.
Scoring: `points_per_game`, `off_td_per_game`, `rush_td_share`,
`off_epa_per_play`.
Units and personnel: `ol_passblock_grade`, `ol_runblock_grade`,
`ol_snaps_returning`, `wr_corps_grade`, `qb_grade`.
Scheme identity: `offensive_coordinator`, `head_coach`, `play_caller`,
`scheme_run`, `scheme_pass`.

**Team defense** (`data/prior/team_season_defense.csv`)

EPA allowed: `def_epa_per_play_allowed`, `def_epa_per_pass_allowed`,
`def_epa_per_rush_allowed`.
Raw allowed: `def_ypa_allowed`, `def_ypc_allowed`, and the per-game yardage
versions.
Scoring allowed: `def_rz_td_rate_allowed`, `def_rush_td_pg_allowed`,
`def_pass_td_pg_allowed`.
Positional splits: `def_vs_wr_out`, `def_vs_wr_slot`, `def_vs_te`,
`def_vs_rb_pass`.
Coverage personnel: `cb1_grade`, `cb2_grade`, `slot_cb_grade`, `shadow_cb`,
`def_man_rate`, `def_zone_rate`, `def_light_box_rate`.
Pace faced: `def_plays_pg_faced`, `def_pace_sec_allowed`.

### 1.2 Required inputs once the current season begins

Three append-only game log files under `data/current/`:
`player_gamelog.csv`, `team_gamelog_offense.csv`, `team_gamelog_defense.csv`.

The player log carries both the player numerator and the team denominator on
every row (`targets` and `team_targets`, `carries` and `team_carries`,
`inside5_carries` and `team_inside5_carries`, and so on). Storing the
denominator inline is what makes recency-weighted shares computable in one pass
without a join, and it makes a mid-season trade harmless: the shares
automatically re-base to whichever team the player was on that week.

Two more files refresh weekly regardless of season stage:
`game_environment.csv` (spread, total, implied team total, dome, wind, precip)
and `prop_lines.csv` (posted numbers and prices).

### 1.3 Merging last season with the new season

Three sources compete for weight on every metric: the current season, the prior
season, and a league or positional baseline.

```
w_cur    = n_games_current / (n_games_current + K_metric)
w_prior  = (1 - w_cur) * rho
w_league = (1 - w_cur) * (1 - rho)

x_blend  = w_cur*x_cur + w_prior*x_prior + w_league*x_league
```

`K_metric` is the stabilization constant: the number of current-season games at
which prior and current get equal weight. It is not one number for the whole
model, because stats do not stabilize at the same rate.

| metric | K | 50/50 crossover | week 4 weight on current |
|---|---|---|---|
| `rush_share` | 3 | 3 games | 0.50 |
| `target_share` | 4 | 4 games | 0.43 |
| `pass_rate`, `pace` | 4 | 4 games | 0.43 |
| `rz_target_share` | 6 | 6 games | 0.33 |
| `yptarget` | 9 | 9 games | 0.25 |
| `ypa` | 10 | 10 games | 0.23 |
| `ypc` | 12 | 12 games | 0.20 |
| `td_rate` | 16 | 16 games | 0.16 |

Week 1 is the special case: `n_games_current = 0`, so `w_cur = 0` and the split
is entirely prior versus baseline, governed by `rho`. A returning starter on the
same team with the same coordinator sits at roughly 92/8 prior/league. A player
who changed teams drops to about 57/43. A rookie sits at 0/100 on the positional
baseline until real snaps exist.

If you want the literal 70/30 behavior instead of the per-metric version, set
`USE_SCHEDULE_OVERRIDE = True` in `config.py` and edit `WEEK_WEIGHT_SCHEDULE`.
The per-metric version is better and is the default, but the override exists so
the two can be compared in backtest.

Within the current season, games are not weighted equally either. Before
averaging, each game gets `0.93 ** (weeks_ago)`, so a Week 10 game carries about
1.9x the weight of a Week 1 game when projecting Week 11.

### 1.4 Roster-change logic

Roster changes act through two separate channels, which is the part most
implementations get wrong by conflating them.

**Channel 1: continuity (`rho`).** Controls how much of the player's own history
is predictive. It moves the *blend*, not the *level*. A receiver who signed
elsewhere did not necessarily get worse; we are just less confident his old
profile describes his new role.

```
rho = 0.92 * prod(triggered multipliers), floored at 0.25

changed_team           0.62
new_hc_offense         0.72
scheme_change_major    0.70
new_oc                 0.80
new_qb                 0.82   (pass catchers only)
new_dc                 0.85   (team defense rows only)
returning_from_ir      0.85
depth chart move       0.65 ^ (0.55 + 0.45*(steps-1))
camp volatility        (1 - 0.35 * volatility)
rookie                 rho = 0
```

**Channel 2: usage shift (`mu`).** A directional change in expected opportunity,
applied after blending, then renormalized so each team's shares still sum to 1.

```
backfield_competitor_removed    1.30
target_competitor_removed_wr1   1.22
moved_to_starter                1.55
backfield_competitor_added      0.74
moved_to_backup                 0.45
```

Renormalization matters. Tag three receivers on one team as promoted and the
naive result is a 130% offense. `roster.renormalize_shares` rescales within each
team and position group so the shares stay a valid partition.

**Channel 3 (efficiency, handled in `features.py`).** Unit changes move yards per
opportunity rather than opportunity count:

```
ypa      += 0.16 * d_ol_passblock_z + 0.21 * d_wr_corps_z
ypc      += 0.19 * d_ol_runblock_z + 0.25 * (ol_snaps_returning - 0.70)
yptarget += 0.32 * d_qb_z
sack_rate -= 0.012 * d_ol_passblock_z
```

Coordinator changes also propagate to the *team* level through `_team_rho`, so a
new play caller makes last year's pass rate and pace less predictive of this
year's, exactly the way a trade makes a player's old share less predictive.

### 1.5 Dataset structure

Full column-by-column dictionary with dtypes and descriptions:
`python -m nflprops.schema`. Seven tables: `player_season`,
`team_season_offense`, `team_season_defense`, `player_gamelog`,
`roster_changes`, `game_environment`, `prop_lines`.

---

## 2. Modeling Approach

### 2.1 Anytime Touchdown

Top-down Poisson rather than a player-level logistic. The reason is sample size:
a player-level logistic has to learn a rare event from a handful of positive
cases per player, while the decomposition below learns each piece from a place
where data is abundant.

```
E[team_off_TD] = 0.70*(-0.42 + 0.1205 * implied_total)
               + 0.30*(hist_td_per_game * def_rz_adjustment)

rush_share_of_td = 0.55*hist_rush_td_share
                 + 0.45*(1 - 0.86*rz_pass_rate)
                 then blended 72/28 with the defense's own rush/pass TD split

lambda_rush = E[team_rush_TD] * (0.62*inside5_rush_share
                               + 0.28*rz_rush_share
                               + 0.10*rush_share)

lambda_rec  = E[team_rec_TD]  * (0.45*inside10_target_share
                               + 0.33*rz_target_share
                               + 0.22*air_yards_share)

lambda = (lambda_rush * rush_matchup + lambda_rec * pass_matchup)
         * def_rz_adj * roster_usage_mult

P(ATD) = 1 - exp(-lambda * 0.965)
```

Air yards share carries weight in the receiving component even though it is not
a red zone stat, because a meaningful share of receiving touchdowns come from
explosive plays outside the red zone rather than goal-line targets.

Position handling: QBs lose the receiving term entirely and use designed
goal-line rush share; RBs get their receiving TD term damped to 0.85; WRs and
TEs get their rushing term damped to 0.35 (jet sweeps only).

The `0.965` kappa exists because a raw Poisson slightly overstates the multi-TD
tail. Once you have scored weeks, `fit_calibration` replaces it with a fitted
`logit(p) = a + b*log(lambda)`, which corrects any systematic bias in the link
without touching the structural model. A fitted slope below 1.0 means the
lambdas are too spread out; above 1.0 means too compressed.

### 2.2 Passing Yards

```
attempts = team_dropbacks * qb_dropback_share * (1 - sack_rate)
  where team_dropbacks = team_plays * pass_rate
        team_plays     = 0.5*(own_plays_pg + opp_plays_faced_pg) * pace_ratio
                       + 0.22*(total - 44)
        pass_rate      = blended_pass_rate + 0.35*proe + 0.0068*spread
        sack_rate      = blended_sack_rate - 0.012*d_ol_passblock_z

ypa = blended_ypa * pass_matchup_mult * weather_penalty + ypa_unit_delta

yards ~ Normal(attempts * ypa, sd)
```

`spread` is positive for underdogs, so trailing teams throw more. `pace_ratio`
compares the team's seconds per play to league average and is capped at
[0.88, 1.14] so one extreme pace figure cannot dominate.

### 2.3 Rushing Yards

```
carries = team_carries * (blended_rush_share - 0.0052*spread)
  where team_carries = team_plays * (1 - pass_rate)

ypc = blended_ypc * rush_matchup_mult + ypc_unit_delta

yards ~ Gamma(shape, scale) with mean carries*ypc, zero-inflated
```

The negative spread coefficient is the game-script effect: favorites run more.
The run matchup multiplier blends the opponent's rush EPA allowed with raw YPC
allowed (45/55) and then adds a light-box adjustment, since a defense that plays
six in the box concedes yards per carry that its EPA figure understates.

### 2.4 Receiving Yards

```
targets = team_pass_attempts * target_share * matchup_share_mult
        (target_share already carries the roster usage multiplier and
         team renormalization from the roster stage)

ypt = blended_yptarget
      * pass_matchup_mult
      * wr_cb_matchup_mult
      * weather_penalty
      * air_yards_correction
      + qb_quality_delta

yards ~ Gamma(shape, scale), zero-inflated
```

The air yards correction is a second opinion on efficiency: if a receiver's air
yards share materially exceeds his target share, his role is deeper than raw
yards-per-target history implies, and ypt is nudged up to +16%.

### 2.5 Distributions

| market | family | base CV | rationale |
|---|---|---|---|
| passing yards | Normal | 0.265 | 30+ attempts, CLT holds |
| rushing yards | Gamma | 0.480 | right skew; a 70-yard run is possible, -20 yards is not |
| receiving yards | Gamma + zero inflation | 0.585 | genuine probability mass at exactly 0 |

Dispersion is heteroskedastic: `cv = cv_base * (mu_ref/mu)^0.16`. A projected
300-yard passer is proportionally more predictable than a projected 150-yard
passer. Role uncertainty then inflates it further:
`cv *= 1 + 0.35 * role_uncertainty`, so a player with a contested camp battle
gets a wider interval rather than a shifted mean. That is the honest response to
not knowing the role: widen, do not guess.

---

## 3. Feature Reference

| feature | what it is | which props it moves and how |
|---|---|---|
| `def_epa_per_pass_allowed` | opponent EPA per dropback | pass matchup multiplier: `1 + 0.55*(x - league)`, blended 50/50 with the YPA ratio, capped [0.86, 1.16]. Raises QB yards and every receiver's ypt. |
| `def_epa_per_rush_allowed` | opponent EPA per rush | run matchup multiplier: `1 + 0.60*(x - league)`, capped [0.84, 1.18]. Moves RB ypc and the rushing TD pool. |
| `def_ypa_allowed`, `def_ypc_allowed` | raw allowed efficiency | blended with the EPA versions to damp small-sample EPA noise. |
| `wr_cb_matchup_grade` | `(receiver grade - assigned CB grade)/10`, where the assignment depends on slot rate, depth rank, and whether CB1 shadows | ypt multiplier `1 + 0.055*gap`; target share nudge `1 + 0.020*gap`. Volume is scheme-driven, so the share effect is deliberately small. |
| `rz_pass_rate` / rz rush rate | play calling inside the 20 | splits `E[team TD]` into rushing and passing pools. This is the main lever separating a goal-line back's ATD price from a slot receiver's. |
| `inside5_rush_share` | player share of goal-line carries | largest single ATD driver for RBs (weight 0.62 in the rush lambda). |
| `proe` | pass rate over expectation | adds `0.35*proe` to projected pass rate, on top of the spread-driven script effect. Separates teams that throw because they are losing from teams that throw by identity. |
| `pace_sec_per_play` | seconds per play, neutral script | drives `pace_ratio`, which scales total plays and therefore every volume-based prop. Combined with the opponent's pace faced, since game pace is a two-team negotiation. |
| `depth_chart_volatility` | 0 to 1 disagreement about the role | does not move the mean. Cuts `rho` by up to 35%, widens the interval, and shrinks the Kelly stake. |
| `roster_change_multiplier` | product of usage tags | multiplies shares directly, then renormalized within team. |
| `continuity_rho` | blend weight on the player's own history | the single number that answers "how much do last year's numbers still describe this player." |
| `d_ol_passblock_z`, `d_ol_runblock_z` | year-over-year unit grade change in z units | `+0.16 ypa` and `+0.19 ypc` per z respectively; pass block also cuts sack rate, which returns attempts. |
| `d_wr_corps_z`, `d_qb_z` | receiver room and QB quality change | `+0.21 ypa` and `+0.32 ypt` per z. |
| `ol_snaps_returning` | fraction of prior-season OL snaps back | cohesion term on ypc: `0.25*(x - 0.70)`. |
| `game_script_index` | spread scaled to roughly [-1.5, 1.5] | favorites run more and throw less; underdogs the reverse. |
| `team_total_implied` | market's expected team points | anchors the expected-TD pool. The market is the sharpest single input in the system; the model uses it rather than competing with it. |
| `weather_penalty` | wind and precipitation | multiplicative on passing efficiency only, floored at 0.82. |

---

## 4. Output Format

One JSON object per player-week. Full JSON Schema in `nflprops/output.py`
(`JSON_SCHEMA`). Top-level blocks:

- `game_environment`: spread, total, implied team total, projected team plays,
  pass attempts, carries, expected offensive TDs
- `data_blend`: games of current-season sample, `continuity_rho`,
  `role_uncertainty`, and the per-metric weight breakdown so any number is
  traceable back to its sources
- `markets`: per market, the projection, median, sd, distribution family,
  `p_zero`, 50% and 80% intervals, expected opportunities, expected yards per
  opportunity, and a `lines` array with `p_over`, `p_under`, fair odds, the
  book's price, the de-vigged market probability, edge in percentage points,
  recommended side, and quarter-Kelly stake
- `key_drivers`: ranked, with a signed `impact` in the units of the market
- `flags`: `high_role_uncertainty`, `major_roster_or_scheme_change`,
  `volatile_depth_chart`, `no_current_season_sample`, `adverse_weather`

`key_drivers` is computed by ablation, not by hand-waving: each multiplier is
reset to neutral, the deterministic projection is recomputed, and the difference
is recorded. That yields an auditable answer to "why is this 12 yards above the
book's number."

Edge uses the de-vigged two-way market price, not the raw implied probability.
Comparing a model probability against a vig-inclusive price manufactures a
roughly 2 to 3 point phantom edge on every bet.

---

## 6. Automation

### Weekly cadence

| when | step | command |
|---|---|---|
| Tue AM | append last week's game logs to `data/current/` | your ETL job |
| Tue AM | update `roster_changes.csv` for trades, IR, depth chart moves | manual or feed |
| Wed | rerun backtest and recalibration on everything scored so far | `python scripts/calibrate.py --root data --season 2026 --through-week N` |
| Wed | review calibration report; update `config.py` if the empirical CV has drifted more than about 15% from the configured value | manual |
| Thu | refresh `game_environment.csv` with opening spreads and totals | your odds feed |
| Thu | first projection pass | `python scripts/run_week.py --root data --season 2026 --week N+1 --out out/weekN1.json` |
| Fri to Sun | refresh `prop_lines.csv` and rerun for line movement | same command |
| Sun AM | final pass with inactives applied via `manual_share_override` | same command |

Leakage is prevented structurally: `DataStore._filter_to_week` drops every game
log row with `week >= target_week`, so a backtest cannot see the week it is
predicting even if the file contains it.

### Monitoring thresholds

Alert when any of these trip:

- absolute mean bias above 8 yards on any yardage market over the trailing 3 weeks
- 80% interval coverage outside 0.72 to 0.88
- ATD Brier skill score below 0 versus the base rate
- more than 25% of the slate flagged `high_role_uncertainty` (usually means the
  roster file went stale)
- mean predicted ATD probability drifting more than 3 points from the observed
  base rate

### Retraining cadence

- Weeks 1 to 3: do not refit anything. Sample is too small, and the structural
  priors are doing the work they were built for.
- Weeks 4 to 6: refit the ATD calibration layer only.
- Week 7 onward: refit the ATD calibration and the yardage residual models
  weekly, and update `DISPERSION_CV` from the observed empirical CV.
- Offseason: re-estimate every coefficient in `config.py` on the full season.

### Extending

Add a market by subclassing `BaseYardageModel`, implementing `opportunity` and
`efficiency`, and registering it in `MODEL_REGISTRY`. Receptions, rush plus
receive yards, and longest reception all fit this shape without touching the
pipeline. Add a feature by writing it into `build_feature_row` and appending it
to `FEATURE_ORDER`, which keeps the learned residual layer's vector aligned.
