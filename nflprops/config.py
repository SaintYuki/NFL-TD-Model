"""
Global configuration: blending weights, stabilization constants, roster-change
multipliers, distribution parameters, and league baselines.

Everything here is a tunable. Values shipped are reasonable priors from
league-average NFL behavior; recalibrate them with `scripts/calibrate.py`
once you have two or three weeks of scored results.
"""

from dataclasses import dataclass, field
from typing import Dict

# ----------------------------------------------------------------------------
# 1. Blending / weighted decay
# ----------------------------------------------------------------------------
# Blend rule for any rate metric x:
#
#   w_cur    = n_games_current / (n_games_current + K_metric)
#   w_prior  = (1 - w_cur) * rho          # rho = continuity factor, see below
#   w_league = (1 - w_cur) * (1 - rho)
#   x_blend  = w_cur*x_cur + w_prior*x_prior + w_league*x_league
#
# K_metric is the "stabilization constant": the number of current-season games
# at which prior-season data and current-season data are weighted 50/50.
# Small K = the stat stabilizes fast (usage shares). Large K = noisy stat that
# needs a long look-back (yards per carry, TD rate).

STABILIZATION_K: Dict[str, float] = {
    # usage / opportunity - stabilizes fast
    "snap_share": 3.0,
    "route_participation": 3.0,
    "target_share": 4.0,
    "rush_share": 3.0,
    "rz_rush_share": 5.0,
    "rz_target_share": 6.0,
    "inside5_rush_share": 6.0,
    # team-level tendencies
    "pace_sec_per_play": 4.0,
    "pass_rate": 4.0,
    "proe": 5.0,
    "plays_per_game": 5.0,
    "rz_pass_rate": 6.0,
    # efficiency - stabilizes slowly
    "ypc": 12.0,
    "ypa": 10.0,
    "yptarget": 9.0,
    "adot": 5.0,
    "catch_rate": 8.0,
    "td_rate": 16.0,
    # defense
    "def_epa_per_play_allowed": 6.0,
    "def_epa_per_rush_allowed": 7.0,
    "def_epa_per_pass_allowed": 6.0,
    "def_ypc_allowed": 9.0,
    "def_ypa_allowed": 8.0,
    "def_rz_td_rate_allowed": 10.0,
    "_default": 6.0,
}

# Optional hard override that reproduces a "70/30 early season" schedule.
# If USE_SCHEDULE_OVERRIDE is True, these weights replace the K-based w_cur.
USE_SCHEDULE_OVERRIDE = False
WEEK_WEIGHT_SCHEDULE: Dict[int, float] = {  # week -> weight on CURRENT season
    1: 0.00, 2: 0.15, 3: 0.30, 4: 0.42, 5: 0.52, 6: 0.60, 7: 0.67,
    8: 0.72, 9: 0.77, 10: 0.81, 11: 0.84, 12: 0.87, 13: 0.89,
    14: 0.91, 15: 0.93, 16: 0.94, 17: 0.95, 18: 0.96,
}

# Within-season recency decay applied to current-season game logs before they
# are averaged. weight_g = DECAY_HALF_LIFE ** (weeks_ago)
GAMELOG_DECAY_PER_WEEK = 0.93

# ----------------------------------------------------------------------------
# 2. Roster / scheme continuity factors (rho)
# ----------------------------------------------------------------------------
# rho controls how much of a player's prior-season profile survives into the
# new season. rho = 1.0 means "last year's numbers are fully mine"; rho = 0.0
# means "throw it out, regress me to the league/positional baseline."
# Multiply the applicable factors together, then floor at RHO_FLOOR.

RHO_BASE = 0.92                 # same team, same scheme, same role
RHO_FLOOR = 0.25

CONTINUITY_MULTIPLIERS: Dict[str, float] = {
    "changed_team": 0.62,           # new team entirely
    "new_oc": 0.80,                 # same team, new offensive coordinator
    "new_hc_offense": 0.72,         # new HC who calls plays / sets identity
    "new_dc": 0.85,                 # defense-side equivalent (team defense rows)
    "scheme_change_major": 0.70,    # e.g. wide zone -> gap, air raid -> play-action
    "new_qb": 0.82,                 # pass catchers only
    "depth_chart_promotion": 0.65,  # WR3 -> WR1, RB2 -> RB1
    "depth_chart_demotion": 0.65,
    "returning_from_ir": 0.85,
    "rookie": 0.0,                  # no prior NFL data at all
    "year2_leap_candidate": 0.85,
}

# ----------------------------------------------------------------------------
# 3. Usage-shift multipliers applied AFTER blending
# ----------------------------------------------------------------------------
# These move the projected opportunity share, not the efficiency. They are
# applied multiplicatively to the blended share, then the team's shares are
# renormalized so they sum to 1.0 within each team/stat.

USAGE_SHIFT: Dict[str, float] = {
    "target_competitor_removed_wr1": 1.22,   # team lost its WR1
    "target_competitor_removed_wr2": 1.10,
    "target_competitor_removed_te1": 1.08,
    "target_competitor_added_wr1": 0.80,
    "target_competitor_added_wr2": 0.90,
    "backfield_competitor_removed": 1.30,
    "backfield_competitor_added": 0.74,
    "moved_to_starter": 1.55,
    "moved_to_backup": 0.45,
    "committee_backfield": 0.85,
    "bellcow_role": 1.20,
}

# Offensive-line and pass-catcher quality effects on EFFICIENCY.
# Inputs are z-scores of unit grade change year-over-year.
OL_PASSBLOCK_YPA_BETA = 0.16     # yards/attempt per +1 z of pass-block grade
OL_RUNBLOCK_YPC_BETA = 0.19      # yards/carry per +1 z of run-block grade
WR_CORPS_YPA_BETA = 0.21         # yards/attempt per +1 z of receiver corps
QB_QUALITY_YPT_BETA = 0.32       # yards/target per +1 z of QB quality

# ----------------------------------------------------------------------------
# 4. League baselines (used as the regression target for low-sample players)
# ----------------------------------------------------------------------------
# Every metric that gets blended MUST have an entry here. A missing entry
# means the "regress to league average" leg of the blend has nothing to
# regress toward; blend_value handles that by folding the weight back into
# the prior season, but an explicit value is always better.
LEAGUE_BASELINE: Dict[str, float] = {
    # team offense
    "plays_per_game": 63.0,
    "pace_sec_per_play": 27.6,
    "pass_rate": 0.575,
    "proe": 0.0,
    "rz_pass_rate": 0.545,
    "rz_trips_per_game": 3.4,
    "points_per_game": 22.4,
    "team_points_per_game": 22.4,
    "off_td_per_game": 2.28,
    "rush_td_share": 0.395,
    "rush_td_share_of_off_td": 0.395,
    "off_epa_per_play": 0.0,
    "team_pass_att_pg": 34.0,
    "team_rush_att_pg": 26.0,
    "team_pass_yds_pg": 228.0,
    "team_rush_yds_pg": 115.0,
    # player efficiency
    "ypa": 7.05,
    "ypc": 4.35,
    "yptarget": 7.55,
    "adot": 8.4,
    "catch_rate": 0.655,
    "sack_rate": 0.065,
    # team defense
    "def_epa_per_play_allowed": 0.0,
    "def_epa_per_rush_allowed": -0.08,
    "def_epa_per_pass_allowed": 0.06,
    "def_ypc_allowed": 4.35,
    "def_ypa_allowed": 7.05,
    "def_pass_yds_pg_allowed": 228.0,
    "def_rush_yds_pg_allowed": 115.0,
    "def_plays_pg_faced": 63.0,
    "def_pace_sec_allowed": 27.6,
    "def_rz_td_rate_allowed": 0.56,
    "def_rush_td_pg_allowed": 0.90,
    "def_pass_td_pg_allowed": 1.38,
    "def_pressure_rate": 0.34,
    "def_light_box_rate": 0.32,
    "def_vs_wr_out": 0.0,
    "def_vs_wr_slot": 0.0,
    "def_vs_te": 0.0,
    "def_vs_rb_pass": 0.0,
}

POSITION_BASELINE: Dict[str, Dict[str, float]] = {
    "QB": {"pass_att_share": 0.92, "ypa": 7.05, "rush_share": 0.09},
    "RB": {"rush_share": 0.42, "target_share": 0.105, "ypc": 4.30, "yptarget": 6.2, "adot": 0.4},
    "WR": {"target_share": 0.155, "yptarget": 8.1, "adot": 10.6, "rush_share": 0.01},
    "TE": {"target_share": 0.135, "yptarget": 7.3, "adot": 7.9, "rush_share": 0.0},
}

# ----------------------------------------------------------------------------
# 5. Game-environment mapping
# ----------------------------------------------------------------------------
# Implied team total -> expected offensive TDs. Linear fit on historical data:
#   E[off_TD] = TD_INTERCEPT + TD_SLOPE * implied_total
TD_INTERCEPT = -0.42
TD_SLOPE = 0.1205

# Game script: spread shifts pass rate and rush volume.
# pass_rate_adj = pass_rate + PASS_RATE_PER_POINT_SPREAD * (team_spread)
# team_spread is POSITIVE when the team is an underdog by that many points.
PASS_RATE_PER_POINT_SPREAD = 0.0068
RUSH_SHARE_PER_POINT_SPREAD = -0.0052     # favorites run more
PLAYS_PER_POINT_TOTAL = 0.22              # higher totals -> slightly more plays

# ----------------------------------------------------------------------------
# 6. Predictive distribution parameters
# ----------------------------------------------------------------------------
# Residual dispersion, expressed as coefficient of variation (sd / mean).
# Passing yards are near-normal; rushing and receiving are right-skewed and
# modeled with a Gamma. Receiving for low-volume players is zero-inflated.
DISPERSION_CV: Dict[str, float] = {
    "passing_yards": 0.265,
    "rushing_yards": 0.480,
    "receiving_yards": 0.585,
    "receptions": 0.400,
}
# CV shrinks as projected volume rises: cv = cv_base * (ref/mu)**CV_VOLUME_EXP
CV_VOLUME_EXP = 0.16
CV_REFERENCE_MU: Dict[str, float] = {
    "passing_yards": 240.0,
    "rushing_yards": 55.0,
    "receiving_yards": 50.0,
    "receptions": 4.0,
}
# Extra variance added when the projection itself is uncertain (roster change,
# rookie, injury return). Multiplies the CV.
CV_UNCERTAINTY_PENALTY = 0.35   # cv *= 1 + penalty * (1 - rho)

# Anytime-TD Poisson -> probability calibration.
# p = 1 - exp(-lambda * TD_POISSON_KAPPA), then optional isotonic recalibration.
TD_POISSON_KAPPA = 0.965

# ----------------------------------------------------------------------------
# 7. Betting / edge settings
# ----------------------------------------------------------------------------
MIN_EDGE_TO_FLAG = 0.035        # 3.5 percentage points of true-vs-implied edge
KELLY_FRACTION = 0.25           # quarter Kelly
CI_LEVELS = (0.50, 0.80)        # reported interval levels


@dataclass
class ModelConfig:
    """Runtime container so callers can override anything without editing this file."""
    stabilization_k: Dict[str, float] = field(default_factory=lambda: dict(STABILIZATION_K))
    continuity: Dict[str, float] = field(default_factory=lambda: dict(CONTINUITY_MULTIPLIERS))
    usage_shift: Dict[str, float] = field(default_factory=lambda: dict(USAGE_SHIFT))
    league: Dict[str, float] = field(default_factory=lambda: dict(LEAGUE_BASELINE))
    position: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(POSITION_BASELINE))
    dispersion_cv: Dict[str, float] = field(default_factory=lambda: dict(DISPERSION_CV))
    use_schedule_override: bool = USE_SCHEDULE_OVERRIDE
    td_kappa: float = TD_POISSON_KAPPA
    gamelog_decay: float = GAMELOG_DECAY_PER_WEEK

    def k(self, metric: str) -> float:
        return self.stabilization_k.get(metric, self.stabilization_k["_default"])


DEFAULT_CONFIG = ModelConfig()
