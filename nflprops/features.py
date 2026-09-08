"""
Feature engineering.

Everything here converts blended raw metrics into the quantities the four
models actually consume. Features are grouped by what they act on:

  VOLUME features   -> how many plays / attempts / targets / carries
  SHARE features    -> what fraction of those the player gets
  EFFICIENCY feats  -> yards per opportunity
  SCORING features  -> touchdown probability

Feature reference (what it is, and which props it moves):

  def_epa_per_pass_allowed   Opponent EPA per dropback allowed. Raises QB YPA
                             and every receiver's yards per target. Multiplier:
                             1 + 0.55 * (opp_epa_pass - league) capped +/-12%.
  def_epa_per_rush_allowed   Same for the run. Drives RB yards per carry and
                             also team rush TD rate.
  def_ypa_allowed /          Raw allowed efficiency. Blended 50/50 with the EPA
  def_ypc_allowed            version to reduce noise on small samples.
  wr_cb_matchup_grade        (receiver grade - assigned corner grade)/10, where
                             the assignment depends on alignment (perimeter vs
                             slot) and whether the corner shadows. Moves yards
                             per target and, weakly, target share.
  rz_rush_rate / rz_pass_rate  Team play-calling inside the 20. Splits expected
                             team TDs into a rushing pool and a passing pool.
  inside5_rush_share         Player's share of goal-line carries. The single
                             largest ATD driver for running backs.
  proe                       Pass rate over expectation. Adds to pass volume
                             beyond what the game script alone implies.
  pace_sec_per_play          Seconds per play. Fewer seconds -> more plays ->
                             more of everything. Combined with the opponent's
                             pace, since game pace is a two-team negotiation.
  depth_chart_volatility     0-1 disagreement about the role. Does not move the
                             mean; widens the interval and cuts stake size.
  roster_change_multiplier   Product of usage tags. Moves shares directly.
  continuity_rho             Blend weight on the player's own history.
  game_script_index          Derived from the spread. Favorites run more and
                             throw less; underdogs do the reverse.
  team_total_implied         Market's expected points for the team. Anchors the
                             expected-TD pool for the ATD model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .util import safe_num
from .config import (
    ModelConfig, DEFAULT_CONFIG, LEAGUE_BASELINE,
    PASS_RATE_PER_POINT_SPREAD, RUSH_SHARE_PER_POINT_SPREAD,
    PLAYS_PER_POINT_TOTAL, TD_INTERCEPT, TD_SLOPE,
    OL_PASSBLOCK_YPA_BETA, OL_RUNBLOCK_YPC_BETA,
    WR_CORPS_YPA_BETA, QB_QUALITY_YPT_BETA,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def z(value: float, mean: float, sd: float) -> float:
    if sd is None or sd == 0 or pd.isna(sd):
        return 0.0
    if pd.isna(value):
        return 0.0
    return float((value - mean) / sd)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def grade_z(grade: float, mean: float = 65.0, sd: float = 8.0) -> float:
    return z(grade, mean, sd)


# ---------------------------------------------------------------------------
# Game environment
# ---------------------------------------------------------------------------
def game_script_features(env: dict) -> dict:
    """
    spread is POSITIVE when the team is an underdog.
    Returns the additive adjustments the volume model applies.
    """
    spread = safe_num(env.get("spread"), 0.0)
    total = safe_num(env.get("total"), 44.0)
    implied = float(env.get("implied_total", total / 2 - spread / 2))
    return {
        "spread": spread,
        "total": total,
        "team_total_implied": implied,
        "pass_rate_script_adj": PASS_RATE_PER_POINT_SPREAD * spread,
        "rush_share_script_adj": RUSH_SHARE_PER_POINT_SPREAD * spread,
        "plays_total_adj": PLAYS_PER_POINT_TOTAL * (total - 44.0),
        "game_script_index": clamp(spread / 10.0, -1.5, 1.5),
        "weather_penalty": _weather_penalty(env),
    }


def _weather_penalty(env: dict) -> float:
    """Multiplicative penalty to passing efficiency. 1.0 = neutral."""
    if bool(env.get("dome", False)):
        return 1.0
    wind = safe_num(env.get("wind_mph"), 0.0)
    precip = safe_num(env.get("precip_prob"), 0.0)
    pen = 1.0
    if wind > 12:
        pen -= 0.011 * (wind - 12)      # ~4% at 16 mph, ~9% at 20 mph
    pen -= 0.03 * max(0.0, precip - 0.4)
    return clamp(pen, 0.82, 1.0)


# ---------------------------------------------------------------------------
# Team volume
# ---------------------------------------------------------------------------
def team_volume_features(team_off: dict, opp_def: dict, env: dict,
                         cfg: ModelConfig = DEFAULT_CONFIG) -> dict:
    """
    Projected team plays, dropbacks, and carries for this game.

      plays = 0.5*(own_plays_pg + opp_plays_faced_pg)
              * pace_ratio + plays_total_adj
      pass_rate = blended_pass_rate + proe_component + script_adj
      dropbacks = plays * pass_rate
      carries   = plays * (1 - pass_rate) - qb_scramble_allowance
    """
    lg = cfg.league
    gs = game_script_features(env)

    own_plays = float(team_off.get("plays_per_game", lg["plays_per_game"]))
    opp_plays = float(opp_def.get("def_plays_pg_faced", lg["plays_per_game"]))
    base_plays = 0.5 * own_plays + 0.5 * opp_plays

    own_pace = float(team_off.get("pace_sec_per_play", lg["pace_sec_per_play"]))
    pace_ratio = clamp(lg["pace_sec_per_play"] / max(own_pace, 18.0), 0.88, 1.14)

    plays = base_plays * pace_ratio + gs["plays_total_adj"]
    plays = clamp(plays, 48.0, 82.0)

    pass_rate = float(team_off.get("pass_rate", lg["pass_rate"]))
    proe = safe_num(team_off.get("proe"), 0.0)
    pass_rate = clamp(pass_rate + 0.35 * proe + gs["pass_rate_script_adj"], 0.36, 0.74)

    dropbacks = plays * pass_rate
    sack_rate = safe_num(team_off.get("sack_rate"), 0.065)
    pass_attempts = dropbacks * (1 - sack_rate)
    carries = plays * (1 - pass_rate)

    return {
        "proj_team_plays": plays,
        "proj_team_pass_rate": pass_rate,
        "proj_team_dropbacks": dropbacks,
        "proj_team_pass_attempts": pass_attempts,
        "proj_team_carries": carries,
        "pace_ratio": pace_ratio,
        **gs,
    }


# ---------------------------------------------------------------------------
# Matchup adjustments
# ---------------------------------------------------------------------------
def pass_matchup_multiplier(opp_def: dict, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    lg = cfg.league
    epa = float(opp_def.get("def_epa_per_pass_allowed", lg["def_epa_per_pass_allowed"]))
    ypa = float(opp_def.get("def_ypa_allowed", lg["def_ypa_allowed"]))
    m_epa = 1.0 + 0.55 * (epa - lg["def_epa_per_pass_allowed"])
    m_ypa = ypa / lg["def_ypa_allowed"]
    return clamp(0.5 * m_epa + 0.5 * m_ypa, 0.86, 1.16)


def rush_matchup_multiplier(opp_def: dict, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    lg = cfg.league
    epa = float(opp_def.get("def_epa_per_rush_allowed", lg["def_epa_per_rush_allowed"]))
    ypc = float(opp_def.get("def_ypc_allowed", lg["def_ypc_allowed"]))
    m_epa = 1.0 + 0.60 * (epa - lg["def_epa_per_rush_allowed"])
    m_ypc = ypc / lg["def_ypc_allowed"]
    m = 0.45 * m_epa + 0.55 * m_ypc
    # light boxes help the run
    lb = float(opp_def.get("def_light_box_rate", np.nan))
    if not pd.isna(lb):
        m *= 1.0 + 0.18 * (lb - 0.32)
    return clamp(m, 0.84, 1.18)


def wr_cb_matchup(player: dict, opp_def: dict) -> dict:
    """
    Assign the receiver to a coverage defender and grade the mismatch.

      alignment: 'perimeter' | 'slot' | 'te' | 'rb'
      grade_gap = (player_grade - defender_grade) / 10
      yptarget multiplier = 1 + 0.055 * grade_gap  (capped +/-12%)
      target-share nudge  = 1 + 0.020 * grade_gap  (small; volume is scheme-driven)
    """
    pos = player.get("position", "WR")
    slot_rate = float(player.get("slot_rate", 0.35 if pos == "WR" else 0.6))
    rank_raw = player.get("depth_chart_rank_new", 2)
    depth_rank = 2 if rank_raw is None or (isinstance(rank_raw, float) and np.isnan(rank_raw)) else int(rank_raw)
    pgrade = safe_num(player.get("player_grade"), 68.0)

    if pos == "TE":
        alignment, dgrade = "te", float(opp_def.get("slot_cb_grade", 65.0))
        unit_epa = opp_def.get("def_vs_te", np.nan)
    elif pos == "RB":
        alignment, dgrade = "rb", 62.0
        unit_epa = opp_def.get("def_vs_rb_pass", np.nan)
    elif slot_rate >= 0.55:
        alignment, dgrade = "slot", float(opp_def.get("slot_cb_grade", 63.0))
        unit_epa = opp_def.get("def_vs_wr_slot", np.nan)
    else:
        shadow = bool(opp_def.get("shadow_cb", False))
        if depth_rank == 1 and shadow:
            dgrade = float(opp_def.get("cb1_grade", 68.0))
        elif depth_rank == 1:
            dgrade = 0.6 * float(opp_def.get("cb1_grade", 68.0)) + 0.4 * float(opp_def.get("cb2_grade", 62.0))
        else:
            dgrade = float(opp_def.get("cb2_grade", 62.0))
        alignment = "perimeter"
        unit_epa = opp_def.get("def_vs_wr_out", np.nan)

    gap = (pgrade - dgrade) / 10.0
    eff_mult = clamp(1.0 + 0.055 * gap, 0.88, 1.12)
    if not pd.isna(unit_epa):
        eff_mult *= clamp(1.0 + 0.45 * float(unit_epa), 0.90, 1.12)
    return {
        "alignment": alignment,
        "defender_grade": dgrade,
        "wr_cb_grade_gap": gap,
        "matchup_eff_mult": clamp(eff_mult, 0.84, 1.18),
        "matchup_share_mult": clamp(1.0 + 0.020 * gap, 0.95, 1.05),
    }


# ---------------------------------------------------------------------------
# Roster-driven efficiency modifiers
# ---------------------------------------------------------------------------
def unit_change_features(team_off_now: dict, team_off_prior: dict) -> dict:
    """
    Year-over-year unit changes converted to yards-per-opportunity deltas.
    Grades are z-scored against a 65 / 8 league distribution.
    """
    d_pb = grade_z(team_off_now.get("ol_passblock_grade", 65.0)) - \
           grade_z(team_off_prior.get("ol_passblock_grade", 65.0))
    d_rb = grade_z(team_off_now.get("ol_runblock_grade", 65.0)) - \
           grade_z(team_off_prior.get("ol_runblock_grade", 65.0))
    d_wr = grade_z(team_off_now.get("wr_corps_grade", 65.0)) - \
           grade_z(team_off_prior.get("wr_corps_grade", 65.0))
    d_qb = grade_z(team_off_now.get("qb_grade", 65.0)) - \
           grade_z(team_off_prior.get("qb_grade", 65.0))
    continuity = safe_num(team_off_now.get("ol_snaps_returning"), 0.7)

    return {
        "d_ol_passblock_z": d_pb,
        "d_ol_runblock_z": d_rb,
        "d_wr_corps_z": d_wr,
        "d_qb_z": d_qb,
        "ol_continuity": continuity,
        "ypa_unit_delta": OL_PASSBLOCK_YPA_BETA * d_pb + WR_CORPS_YPA_BETA * d_wr,
        "ypc_unit_delta": OL_RUNBLOCK_YPC_BETA * d_rb
                          + 0.25 * (continuity - 0.70),   # cohesion bonus/penalty
        "yptarget_unit_delta": QB_QUALITY_YPT_BETA * d_qb,
    }


# ---------------------------------------------------------------------------
# Scoring environment
# ---------------------------------------------------------------------------
def expected_team_touchdowns(env: dict, team_off: dict, opp_def: dict,
                             cfg: ModelConfig = DEFAULT_CONFIG) -> dict:
    """
    Convert the market's implied team total into an expected offensive TD pool,
    then split it into rushing and passing components.

      E[off_TD] = TD_INTERCEPT + TD_SLOPE * implied_total
      blended 70/30 with the team's own historical off_td_per_game rate scaled
      by the defense's TD-allowed rate.
      rush_share_of_td = blend(team rz_rush_rate tendency, historical share)
                         adjusted by the defense's rush vs pass TD split allowed.
    """
    lg = cfg.league
    implied = float(env.get("implied_total", lg["team_points_per_game"]))
    td_market = TD_INTERCEPT + TD_SLOPE * implied

    td_hist = float(team_off.get("off_td_per_game", lg["off_td_per_game"]))
    def_rz = safe_num(opp_def.get("def_rz_td_rate_allowed"), 0.56)
    def_adj = clamp(def_rz / 0.56, 0.85, 1.15)
    td_hist *= def_adj

    e_td = clamp(0.70 * td_market + 0.30 * td_hist, 0.85, 4.6)

    rz_pass_rate = float(team_off.get("rz_pass_rate", lg["rz_pass_rate"]))
    rush_share_hist = float(team_off.get("rush_td_share", lg["rush_td_share_of_off_td"]))
    rush_share_tend = clamp(1.0 - rz_pass_rate * 0.86, 0.22, 0.60)
    rush_share = 0.55 * rush_share_hist + 0.45 * rush_share_tend

    # defense that gives up rushing TDs specifically
    drtd = opp_def.get("def_rush_td_pg_allowed", np.nan)
    dptd = opp_def.get("def_pass_td_pg_allowed", np.nan)
    if not pd.isna(drtd) and not pd.isna(dptd) and (drtd + dptd) > 0:
        def_rush_frac = drtd / (drtd + dptd)
        rush_share = 0.72 * rush_share + 0.28 * def_rush_frac

    rush_share = clamp(rush_share, 0.18, 0.65)
    return {
        "e_team_off_td": e_td,
        "e_team_rush_td": e_td * rush_share,
        "e_team_rec_td": e_td * (1 - rush_share),
        "rush_td_share_of_off": rush_share,
        "team_total_implied": implied,
        "def_rz_td_adj": def_adj,
    }


# ---------------------------------------------------------------------------
# Assemble the full feature row for one player
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "proj_team_plays", "proj_team_pass_rate", "proj_team_dropbacks",
    "proj_team_pass_attempts", "proj_team_carries",
    "team_total_implied", "spread", "total", "game_script_index",
    "pace_ratio", "weather_penalty",
    "target_share", "air_yards_share", "rush_share", "route_participation",
    "snap_share", "rz_target_share", "rz_rush_share", "inside5_rush_share",
    "ypc", "yptarget", "adot", "catch_rate", "ypa", "pass_att_share",
    "pass_matchup_mult", "rush_matchup_mult", "matchup_eff_mult",
    "wr_cb_grade_gap", "d_ol_passblock_z", "d_ol_runblock_z",
    "d_wr_corps_z", "d_qb_z", "ol_continuity",
    "e_team_off_td", "e_team_rush_td", "e_team_rec_td", "rush_td_share_of_off",
    "continuity_rho", "roster_usage_mult", "role_uncertainty",
    "depth_chart_volatility",
]


def build_feature_row(player_blend: dict, team_off: dict, team_off_prior: dict,
                      opp_def: dict, env: dict, roster_ctx: dict,
                      cfg: ModelConfig = DEFAULT_CONFIG) -> dict:
    """One dict with every feature the four models need."""
    f = {}
    f.update(team_volume_features(team_off, opp_def, env, cfg))
    f.update(expected_team_touchdowns(env, team_off, opp_def, cfg))
    f.update(unit_change_features(team_off, team_off_prior))
    f.update(wr_cb_matchup({**player_blend, **roster_ctx}, opp_def))
    f["pass_matchup_mult"] = pass_matchup_multiplier(opp_def, cfg)
    f["rush_matchup_mult"] = rush_matchup_multiplier(opp_def, cfg)

    for k in ("target_share", "air_yards_share", "rush_share", "route_participation",
              "snap_share", "rz_target_share", "rz_rush_share", "inside5_rush_share",
              "ypc", "yptarget", "adot", "catch_rate", "ypa", "pass_att_share",
              "inside10_target_share"):
        if k in player_blend:
            f[k] = player_blend[k]

    f["continuity_rho"] = roster_ctx.get("rho", 0.9)
    f["roster_usage_mult"] = roster_ctx.get("usage_mult", 1.0)
    f["role_uncertainty"] = roster_ctx.get("role_uncertainty", 0.1)
    f["depth_chart_volatility"] = safe_num(roster_ctx.get("depth_chart_volatility"), 0.0)
    f["position"] = player_blend.get("position", roster_ctx.get("position", "WR"))
    return f


def feature_vector(f: dict) -> np.ndarray:
    return np.array([safe_num(f.get(k), 0.0) for k in FEATURE_ORDER])
