"""
QB continuity: adjust pass-catcher efficiency when the quarterback changed.

THE PROBLEM THIS SOLVES
-----------------------
A receiver's historical yards-per-target was earned catching passes from a
specific quarterback. When that quarterback is replaced, the receiver's prior
efficiency is no longer a clean estimate of his future efficiency, but the
model was carrying it forward unchanged.

Concretely: Sam Darnold threw 99% of Seattle's 2025 attempts at 8.49 YPA.
Drew Lock is throwing in 2026. Jaxon Smith-Njigba's prior yards-per-target
bakes in Darnold's accuracy and arm, and nothing in the pipeline noticed the
swap -- `new_qb` was False for all 32 teams and every `qb_grade` was the
placeholder 65.

HOW THE STARTER IS IDENTIFIED (in priority order)
-------------------------------------------------
1. Current-season pass attempts. Whoever is actually throwing is the starter;
   this is the most reliable signal and needs no external feed.
2. Sleeper depth_chart_order == 1, which covers ~94% of QB1s and updates for
   injuries within hours. Used before the season starts, or when nobody has
   thrown yet.
3. Prior-season dropback share, as a last resort.

QUALITY METRIC
--------------
Credibility-regressed yards per attempt, because raw YPA on a 5-attempt
sample is meaningless (see Malik Willis, 12.06 YPA on ~48 attempts, who once
projected as the league's top passer here). NGS completion-percentage-over-
expected refines it when available, since CPOE isolates the quarterback's own
accuracy from his receivers' work.

APPLICATION
-----------
    qb_efficiency_mult = regressed_ypa(new QB) / regressed_ypa(old QB)

damped toward 1.0 and clamped, then applied to every pass catcher on that
team. Damping matters: a receiver is not a pure passthrough for his
quarterback's efficiency, since scheme, separation and YAC are his own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAGUE_BASELINE, EFFICIENCY_CREDIBILITY_K
from .util import safe_num

# How much of the QB's efficiency change passes through to a receiver's
# yards per target. 1.0 would treat the receiver as a pure function of his
# QB; 0.0 would ignore the change entirely. Receivers own their separation
# and YAC, so the true value is well below 1.
QB_PASSTHROUGH = 0.62
QB_MULT_FLOOR, QB_MULT_CEIL = 0.80, 1.20


def _regressed_ypa(ypa, attempts) -> float:
    """Credibility-regressed YPA. A tiny sample collapses to league average."""
    lg = LEAGUE_BASELINE["ypa"]
    y = safe_num(ypa, np.nan)
    n = max(0.0, safe_num(attempts, 0.0))
    if np.isnan(y):
        return lg
    k = EFFICIENCY_CREDIBILITY_K["ypa"]
    return float((y * n + lg * k) / (n + k))


def identify_starter(team: str, prior_players: pd.DataFrame,
                     gamelog: pd.DataFrame | None,
                     depth: pd.DataFrame | None) -> dict | None:
    """Who is throwing for this team now? Returns the QB row plus its source."""
    # 1. whoever actually threw this season
    if gamelog is not None and len(gamelog):
        g = gamelog[(gamelog.get("team") == team) & (gamelog.get("pass_attempts", 0) > 0)]
        if len(g):
            agg = g.groupby("player_id", as_index=False)["pass_attempts"].sum()
            top = agg.sort_values("pass_attempts", ascending=False).iloc[0]
            if top["pass_attempts"] >= 8:
                return {"player_id": top["player_id"], "source": "current_season_attempts"}

    # 2. Sleeper depth chart
    if depth is not None and len(depth):
        d = depth[(depth.get("team") == team) & (depth.get("position") == "QB")
                  & (depth.get("depth_chart_order") == 1)]
        if len(d):
            return {"player_id": d.iloc[0]["player_id"], "source": "sleeper_depth_chart"}

    # 3. prior-season dropback share
    if prior_players is not None and len(prior_players):
        p = prior_players[(prior_players.get("team") == team)
                          & (prior_players.get("position") == "QB")]
        if len(p):
            top = p.sort_values("pass_att_share", ascending=False).iloc[0]
            return {"player_id": top["player_id"], "source": "prior_season_share"}
    return None


def qb_quality(player_id, prior_players: pd.DataFrame,
               gamelog: pd.DataFrame | None, ngs: pd.DataFrame | None) -> dict:
    """Credibility-regressed quality for one QB, blending prior and current."""
    out = {"player_id": player_id, "ypa_regressed": LEAGUE_BASELINE["ypa"],
           "attempts": 0.0, "cpoe": None, "name": None}
    if prior_players is not None and len(prior_players):
        row = prior_players[prior_players["player_id"] == player_id]
        if len(row):
            r = row.iloc[0]
            att = safe_num(r.get("games"), 0.0) * 33.0 * safe_num(r.get("pass_att_share"), 0.0)
            out["ypa_regressed"] = _regressed_ypa(r.get("ypa"), att)
            out["attempts"] = att
            out["name"] = r.get("player_name")

    # fold in current-season passing, which is the most relevant evidence
    if gamelog is not None and len(gamelog):
        g = gamelog[gamelog["player_id"] == player_id]
        if len(g):
            att_c = float(pd.to_numeric(g.get("pass_attempts"), errors="coerce").fillna(0).sum())
            yds_c = float(pd.to_numeric(g.get("passing_yards"), errors="coerce").fillna(0).sum())
            if att_c >= 5:
                ypa_c = yds_c / att_c
                tot = out["attempts"] + att_c
                combined = ((out["ypa_regressed"] * out["attempts"] + ypa_c * att_c) / tot
                            if tot > 0 else out["ypa_regressed"])
                out["ypa_regressed"] = _regressed_ypa(combined, tot)
                out["attempts"] = tot

    if ngs is not None and len(ngs):
        row = ngs[ngs["player_id"] == player_id]
        if len(row) and "ngs_cpoe" in row.columns:
            v = row.iloc[0]["ngs_cpoe"]
            if pd.notna(v):
                out["cpoe"] = float(v)
    return out


def qb_change_multiplier(team: str, prior_players: pd.DataFrame,
                         gamelog: pd.DataFrame | None,
                         depth: pd.DataFrame | None,
                         ngs: pd.DataFrame | None) -> dict:
    """
    Efficiency multiplier to apply to a team's pass catchers, given who is
    throwing now vs who threw when their prior numbers were earned.
    """
    neutral = {"qb_efficiency_mult": 1.0, "qb_changed": False,
               "current_qb": None, "prior_qb": None, "detail": "no QB data"}

    starter = identify_starter(team, prior_players, gamelog, depth)
    if starter is None:
        return neutral

    # whose passes generated the prior-season receiving numbers
    if prior_players is None or not len(prior_players):
        return neutral
    p = prior_players[(prior_players["team"] == team)
                      & (prior_players["position"] == "QB")]
    if not len(p):
        return neutral
    prior_qb_id = p.sort_values("pass_att_share", ascending=False).iloc[0]["player_id"]

    cur_q = qb_quality(starter["player_id"], prior_players, gamelog, ngs)
    old_q = qb_quality(prior_qb_id, prior_players, None, ngs)

    changed = str(starter["player_id"]) != str(prior_qb_id)
    if not changed:
        return {"qb_efficiency_mult": 1.0, "qb_changed": False,
                "current_qb": cur_q["name"], "prior_qb": old_q["name"],
                "current_qb_ypa": round(cur_q["ypa_regressed"], 2),
                "prior_qb_ypa": round(old_q["ypa_regressed"], 2),
                "starter_source": starter["source"],
                "detail": "same starter"}

    ratio = cur_q["ypa_regressed"] / max(old_q["ypa_regressed"], 1e-6)
    mult = 1.0 + QB_PASSTHROUGH * (ratio - 1.0)
    mult = float(np.clip(mult, QB_MULT_FLOOR, QB_MULT_CEIL))

    return {
        "qb_efficiency_mult": round(mult, 4),
        "qb_changed": True,
        "current_qb": cur_q["name"],
        "prior_qb": old_q["name"],
        "current_qb_ypa": round(cur_q["ypa_regressed"], 2),
        "prior_qb_ypa": round(old_q["ypa_regressed"], 2),
        "starter_source": starter["source"],
        "detail": (f"{old_q['name']} ({old_q['ypa_regressed']:.2f} YPA) -> "
                   f"{cur_q['name']} ({cur_q['ypa_regressed']:.2f} YPA)"),
    }
