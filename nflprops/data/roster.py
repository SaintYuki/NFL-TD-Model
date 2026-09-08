"""
Roster-change logic.

Two distinct effects, deliberately kept separate:

1. CONTINUITY (rho) -- how much of the player's own prior-season profile is
   still predictive. This controls the BLEND, not the level. A WR who changed
   teams did not necessarily get worse; we are just less sure his old numbers
   describe him now, so he gets pulled toward the positional baseline.

       rho = RHO_BASE * prod(multiplier for each triggered flag), floored.

2. USAGE SHIFT (mu) -- a directional change in expected opportunity, applied
   after blending, then renormalized within the team so shares still sum to 1.

       share_adj = share_blend * prod(usage multipliers)
       share_final = share_adj / sum(share_adj over team) * team_share_total

Depth chart moves feed both: a WR3 promoted to WR1 gets lower rho (his old
snap profile is stale) and a usage multiplier above 1.

Efficiency-side roster effects (OL, QB, receiver corps) are handled in
features.py, not here, because they modify yards-per-opportunity rather than
opportunity itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..util import safe_num
from ..config import (
    ModelConfig, DEFAULT_CONFIG, RHO_BASE, RHO_FLOOR,
)

CONTINUITY_FLAGS = [
    "changed_team",
    "new_oc",
    "new_hc_offense",
    "scheme_change_major",
    "new_qb",
    "returning_from_ir",
    "rookie",
]

PASS_CATCHER_ONLY_FLAGS = {"new_qb"}


def continuity_rho(row: pd.Series, position: str,
                   cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """Compute rho for a single player from a roster_changes row."""
    if bool(row.get("rookie", False)):
        return 0.0

    rho = RHO_BASE
    for flag in CONTINUITY_FLAGS:
        if flag == "rookie":
            continue
        if not bool(row.get(flag, False)):
            continue
        if flag in PASS_CATCHER_ONLY_FLAGS and position not in ("WR", "TE", "RB"):
            continue
        rho *= cfg.continuity.get(flag, 1.0)

    # depth chart movement
    prior_rank = row.get("depth_chart_rank_prior", np.nan)
    new_rank = row.get("depth_chart_rank_new", np.nan)
    if pd.notna(prior_rank) and pd.notna(new_rank) and prior_rank != new_rank:
        key = ("depth_chart_promotion" if new_rank < prior_rank
               else "depth_chart_demotion")
        # scale the penalty by the size of the move
        steps = min(3, abs(int(new_rank) - int(prior_rank)))
        base = cfg.continuity.get(key, 1.0)
        rho *= base ** (0.55 + 0.45 * (steps - 1))

    # camp uncertainty widens the regression toward baseline
    vol = safe_num(row.get("depth_chart_volatility"), 0.0)
    rho *= (1.0 - 0.35 * np.clip(vol, 0.0, 1.0))

    return float(np.clip(rho, RHO_FLOOR, 1.0))


def usage_multiplier(row: pd.Series, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """Product of the usage-shift tags attached to a player."""
    tags = row.get("usage_shift_tags", "") or ""
    if isinstance(tags, float) and np.isnan(tags):
        return 1.0
    mult = 1.0
    for tag in [t.strip() for t in str(tags).split(";") if t.strip()]:
        mult *= cfg.usage_shift.get(tag, 1.0)

    # implicit depth-chart move if no explicit tag was given
    if not tags:
        pr, nr = row.get("depth_chart_rank_prior"), row.get("depth_chart_rank_new")
        if pd.notna(pr) and pd.notna(nr):
            if nr == 1 and pr > 1:
                mult *= cfg.usage_shift["moved_to_starter"] ** 0.6
            elif pr == 1 and nr > 1:
                mult *= cfg.usage_shift["moved_to_backup"] ** 0.6
    return float(mult)


def uncertainty_score(row: pd.Series, rho: float) -> float:
    """
    0..1 measure of how unsure we are about this player's role.
    Widens predictive intervals and shrinks bet sizing.
    """
    u = 1.0 - rho
    u += 0.25 * safe_num(row.get("depth_chart_volatility"), 0.0)
    if bool(row.get("rookie", False)):
        u += 0.20
    if bool(row.get("returning_from_ir", False)):
        u += 0.10
    return float(np.clip(u, 0.0, 1.0))


def build_roster_context(roster_changes: pd.DataFrame,
                         player_meta: pd.DataFrame,
                         cfg: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Returns one row per player with: rho, usage_mult, uncertainty, team, position.
    player_meta needs player_id, position, and team (current).
    """
    meta = player_meta[["player_id", "position", "team"]].drop_duplicates("player_id")
    rc = roster_changes.copy()
    df = meta.merge(rc, on="player_id", how="left", suffixes=("", "_rc"))

    for c in CONTINUITY_FLAGS:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)
    for c in ["depth_chart_rank_prior", "depth_chart_rank_new",
              "depth_chart_volatility", "manual_share_override"]:
        if c not in df.columns:
            df[c] = np.nan
    if "usage_shift_tags" not in df.columns:
        df["usage_shift_tags"] = ""
    df["usage_shift_tags"] = df["usage_shift_tags"].fillna("")

    df["rho"] = [continuity_rho(r, r["position"], cfg) for _, r in df.iterrows()]
    df["usage_mult"] = [usage_multiplier(r, cfg) for _, r in df.iterrows()]
    df["role_uncertainty"] = [uncertainty_score(r, r["rho"]) for _, r in df.iterrows()]
    keep = ["player_id", "position", "team", "rho", "usage_mult", "role_uncertainty",
            "changed_team", "rookie", "new_oc", "new_qb", "scheme_change_major",
            "depth_chart_rank_new", "depth_chart_volatility", "manual_share_override",
            "usage_shift_tags"]
    return df[[c for c in keep if c in df.columns]]


def renormalize_shares(df: pd.DataFrame, share_col: str,
                       team_col: str = "team",
                       target_total: float = 1.0,
                       eligible_mask: pd.Series | None = None) -> pd.Series:
    """
    After multiplying shares by usage multipliers, force each team's shares back
    to `target_total`. Without this step, tagging three players on one team as
    'moved_to_starter' would silently create a 140%-share offense.
    """
    s = df[share_col].astype(float).fillna(0.0).clip(lower=0)
    mask = pd.Series(True, index=df.index) if eligible_mask is None else eligible_mask
    out = s.copy()
    for team, idx in df.groupby(team_col).groups.items():
        idx = [i for i in idx if mask.loc[i]]
        if not idx:
            continue
        tot = s.loc[idx].sum()
        if tot > 0:
            out.loc[idx] = s.loc[idx] / tot * target_total
    return out


def apply_roster_adjustments(usage: pd.DataFrame,
                             ctx: pd.DataFrame,
                             share_cols: tuple[str, ...] = (
                                 "target_share", "rush_share", "rz_rush_share",
                                 "rz_target_share", "inside5_rush_share",
                                 "air_yards_share"),
                             cfg: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Apply usage multipliers to every share column, honor manual overrides, then
    renormalize per team so shares remain a valid partition of team opportunity.
    """
    df = usage.merge(ctx[["player_id", "usage_mult", "manual_share_override",
                          "role_uncertainty", "rho"]],
                     on="player_id", how="left")
    df["usage_mult"] = df["usage_mult"].fillna(1.0)

    for col in share_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(float) * df["usage_mult"]
        if col == "target_share" and "manual_share_override" in df.columns:
            ov = df["manual_share_override"]
            df.loc[ov.notna(), col] = ov[ov.notna()]
        # QBs do not consume target share; RB/WR/TE partition it
        if col in ("target_share", "rz_target_share", "air_yards_share"):
            elig = df["position"].isin(["RB", "WR", "TE"])
        elif col in ("rush_share", "rz_rush_share", "inside5_rush_share"):
            elig = df["position"].isin(["RB", "WR", "QB"])
        else:
            elig = pd.Series(True, index=df.index)
        df[col] = renormalize_shares(df, col, eligible_mask=elig)
    return df
