"""
Advanced-data ingestion: NGS, FTN charting, PFR advanced stats, Sleeper.

This module exists because an earlier version of this project wrongly assumed
most of the spec's inputs required a paid feed (PFF/SIS). They do not. Nearly
everything is free:

  NGS (nflverse)          separation, cushion, YAC above expectation, time to
                          throw, aggressiveness, CPOE, rush yards over
                          expected, 8+ box rate. Runs through the CURRENT
                          season, weekly.
  FTN charting (nflverse) blitzers, pass rushers, box count, play action,
                          screen, RPO, no-huddle, motion, contested ball,
                          drops, QB out of pocket, and read_thrown -- which
                          is a genuine first-read indicator.
  PFR advstats            per-PLAYER yards before/after contact, broken
                          tackles, drop rate, and -- most valuable -- a
                          per-DEFENDER table with pressures, hurries, QB
                          knockdowns, blitzes, plus completion % / yards per
                          target / passer rating ALLOWED in coverage.
  Sleeper API             depth chart order and injury status, free and
                          without an API key.

WHAT EACH ONE UNLOCKS
---------------------
The per-defender PFR table is what makes a real matchup engine possible: it
supports "this receiver vs the specific corner who covers his alignment"
rather than a single team-level multiplier.

NGS separates EFFICIENCY from OPPORTUNITY properly. Rush yards over expected
per attempt is a back's own contribution with blocking and box count already
removed -- which is exactly the distinction this model kept blurring.

Sleeper's depth_chart_order is the direct fix for the share-dilution bug that
has recurred through this build: the model has been inferring "who is the
starter" from noisy prior-season usage, when a maintained depth chart states
it outright.

NETWORK NOTE
------------
nflverse reads are GitHub-hosted and verified working. The Sleeper client is
written to spec and is NOT live-tested here (the sandbox cannot reach
api.sleeper.app); run `python scripts/fetch_advanced.py --selftest-sleeper`
on a normal network before trusting it.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
NGS_URL = NFLVERSE + "nextgen_stats/ngs_{kind}.parquet"
FTN_URL = NFLVERSE + "ftn_charting/ftn_charting_{season}.parquet"
PFR_URL = NFLVERSE + "pfr_advstats/advstats_season_{kind}.parquet"

SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"

_cache: dict = {}


def _get(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


# ---------------------------------------------------------------------------
# NGS
# ---------------------------------------------------------------------------
def ngs(kind: str, season: int, through_week: int | None = None) -> pd.DataFrame:
    """
    kind: 'passing' | 'receiving' | 'rushing'.
    NGS ships week==0 rows as season-to-date aggregates; those are the most
    useful form here, so they are preferred when present.
    """
    df = _get(("ngs", kind), lambda: pd.read_parquet(NGS_URL.format(kind=kind)))
    df = df[df["season"] == season]
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    if through_week is not None:
        wk = df[(df["week"] > 0) & (df["week"] < through_week)]
        if len(wk):
            return wk
    agg = df[df["week"] == 0]
    return agg if len(agg) else df


def ngs_player_summary(season: int, through_week: int | None = None) -> pd.DataFrame:
    """One row per player with the NGS metrics the model consumes."""
    frames = []

    p = ngs("passing", season, through_week)
    if len(p):
        g = p.groupby("player_gsis_id", as_index=False).agg(
            ngs_time_to_throw=("avg_time_to_throw", "mean"),
            ngs_aggressiveness=("aggressiveness", "mean"),
            ngs_cpoe=("completion_percentage_above_expectation", "mean"),
            ngs_intended_air_yards=("avg_intended_air_yards", "mean"),
            ngs_air_yards_to_sticks=("avg_air_yards_to_sticks", "mean"))
        frames.append(g)

    r = ngs("receiving", season, through_week)
    if len(r):
        g = r.groupby("player_gsis_id", as_index=False).agg(
            ngs_separation=("avg_separation", "mean"),
            ngs_cushion=("avg_cushion", "mean"),
            ngs_yac_above_expected=("avg_yac_above_expectation", "mean"),
            ngs_intended_air_yards_rec=("avg_intended_air_yards", "mean"),
            ngs_air_yards_share=("percent_share_of_intended_air_yards", "mean"))
        frames.append(g)

    ru = ngs("rushing", season, through_week)
    if len(ru):
        g = ru.groupby("player_gsis_id", as_index=False).agg(
            ngs_ryoe_per_att=("rush_yards_over_expected_per_att", "mean"),
            ngs_eight_box_rate=("percent_attempts_gte_eight_defenders", "mean"),
            ngs_time_to_los=("avg_time_to_los", "mean"),
            ngs_rush_efficiency=("efficiency", "mean"))
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=["player_id"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="player_gsis_id", how="outer")
    return out.rename(columns={"player_gsis_id": "player_id"})


# ---------------------------------------------------------------------------
# FTN charting -> team-level scheme + defensive pressure tendencies
# ---------------------------------------------------------------------------
def ftn(season: int, through_week: int | None = None) -> pd.DataFrame:
    try:
        df = _get(("ftn", season), lambda: pd.read_parquet(FTN_URL.format(season=season)))
    except Exception:
        return pd.DataFrame()
    if through_week is not None and "week" in df.columns:
        df = df[df["week"] < through_week]
    return df


def ftn_team_tendencies(season: int, pbp: pd.DataFrame,
                        through_week: int | None = None) -> pd.DataFrame:
    """
    Join FTN charting to play-by-play to get OFFENSE scheme rates and DEFENSE
    pressure tendencies per team. FTN carries no team column of its own, so
    the join to pbp (which has posteam/defteam) is required.
    """
    f = ftn(season, through_week)
    if not len(f) or not len(pbp):
        return pd.DataFrame()
    keep = ["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass",
            "is_rpo", "is_no_huddle", "is_motion", "n_blitzers", "n_pass_rushers",
            "n_defense_box", "is_qb_out_of_pocket", "read_thrown",
            "is_contested_ball", "is_drop"]
    f = f[[c for c in keep if c in f.columns]]
    p = pbp[["game_id", "play_id", "posteam", "defteam", "pass", "rush"]].copy()
    m = p.merge(f, left_on=["game_id", "play_id"],
                right_on=["nflverse_game_id", "nflverse_play_id"], how="inner")
    if not len(m):
        return pd.DataFrame()

    for c in ("is_play_action", "is_screen_pass", "is_rpo", "is_no_huddle",
              "is_motion", "is_qb_out_of_pocket"):
        if c in m.columns:
            m[c] = m[c].astype("float")

    off = m[m["pass"] == 1].groupby("posteam", as_index=False).agg(
        off_play_action_rate=("is_play_action", "mean"),
        off_screen_rate=("is_screen_pass", "mean"),
        off_rpo_rate=("is_rpo", "mean"),
        off_no_huddle_rate=("is_no_huddle", "mean"),
        off_motion_rate=("is_motion", "mean"),
        off_out_of_pocket_rate=("is_qb_out_of_pocket", "mean"),
    ).rename(columns={"posteam": "team"})

    d = m[m["pass"] == 1].groupby("defteam", as_index=False).agg(
        def_blitz_rate=("n_blitzers", lambda s: float((pd.to_numeric(s, errors="coerce") >= 1).mean())),
        def_avg_pass_rushers=("n_pass_rushers", "mean"),
    ).rename(columns={"defteam": "team"})
    dbox = m[m["rush"] == 1].groupby("defteam", as_index=False).agg(
        def_avg_box=("n_defense_box", "mean"),
        def_heavy_box_rate=("n_defense_box", lambda s: float((pd.to_numeric(s, errors="coerce") >= 8).mean())),
    ).rename(columns={"defteam": "team"})
    d = d.merge(dbox, on="team", how="outer")
    return off.merge(d, on="team", how="outer")


# ---------------------------------------------------------------------------
# PFR advanced stats
# ---------------------------------------------------------------------------
def pfr(kind: str, season: int) -> pd.DataFrame:
    try:
        df = _get(("pfr", kind), lambda: pd.read_parquet(PFR_URL.format(kind=kind)))
    except Exception:
        return pd.DataFrame()
    return df[df["season"] == season] if "season" in df.columns else df


def pfr_player_efficiency(season: int, ids: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player contact-adjusted efficiency, keyed to gsis ids.
    ids: crosswalk with gsis_id + pfr_id.
    """
    frames = []
    ru = pfr("rush", season)
    if len(ru):
        ru = ru[["pfr_id", "ybc_att", "yac_att", "brk_tkl", "att_br", "att"]].rename(
            columns={"ybc_att": "rush_ybc_att", "yac_att": "rush_yac_att",
                     "brk_tkl": "rush_broken_tackles", "att_br": "rush_att_per_broken",
                     "att": "pfr_rush_att"})
        frames.append(ru)
    rec = pfr("rec", season)
    if len(rec):
        rec = rec[["pfr_id", "ybc_r", "yac_r", "adot", "brk_tkl", "drop_percent", "tgt"]].rename(
            columns={"ybc_r": "rec_ybc_per_rec", "yac_r": "rec_yac_per_rec",
                     "adot": "pfr_adot", "brk_tkl": "rec_broken_tackles",
                     "drop_percent": "rec_drop_pct", "tgt": "pfr_targets"})
        frames.append(rec)
    if not frames:
        return pd.DataFrame(columns=["player_id"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="pfr_id", how="outer")
    out = out.merge(ids[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id"),
                    on="pfr_id", how="left")
    return out.rename(columns={"gsis_id": "player_id"}).dropna(subset=["player_id"])


def pfr_defense_units(season: int) -> pd.DataFrame:
    """
    Team defensive pressure and coverage, aggregated from the per-defender
    table. Also surfaces the top coverage defenders per team, which is what
    a receiver-vs-corner matchup needs.
    """
    d = pfr("def", season)
    if not len(d):
        return pd.DataFrame()
    d = d.copy()
    for c in ("prss", "hrry", "qbkd", "bltz", "tgt", "cmp_percent", "yds_tgt", "rat", "dadot"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    team = d.groupby("tm", as_index=False).agg(
        def_pressures=("prss", "sum"),
        def_hurries=("hrry", "sum"),
        def_qb_knockdowns=("qbkd", "sum"),
        def_blitzes=("bltz", "sum"),
    ).rename(columns={"tm": "team"})

    # coverage quality: target-weighted, DBs and LBs only
    cov = d[d["pos"].isin(["CB", "S", "FS", "SS", "DB", "LB", "ILB", "OLB"])].copy()
    cov = cov[cov["tgt"].fillna(0) >= 10]
    if len(cov):
        cov["w"] = cov["tgt"]
        agg = cov.groupby("tm").apply(
            lambda g: pd.Series({
                "def_cov_yds_per_target": np.average(g["yds_tgt"].fillna(7.0), weights=g["w"]),
                "def_cov_cmp_pct": np.average(g["cmp_percent"].fillna(65.0), weights=g["w"]),
                "def_cov_rating": np.average(g["rat"].fillna(90.0), weights=g["w"]),
            }), include_groups=False).reset_index().rename(columns={"tm": "team"})
        team = team.merge(agg, on="team", how="outer")
    return team


def pfr_top_coverage_defenders(season: int, top_n: int = 3) -> pd.DataFrame:
    """
    Per team, the most-targeted coverage defenders with their allowed numbers.
    This replaces the placeholder cb1_grade/cb2_grade constants with real,
    per-defender coverage performance.
    """
    d = pfr("def", season)
    if not len(d):
        return pd.DataFrame()
    d = d.copy()
    for c in ("tgt", "cmp_percent", "yds_tgt", "rat", "dadot"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    cb = d[d["pos"].isin(["CB", "DB"])].copy()
    cb = cb[cb["tgt"].fillna(0) >= 8]
    if not len(cb):
        return pd.DataFrame()
    cb = cb.sort_values(["tm", "tgt"], ascending=[True, False])
    cb["rank"] = cb.groupby("tm").cumcount() + 1
    cb = cb[cb["rank"] <= top_n]
    return cb[["tm", "player", "pos", "rank", "tgt", "cmp_percent",
               "yds_tgt", "rat", "dadot"]].rename(columns={"tm": "team"})


# ---------------------------------------------------------------------------
# Sleeper: depth chart + injury status
# ---------------------------------------------------------------------------
def sleeper_players(timeout: float = 30.0) -> pd.DataFrame:
    """
    Sleeper's full NFL player blob: free, no API key. It is a ~5MB JSON
    document, so it is fetched once and reused.

    The two fields that matter here:
      depth_chart_order     1 = starter. Directly states who the starter is,
                            instead of the model inferring it from noisy
                            prior-season usage -- which is the root of the
                            share-dilution bug seen repeatedly in this build.
      injury_status         Out / Doubtful / Questionable / IR / etc.
    """
    import requests
    resp = requests.get(SLEEPER_PLAYERS, timeout=timeout)
    resp.raise_for_status()
    blob = resp.json()
    rows = []
    for pid, p in blob.items():
        if not isinstance(p, dict):
            continue
        rows.append({
            "sleeper_id": pid,
            "player_id": p.get("gsis_id"),
            "player_name": p.get("full_name"),
            "team": p.get("team"),
            "position": p.get("position"),
            "depth_chart_order": p.get("depth_chart_order"),
            "depth_chart_position": p.get("depth_chart_position"),
            "injury_status": p.get("injury_status"),
            "injury_body_part": p.get("injury_body_part"),
            "status": p.get("status"),
            "years_exp": p.get("years_exp"),
        })
    df = pd.DataFrame(rows)
    return df[df["player_id"].notna()].copy()


# injury status -> availability multiplier applied to projected opportunity
INJURY_MULTIPLIER = {
    "Out": 0.0,
    "IR": 0.0,
    "PUP": 0.0,
    "Suspended": 0.0,
    "Doubtful": 0.15,
    "Questionable": 0.85,
    None: 1.0,
}


def injury_multiplier(status) -> float:
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return 1.0
    return INJURY_MULTIPLIER.get(str(status).strip(), 1.0)
