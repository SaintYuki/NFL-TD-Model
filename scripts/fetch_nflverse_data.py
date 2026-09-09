"""
Pulls real NFL data from nflverse (the open, free, no-API-key data project the
whole public NFL analytics community builds on) and writes it into the exact
CSV shapes nflprops/schema.py expects.

Sources, all free, no API key:
    weekly player stats   github.com/nflverse/nflverse-data  (via nfl_data_py)
    play-by-play          github.com/nflverse/nflverse-data  (via nfl_data_py)
    snap counts           github.com/nflverse/nflverse-data  (via nfl_data_py)
    id crosswalk          github.com/nflverse/nflverse-data  (via nfl_data_py)
    schedules/game lines  github.com/nflverse/nfldata        (raw CSV)

What this does NOT cover, because no free source has it:
    - OL/WR corps/QB grades (ol_passblock_grade, wr_corps_grade, qb_grade, ...)
      These need a paid grading service (PFF or similar) or your own manual
      1-100 ratings. Left as league-average placeholders (65) until supplied.
    - Coordinator names, scheme labels, depth chart ranks, "new OC" flags.
      Nobody publishes this in a clean feed; it changes by press release and
      beat-reporter tweets. roster_changes.csv is written with `changed_team`
      and `rookie` computed for real from roster data, but the OC/scheme/
      depth-chart columns are left at defaults for you to hand-edit -- that
      file is meant to be a spreadsheet you touch up once a week in August
      and lightly during the season, not something to scrape.
    - Sportsbook prop lines. Deliberately out of scope for this pass --
      see prop_lines.csv, left empty. Wiring that up is next.

Usage:
    # prior season tables (run once per year, after the season ends)
    python scripts/fetch_nflverse_data.py prior --season 2025 --root data

    # current season game logs + schedule (run weekly, in-season)
    python scripts/fetch_nflverse_data.py current --season 2026 --through-week 4 --root data
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflprops import schema  # noqa: E402

# All four data sources are read directly from their nflverse-data (or
# dynastyprocess) GitHub release URLs, with no wrapper library in between.
#
# This project originally used the `nfl_data_py` package for pbp, snap
# counts, rosters, and the id crosswalk. That package was archived by its
# maintainer in September 2025 -- it still runs, but nobody is fixing it if
# nflverse changes another file layout the way they changed player_stats in
# August 2025 (see the note on PBP_URL below for exactly that kind of
# breakage). Reading the release URLs directly means one less unmaintained
# layer between this pipeline and the day nflverse changes something else.
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{0}.parquet"
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{0}.parquet"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{0}.parquet"
IDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"

GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
GRADE_PLACEHOLDER = 65.0  # league-average stand-in until real unit grades are supplied

pd.set_option("mode.chained_assignment", None)


# ---------------------------------------------------------------------------
# Raw pulls (each cached in-process per season so prior+current sharing a
# season during testing does not double-fetch)
# ---------------------------------------------------------------------------
_cache: dict = {}


NEW_STATS_PLAYER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{0}.parquet"
)

# nflverse deprecated the old per-year player_stats_YYYY.parquet file on
# 2025-08-01 in favor of stats_player_week_YYYY.parquet, which uses slightly
# different column names (team instead of recent_team, sacks_suffered instead
# of sacks, passing_interceptions instead of interceptions) and now includes
# defense/kicking/punting stats we don't need. nfl_data_py's import_weekly_data
# still points at the old, now-frozen file (nfl_data_py itself was archived by
# its maintainer in September 2025), so it silently stops updating past the
# 2024 season. We read the new file directly and rename it back to the shape
# the rest of this script was written against.
_WEEKLY_RENAME = {"team": "recent_team", "sacks_suffered": "sacks",
                  "passing_interceptions": "interceptions"}


def _weekly(season: int) -> pd.DataFrame:
    key = ("weekly", season)
    if key not in _cache:
        df = pd.read_parquet(NEW_STATS_PLAYER_URL.format(season))
        df = df.rename(columns=_WEEKLY_RENAME)
        # keep skill-position offense only; the new file also carries
        # defense/kicking/punting rows this pipeline has no use for
        df = df[df["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
        _cache[key] = df
    return _cache[key]


def _pbp(season: int) -> pd.DataFrame:
    key = ("pbp", season)
    if key not in _cache:
        df = pd.read_parquet(PBP_URL.format(season))
        df["season"] = season
        cols = df.select_dtypes(include=["float64"]).columns
        df[cols] = df[cols].astype("float32")
        _cache[key] = df[df["season_type"] == "REG"].copy()
    return _cache[key]


def _snaps(season: int) -> pd.DataFrame:
    key = ("snaps", season)
    if key not in _cache:
        _cache[key] = pd.read_parquet(SNAP_COUNTS_URL.format(season))
    return _cache[key]


def _ids() -> pd.DataFrame:
    if "ids" not in _cache:
        df = pd.read_csv(IDS_URL)
        # keep the two id columns this pipeline actually joins on
        _cache["ids"] = df[[c for c in ("gsis_id", "pfr_id") if c in df.columns]].copy()
    return _cache["ids"]


def _rosters(season: int) -> pd.DataFrame:
    key = ("roster", season)
    if key not in _cache:
        df = pd.read_parquet(ROSTER_URL.format(season))
        df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        _cache[key] = df
    return _cache[key]


def _games() -> pd.DataFrame:
    if "games" not in _cache:
        _cache["games"] = pd.read_csv(GAMES_CSV_URL)
    return _cache["games"]


def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where((den is not None) & (den > 0), num / den.replace(0, np.nan), np.nan)


# ---------------------------------------------------------------------------
# Team-week volume, derived from the player weekly file itself (summing every
# player's row gives the team total without a separate pbp pass)
# ---------------------------------------------------------------------------
def team_week_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    g = weekly.groupby(["season", "week", "recent_team"], as_index=False).agg(
        team_targets=("targets", "sum"),
        team_air_yards=("receiving_air_yards", "sum"),
        team_carries=("carries", "sum"),
        team_pass_attempts=("attempts", "sum"),
    )
    return g.rename(columns={"recent_team": "team"})


# ---------------------------------------------------------------------------
# Red zone / goal line shares from play-by-play
# ---------------------------------------------------------------------------
def redzone_player_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-week: rz/inside-10/inside-5 targets and carries."""
    p = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    p["is_rz"] = p["yardline_100"] <= 20
    p["is_i10"] = p["yardline_100"] <= 10
    p["is_i5"] = p["yardline_100"] <= 5

    tgt = p[p["play_type"] == "pass"].groupby(
        ["season", "week", "posteam", "receiver_player_id"], as_index=False
    ).agg(rz_targets=("is_rz", "sum"), inside10_targets=("is_i10", "sum"))
    tgt = tgt.rename(columns={"posteam": "team", "receiver_player_id": "player_id"})

    rsh = p[p["play_type"] == "run"].groupby(
        ["season", "week", "posteam", "rusher_player_id"], as_index=False
    ).agg(rz_carries=("is_rz", "sum"), inside5_carries=("is_i5", "sum"))
    rsh = rsh.rename(columns={"posteam": "team", "rusher_player_id": "player_id"})

    return tgt, rsh


def redzone_team_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-week: rz trips faced/created, rz play calling, inside-5 volume."""
    p = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    p["is_rz"] = p["yardline_100"] <= 20
    p["is_i5"] = p["yardline_100"] <= 5
    g = p.groupby(["season", "week", "posteam"], as_index=False).agg(
        team_rz_targets=("is_rz", lambda s: int((s & (p.loc[s.index, "play_type"] == "pass")).sum())),
    )
    # simpler: recompute directly rather than lambda-on-slice (clearer and correct)
    rz = p[p["is_rz"]]
    team_rz_pass = rz[rz.play_type == "pass"].groupby(
        ["season", "week", "posteam"], as_index=False).size().rename(
        columns={"size": "team_rz_targets", "posteam": "team"})
    team_rz_rush = rz[rz.play_type == "run"].groupby(
        ["season", "week", "posteam"], as_index=False).size().rename(
        columns={"size": "team_rz_carries", "posteam": "team"})
    i5 = p[p["is_i5"] & (p.play_type == "run")].groupby(
        ["season", "week", "posteam"], as_index=False).size().rename(
        columns={"size": "team_inside5_carries", "posteam": "team"})
    out = team_rz_pass.merge(team_rz_rush, on=["season", "week", "team"], how="outer") \
                       .merge(i5, on=["season", "week", "team"], how="outer")
    return out.fillna(0)


# ---------------------------------------------------------------------------
# Team offense / defense season and week aggregates from play-by-play
# ---------------------------------------------------------------------------
def team_offense_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["play_type"].isin(["pass", "run", "qb_kneel", "qb_spike"])].copy()
    g = p.groupby(["season", "week", "posteam"], as_index=False).agg(
        plays=("play_id", "count"),
        dropbacks=("qb_dropback", "sum"),
        epa_per_play=("epa", "mean"),
        pass_oe=("pass_oe", "mean") if "pass_oe" in p.columns else ("epa", "mean"),
    )
    g = g.rename(columns={"posteam": "team"})
    scores = pbp.groupby(["season", "week", "posteam"], as_index=False).agg(
        off_td=("touchdown", "sum"),
        rush_td=("rush_touchdown", "sum"),
    ).rename(columns={"posteam": "team"})
    # points require the schedule (drive-level scoring is messier from raw pbp);
    # approximate via touchdowns*7 + a league-average field goal rate, refined
    # against the schedule file's actual final score downstream if needed.
    out = g.merge(scores, on=["season", "week", "team"], how="left")
    out["points"] = out["off_td"].fillna(0) * 7.0
    out["proe"] = out.pop("pass_oe") / 100.0 if out["pass_oe"].abs().max() > 1.5 else out.pop("pass_oe")
    return out


def team_defense_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pass_p = p[p.play_type == "pass"]
    run_p = p[p.play_type == "run"]

    epa_allowed = p.groupby(["season", "week", "defteam"], as_index=False).agg(
        epa_per_play_allowed=("epa", "mean"), plays_faced=("play_id", "count"))
    epa_pass = pass_p.groupby(["season", "week", "defteam"], as_index=False).agg(
        epa_per_pass_allowed=("epa", "mean"),
        pass_yds_allowed=("yards_gained", "sum"),
        pass_att_faced=("play_id", "count"))
    epa_rush = run_p.groupby(["season", "week", "defteam"], as_index=False).agg(
        epa_per_rush_allowed=("epa", "mean"),
        rush_yds_allowed=("yards_gained", "sum"),
        carries_faced=("play_id", "count"))
    tds = pbp.groupby(["season", "week", "defteam"], as_index=False).agg(
        pass_td_allowed=("pass_touchdown", "sum"),
        rush_td_allowed=("rush_touchdown", "sum"))
    rz = p[p.yardline_100 <= 20]
    rz_trips = rz.groupby(["season", "week", "defteam"], as_index=False).agg(
        rz_trips_faced=("play_id", "nunique"))
    rz_td = pbp[(pbp.yardline_100 <= 20) & (pbp.touchdown == 1)].groupby(
        ["season", "week", "defteam"], as_index=False).agg(rz_td_allowed=("play_id", "count"))

    out = epa_allowed
    for other in (epa_pass, epa_rush, tds, rz_trips, rz_td):
        out = out.merge(other, on=["season", "week", "defteam"], how="left")
    return out.rename(columns={"defteam": "team"}).fillna(0)


# ---------------------------------------------------------------------------
# Builders matching schema.py exactly
# ---------------------------------------------------------------------------
def build_player_season(season: int) -> pd.DataFrame:
    wk = _weekly(season)
    wk = wk[wk["season_type"] == "REG"] if "season_type" in wk.columns else wk
    tt = team_week_totals(wk)
    wk = wk.merge(tt, left_on=["season", "week", "recent_team"],
                 right_on=["season", "week", "team"], how="left")

    snaps = _snaps(season)
    ids = _ids()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    snap_season = snaps.groupby("gsis_id", as_index=False).agg(
        snap_share=("offense_pct", "mean"))

    pbp = _pbp(season)
    rz_tgt, rz_rsh = redzone_player_week(pbp)
    team_rz_week_lookup = redzone_team_week(pbp)  # season, week, team -> rz volume

    agg = wk.groupby(["player_id", "player_display_name", "position"],
                     as_index=False).agg(
        games=("week", "nunique"),
        targets=("targets", "sum"), team_targets=("team_targets", "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_air_yards=("receiving_air_yards", "sum"),
        team_air_yards=("team_air_yards", "sum"),
        receiving_yards_after_catch=("receiving_yards_after_catch", "sum"),
        carries=("carries", "sum"), team_carries=("team_carries", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        attempts=("attempts", "sum"), team_pass_attempts=("team_pass_attempts", "sum"),
        passing_yards=("passing_yards", "sum"),
        sacks=("sacks", "sum"),
        rush_td=("rushing_tds", "sum"), rec_td=("receiving_tds", "sum"),
        pass_td=("passing_tds", "sum"),
        target_share_avg=("target_share", "mean"),
        air_yards_share_avg=("air_yards_share", "mean"),
    )
    agg = agg.rename(columns={"player_display_name": "player_name"})
    # team = whichever team the player played for in their most recent week
    # that season (handles in-season trades without double-counting the player)
    last_team = (wk.sort_values("week").groupby("player_id")["recent_team"]
                .last().rename("team"))
    agg = agg.merge(last_team, on="player_id", how="left")

    # red zone denominators need the SAME team the numerator was earned under,
    # so join those before collapsing to one row per player, then sum.
    rz_tgt_pw = rz_tgt.rename(columns={"team": "_team"})
    rz_rsh_pw = rz_rsh.rename(columns={"team": "_team"})
    rz_tgt_season = rz_tgt_pw.groupby("player_id", as_index=False).agg(
        rz_targets=("rz_targets", "sum"), inside10_targets=("inside10_targets", "sum"))
    rz_rsh_season = rz_rsh_pw.groupby("player_id", as_index=False).agg(
        rz_carries=("rz_carries", "sum"), inside5_carries=("inside5_carries", "sum"))
    # team-level red zone denominator, attributed per player-week then summed,
    # so a traded player's share still uses the right team's RZ volume each week
    tgt_denom = wk[["player_id", "season", "week", "recent_team"]].merge(
        team_rz_week_lookup, left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"], how="left").groupby(
        "player_id", as_index=False).agg(
        team_rz_targets=("team_rz_targets", "sum"), team_rz_carries=("team_rz_carries", "sum"),
        team_inside5_carries=("team_inside5_carries", "sum"))

    agg = agg.merge(rz_tgt_season, on="player_id", how="left")
    agg = agg.merge(rz_rsh_season, on="player_id", how="left")
    agg = agg.merge(tgt_denom, on="player_id", how="left")
    agg = agg.merge(snap_season, left_on="player_id", right_on="gsis_id", how="left")

    out = pd.DataFrame()
    out["player_id"] = agg["player_id"]
    out["player_name"] = agg["player_name"]
    out["season"] = season
    out["team"] = agg["team"]
    out["position"] = agg["position"]
    out["age"] = np.nan
    out["games"] = agg["games"]
    out["snap_share"] = agg["snap_share"]
    out["route_participation"] = np.nan  # not in free data; needs charting/PFF
    out["target_share"] = safe_div(agg["targets"], agg["team_targets"])
    out["targets_per_game"] = safe_div(agg["targets"], agg["games"])
    out["air_yards_share"] = safe_div(agg["receiving_air_yards"], agg["team_air_yards"])
    out["adot"] = safe_div(agg["receiving_air_yards"], agg["targets"])
    out["catch_rate"] = safe_div(agg["receptions"], agg["targets"])
    out["yptarget"] = safe_div(agg["receiving_yards"], agg["targets"])
    out["yac_per_rec"] = safe_div(agg["receiving_yards_after_catch"], agg["receptions"])
    out["rush_share"] = safe_div(agg["carries"], agg["team_carries"])
    out["carries_per_game"] = safe_div(agg["carries"], agg["games"])
    out["ypc"] = safe_div(agg["rushing_yards"], agg["carries"])
    out["rz_target_share"] = safe_div(agg["rz_targets"], agg["team_rz_targets"])
    out["rz_rush_share"] = safe_div(agg["rz_carries"], agg["team_rz_carries"])
    out["inside5_rush_share"] = safe_div(agg["inside5_carries"], agg["team_inside5_carries"])
    out["inside10_target_share"] = safe_div(agg["inside10_targets"], agg["team_rz_targets"])
    out["rush_td"] = agg["rush_td"]
    out["rec_td"] = agg["rec_td"]
    out["pass_att_share"] = safe_div(agg["attempts"], agg["team_pass_attempts"])
    out["ypa"] = safe_div(agg["passing_yards"], agg["attempts"])
    out["sack_rate"] = safe_div(agg["sacks"], agg["attempts"] + agg["sacks"])
    out["pass_td"] = agg["pass_td"]
    out["receiving_yards"] = agg["receiving_yards"]
    out["rushing_yards"] = agg["rushing_yards"]
    out["passing_yards"] = agg["passing_yards"]
    return out


def build_team_offense_season(season: int) -> pd.DataFrame:
    pbp = _pbp(season)
    wk = team_offense_week(pbp)
    games = wk.groupby("team")["week"].nunique().rename("games")
    g = wk.groupby("team", as_index=False).agg(
        plays_per_game=("plays", "mean"),
        pass_rate=("dropbacks", "sum"),  # placeholder, replaced below
        proe=("proe", "mean"),
        off_epa_per_play=("epa_per_play", "mean"),
        off_td_per_game=("off_td", "mean"),
        rush_td_share=("rush_td", "sum"),
        points_per_game=("points", "mean"),
    )
    totals = wk.groupby("team", as_index=False).agg(
        total_plays=("plays", "sum"), total_dropbacks=("dropbacks", "sum"),
        total_off_td=("off_td", "sum"), total_rush_td=("rush_td", "sum"))
    g = g.drop(columns=["pass_rate"]).merge(totals, on="team")
    g["pass_rate"] = safe_div(g["total_dropbacks"], g["total_plays"])
    g["rush_td_share"] = safe_div(g["total_rush_td"], g["total_off_td"])
    g = g.merge(games, on="team")

    team_rz = redzone_team_week(pbp).groupby("team", as_index=False).agg(
        team_rz_targets=("team_rz_targets", "sum"), team_rz_carries=("team_rz_carries", "sum"))
    g = g.merge(team_rz, on="team", how="left")
    g["rz_pass_rate"] = safe_div(g["team_rz_targets"], g["team_rz_targets"] + g["team_rz_carries"])
    g["rz_trips_per_game"] = safe_div(g["team_rz_targets"] + g["team_rz_carries"], g["games"]) / 3.0
    g["inside5_plays_pg"] = np.nan  # filled from redzone_team_week's inside5 if desired

    out = pd.DataFrame()
    out["team"] = g["team"]
    out["season"] = season
    out["games"] = g["games"]
    out["plays_per_game"] = g["plays_per_game"]
    out["pace_sec_per_play"] = np.nan  # needs game-duration join; leave to config default
    out["pass_rate"] = g["pass_rate"]
    out["proe"] = g["proe"]
    out["rz_pass_rate"] = g["rz_pass_rate"]
    out["rz_trips_per_game"] = g["rz_trips_per_game"]
    out["inside5_plays_pg"] = g["inside5_plays_pg"]
    out["points_per_game"] = g["points_per_game"]
    out["off_td_per_game"] = g["off_td_per_game"]
    out["rush_td_share"] = g["rush_td_share"]
    out["off_epa_per_play"] = g["off_epa_per_play"]
    out["team_pass_att_pg"] = safe_div(g["total_dropbacks"], g["games"])
    out["team_rush_att_pg"] = safe_div(g["total_plays"] - g["total_dropbacks"], g["games"])
    out["team_pass_yds_pg"] = np.nan
    out["team_rush_yds_pg"] = np.nan
    out["ol_passblock_grade"] = GRADE_PLACEHOLDER
    out["ol_runblock_grade"] = GRADE_PLACEHOLDER
    out["ol_snaps_returning"] = 0.70
    out["wr_corps_grade"] = GRADE_PLACEHOLDER
    out["qb_grade"] = GRADE_PLACEHOLDER
    out["offensive_coordinator"] = "UNKNOWN"
    out["head_coach"] = "UNKNOWN"
    out["play_caller"] = "UNKNOWN"
    out["scheme_run"] = "mixed"
    out["scheme_pass"] = "mixed"
    return out


def build_team_defense_season(season: int) -> pd.DataFrame:
    pbp = _pbp(season)
    wk = team_defense_week(pbp)
    g = wk.groupby("team", as_index=False).agg(
        epa_per_play_allowed=("epa_per_play_allowed", "mean"),
        epa_per_pass_allowed=("epa_per_pass_allowed", "mean"),
        epa_per_rush_allowed=("epa_per_rush_allowed", "mean"),
        pass_yds_allowed=("pass_yds_allowed", "sum"),
        pass_att_faced=("pass_att_faced", "sum"),
        rush_yds_allowed=("rush_yds_allowed", "sum"),
        carries_faced=("carries_faced", "sum"),
        pass_td_allowed=("pass_td_allowed", "sum"),
        rush_td_allowed=("rush_td_allowed", "sum"),
        rz_trips_faced=("rz_trips_faced", "sum"),
        rz_td_allowed=("rz_td_allowed", "sum"),
        plays_faced=("plays_faced", "sum"),
        games=("week", "nunique"),
    )
    out = pd.DataFrame()
    out["team"] = g["team"]
    out["season"] = season
    out["games"] = g["games"]
    out["def_epa_per_play_allowed"] = g["epa_per_play_allowed"]
    out["def_epa_per_pass_allowed"] = g["epa_per_pass_allowed"]
    out["def_epa_per_rush_allowed"] = g["epa_per_rush_allowed"]
    out["def_ypa_allowed"] = safe_div(g["pass_yds_allowed"], g["pass_att_faced"])
    out["def_ypc_allowed"] = safe_div(g["rush_yds_allowed"], g["carries_faced"])
    out["def_pass_yds_pg_allowed"] = safe_div(g["pass_yds_allowed"], g["games"])
    out["def_rush_yds_pg_allowed"] = safe_div(g["rush_yds_allowed"], g["games"])
    out["def_plays_pg_faced"] = safe_div(g["plays_faced"], g["games"])
    out["def_pace_sec_allowed"] = np.nan
    out["def_rz_td_rate_allowed"] = safe_div(g["rz_td_allowed"], g["rz_trips_faced"])
    out["def_rush_td_pg_allowed"] = safe_div(g["rush_td_allowed"], g["games"])
    out["def_pass_td_pg_allowed"] = safe_div(g["pass_td_allowed"], g["games"])
    out["def_pressure_rate"] = np.nan
    out["def_light_box_rate"] = np.nan
    out["def_man_rate"] = np.nan
    out["def_zone_rate"] = np.nan
    out["def_vs_wr_out"] = 0.0
    out["def_vs_wr_slot"] = 0.0
    out["def_vs_te"] = 0.0
    out["def_vs_rb_pass"] = 0.0
    out["cb1_grade"] = GRADE_PLACEHOLDER
    out["cb2_grade"] = GRADE_PLACEHOLDER
    out["slot_cb_grade"] = GRADE_PLACEHOLDER
    out["shadow_cb"] = False
    out["defensive_coordinator"] = "UNKNOWN"
    return out


def build_roster_changes(season: int, prior_season: int) -> pd.DataFrame:
    """
    Real for team changes and rookies (from roster files). Everything about
    coordinators, scheme, and depth chart is left at safe defaults for you to
    hand-edit -- see the module docstring for why that part isn't scraped.
    """
    cur = _rosters(season)[["player_id", "team", "position"]].dropna().drop_duplicates("player_id")
    cur = cur.rename(columns={"player_id": "gsis_id"})
    prior = _rosters(prior_season)[["player_id", "team"]].dropna().drop_duplicates("player_id")
    prior = prior.rename(columns={"player_id": "gsis_id", "team": "prior_team"})

    m = cur.merge(prior, on="gsis_id", how="left")
    out = pd.DataFrame()
    out["player_id"] = m["gsis_id"]
    out["season"] = season
    out["position"] = m["position"]
    out["prior_team"] = m["prior_team"]
    out["new_team"] = m["team"]
    out["changed_team"] = (m["prior_team"].notna()) & (m["prior_team"] != m["team"])
    out["rookie"] = m["prior_team"].isna()
    out["new_oc"] = False
    out["new_hc_offense"] = False
    out["new_dc"] = False
    out["scheme_change_major"] = False
    out["new_qb"] = False
    out["depth_chart_rank_prior"] = np.nan
    out["depth_chart_rank_new"] = np.nan
    out["depth_chart_volatility"] = 0.10
    out["returning_from_ir"] = False
    out["usage_shift_tags"] = ""
    out["manual_share_override"] = np.nan
    out["notes"] = ""
    return out


def build_game_environment(season: int, week: int) -> pd.DataFrame:
    games = _games()
    g = games[(games.season == season) & (games.week == week)].copy()
    rows = []
    for _, r in g.iterrows():
        total = r.get("total_line", np.nan)
        for team, opp, spread, home in (
            (r["home_team"], r["away_team"], -r.get("spread_line", 0.0) or 0.0, True),
            (r["away_team"], r["home_team"], r.get("spread_line", 0.0) or 0.0, False),
        ):
            rows.append({
                "season": season, "week": week, "team": team, "opponent": opp,
                "home": home, "spread": spread, "total": total,
                "implied_total": (total / 2 - spread / 2) if pd.notna(total) else np.nan,
                "dome": r.get("roof", "") in ("dome", "closed"),
                "wind_mph": r.get("wind", np.nan),
                "precip_prob": np.nan,
            })
    return pd.DataFrame(rows)


def build_player_gamelog(season: int, through_week: int) -> pd.DataFrame:
    from nflprops import schema
    cols = list(schema.PLAYER_GAMELOG.keys())
    if through_week <= 1:
        # Week 1 has no prior games this season to log yet -- don't even try
        # to fetch a season file that doesn't exist until games are played.
        return pd.DataFrame(columns=cols)

    wk = _weekly(season)
    wk = wk[(wk["season_type"] == "REG") & (wk["week"] < through_week)]
    if not len(wk):
        return pd.DataFrame()
    tt = team_week_totals(wk)
    wk = wk.merge(tt, left_on=["season", "week", "recent_team"],
                 right_on=["season", "week", "team"], how="left")

    pbp = _pbp(season)
    pbp = pbp[pbp["week"] < through_week]
    rz_tgt, rz_rsh = redzone_player_week(pbp)
    team_rz = redzone_team_week(pbp)

    snaps = _snaps(season)
    snaps = snaps[snaps["week"] < through_week]
    ids = _ids()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")

    out = pd.DataFrame()
    out["player_id"] = wk["player_id"]
    out["season"] = season
    out["week"] = wk["week"]
    out["team"] = wk["recent_team"]
    out["opponent"] = wk["opponent_team"]
    out["position"] = wk["position"]
    out["team_snaps"] = np.nan
    out["snaps"] = np.nan
    out["routes"] = np.nan
    out["team_dropbacks"] = wk["team_pass_attempts"]
    out["targets"] = wk["targets"]
    out["team_targets"] = wk["team_targets"]
    out["receptions"] = wk["receptions"]
    out["receiving_yards"] = wk["receiving_yards"]
    out["air_yards"] = wk["receiving_air_yards"]
    out["team_air_yards"] = wk["team_air_yards"]
    out["carries"] = wk["carries"]
    out["team_carries"] = wk["team_carries"]
    out["rushing_yards"] = wk["rushing_yards"]
    out["pass_attempts"] = wk["attempts"]
    out["passing_yards"] = wk["passing_yards"]
    out["rush_td"] = wk["rushing_tds"]
    out["rec_td"] = wk["receiving_tds"]
    out["pass_td"] = wk["passing_tds"]
    out["anytime_td"] = ((wk["rushing_tds"] + wk["receiving_tds"]) > 0).astype(int)

    out = out.merge(rz_tgt.rename(columns={"team": "_t"}), on=["player_id", "season", "week"],
                    how="left")
    out = out.merge(rz_rsh.rename(columns={"team": "_t2"}), on=["player_id", "season", "week"],
                    how="left")
    out = out.merge(team_rz, left_on=["season", "week", "team"],
                    right_on=["season", "week", "team"], how="left")
    out["rz_targets"] = out.pop("rz_targets") if "rz_targets" in out else np.nan
    out["team_rz_targets"] = out.pop("team_rz_targets") if "team_rz_targets" in out else np.nan
    out["rz_carries"] = out.pop("rz_carries") if "rz_carries" in out else np.nan
    out["team_rz_carries"] = out.pop("team_rz_carries") if "team_rz_carries" in out else np.nan
    out["inside5_carries"] = out.pop("inside5_carries") if "inside5_carries" in out else np.nan
    out["team_inside5_carries"] = out.pop("team_inside5_carries") if "team_inside5_carries" in out else np.nan

    snap_gl = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left") \
        if "pfr_id" not in snaps.columns else snaps
    snap_gl = snap_gl[["gsis_id", "week", "offense_snaps", "offense_pct"]].dropna(subset=["gsis_id"])
    out = out.merge(snap_gl, left_on=["player_id", "week"], right_on=["gsis_id", "week"], how="left")
    out["snaps"] = out.pop("offense_snaps")
    out["team_snaps"] = safe_div(out["snaps"], out.pop("offense_pct"))
    out = out.drop(columns=[c for c in ("gsis_id", "_t", "_t2") if c in out.columns])

    from nflprops import schema
    cols = list(schema.PLAYER_GAMELOG.keys())
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def build_team_gamelogs(season: int, through_week: int):
    off_cols = ["season", "week", "team", "plays", "dropbacks", "pass_attempts",
               "carries", "epa_per_play", "proe", "off_td", "rush_td", "points",
               "pace_sec_per_play", "rz_dropbacks", "rz_plays"]
    def_cols = ["season", "week", "team", "epa_per_play_allowed", "plays_faced",
               "epa_per_pass_allowed", "pass_yds_allowed", "pass_att_faced",
               "epa_per_rush_allowed", "rush_yds_allowed", "carries_faced",
               "pass_td_allowed", "rush_td_allowed", "rz_trips_faced", "rz_td_allowed"]
    if through_week <= 1:
        return pd.DataFrame(columns=off_cols), pd.DataFrame(columns=def_cols)

    pbp = _pbp(season)
    pbp = pbp[pbp["week"] < through_week]
    off = team_offense_week(pbp)
    off["pace_sec_per_play"] = np.nan
    off["pass_attempts"] = off["dropbacks"]
    off["carries"] = off["plays"] - off["dropbacks"]
    off["rz_dropbacks"] = np.nan
    off["rz_plays"] = np.nan
    off = off[["season", "week", "team", "plays", "dropbacks", "pass_attempts",
              "carries", "epa_per_play", "proe", "off_td", "rush_td", "points",
              "pace_sec_per_play", "rz_dropbacks", "rz_plays"]]

    d = team_defense_week(pbp)
    return off, d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("prior", help="build data/prior/*.csv for one completed season")
    p1.add_argument("--season", type=int, required=True)
    p1.add_argument("--root", default="data")

    p2 = sub.add_parser("current", help="build data/current/*.csv + game_environment for a week")
    p2.add_argument("--season", type=int, required=True)
    p2.add_argument("--through-week", type=int, required=True)
    p2.add_argument("--root", default="data")

    p3 = sub.add_parser("roster", help="build roster_changes.csv for a season")
    p3.add_argument("--season", type=int, required=True)
    p3.add_argument("--prior-season", type=int, required=True)
    p3.add_argument("--root", default="data")

    args = ap.parse_args()

    if args.mode == "prior":
        os.makedirs(f"{args.root}/prior", exist_ok=True)
        build_player_season(args.season).to_csv(f"{args.root}/prior/player_season.csv", index=False)
        build_team_offense_season(args.season).to_csv(f"{args.root}/prior/team_season_offense.csv", index=False)
        build_team_defense_season(args.season).to_csv(f"{args.root}/prior/team_season_defense.csv", index=False)
        print(f"wrote prior-season tables for {args.season} to {args.root}/prior/", file=sys.stderr)

    elif args.mode == "current":
        os.makedirs(f"{args.root}/current", exist_ok=True)
        gl = build_player_gamelog(args.season, args.through_week)
        gl.to_csv(f"{args.root}/current/player_gamelog.csv", index=False)
        off, d = build_team_gamelogs(args.season, args.through_week)
        off.to_csv(f"{args.root}/current/team_gamelog_offense.csv", index=False)
        d.to_csv(f"{args.root}/current/team_gamelog_defense.csv", index=False)
        env = build_game_environment(args.season, args.through_week)
        env_path = f"{args.root}/game_environment.csv"
        if os.path.exists(env_path):
            old = pd.read_csv(env_path)
            old = old[~((old.season == args.season) & (old.week == args.through_week))]
            env = pd.concat([old, env], ignore_index=True)
        env.to_csv(env_path, index=False)
        print(f"wrote current-season data through week {args.through_week - 1} "
              f"and week {args.through_week} game environment to {args.root}/", file=sys.stderr)

    elif args.mode == "roster":
        build_roster_changes(args.season, args.prior_season).to_csv(
            f"{args.root}/roster_changes.csv", index=False)
        print(f"wrote roster_changes.csv for {args.season} to {args.root}/ "
              f"(team changes + rookies are real; edit OC/scheme/depth-chart "
              f"columns by hand)", file=sys.stderr)


if __name__ == "__main__":
    main()
