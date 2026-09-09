"""
Loaders. Everything is CSV/Parquet in, validated DataFrame out.

Directory layout expected:

    data/
      prior/
        player_season.csv
        team_season_offense.csv
        team_season_defense.csv
      current/
        player_gamelog.csv          <- appended every week
        team_gamelog_offense.csv    <- appended every week
        team_gamelog_defense.csv    <- appended every week
      roster_changes.csv
      game_environment.csv          <- refreshed weekly from the market
      prop_lines.csv                <- refreshed as often as you like

Week 1 works with prior/ + roster_changes + game_environment only. The current/
files can be empty or absent; the aggregators return empty frames and the blend
falls back to prior-season plus baseline automatically.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .. import schema
from ..config import ModelConfig, DEFAULT_CONFIG
from .blend import decayed_mean, decayed_rate


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


class DataStore:
    """Holds every table the pipeline needs, plus the derived current-season aggregates."""

    def __init__(self, root: str, season: int, week: int,
                 cfg: ModelConfig = DEFAULT_CONFIG):
        self.root = root
        self.season = season
        self.week = week
        self.cfg = cfg
        self.prior_season = season - 1

        p = lambda *a: os.path.join(root, *a)
        self.player_season = _read(p("prior", "player_season.csv"))
        self.team_off_prior = _read(p("prior", "team_season_offense.csv"))
        self.team_def_prior = _read(p("prior", "team_season_defense.csv"))
        self.player_gamelog = _read(p("current", "player_gamelog.csv"))
        self.team_off_gamelog = _read(p("current", "team_gamelog_offense.csv"))
        self.team_def_gamelog = _read(p("current", "team_gamelog_defense.csv"))
        self.roster_changes = _read(p("roster_changes.csv"))
        self.game_env = _read(p("game_environment.csv"))
        self.prop_lines = _read(p("prop_lines.csv"))

        self._filter_to_week()

    # ------------------------------------------------------------------
    def _filter_to_week(self):
        """Only use games strictly before the target week. Prevents leakage."""
        for name in ("player_gamelog", "team_off_gamelog", "team_def_gamelog"):
            df = getattr(self, name)
            if len(df) and "week" in df.columns:
                setattr(self, name, df[(df["season"] == self.season)
                                       & (df["week"] < self.week)].copy())

    def validate(self, strict: bool = False) -> dict:
        report = {}
        pairs = [("player_season", self.player_season),
                 ("team_season_offense", self.team_off_prior),
                 ("team_season_defense", self.team_def_prior),
                 ("roster_changes", self.roster_changes),
                 ("game_environment", self.game_env)]
        if len(self.player_gamelog):
            pairs.append(("player_gamelog", self.player_gamelog))
        for name, df in pairs:
            report[name] = schema.validate(df, name, strict=strict) if len(df) else ["<empty>"]
        return report

    # ------------------------------------------------------------------
    # Current-season aggregation with recency decay
    # ------------------------------------------------------------------
    def current_player_usage(self) -> pd.DataFrame:
        """Recency-weighted current-season rates, one row per player."""
        gl = self.player_gamelog
        cols = ["player_id", "n_games_current", "snap_share", "route_participation",
                "target_share", "air_yards_share", "rush_share", "rz_target_share",
                "rz_rush_share", "inside5_rush_share", "ypc", "yptarget", "adot",
                "catch_rate", "ypa", "pass_att_share", "targets_per_game",
                "carries_per_game"]
        if not len(gl):
            return pd.DataFrame(columns=cols)

        w = self.week
        rows = []
        for pid, g in gl.groupby("player_id"):
            wk = g["week"]
            r = {
                "player_id": pid,
                "n_games_current": int((g.get("snaps", pd.Series(0, index=g.index)) > 0).sum()),
                "snap_share": decayed_rate(g.get("snaps", 0), g.get("team_snaps", 0), wk, w),
                "route_participation": decayed_rate(g.get("routes", 0), g.get("team_dropbacks", 0), wk, w),
                "target_share": decayed_rate(g.get("targets", 0), g.get("team_targets", 0), wk, w),
                "air_yards_share": decayed_rate(g.get("air_yards", 0), g.get("team_air_yards", 0), wk, w),
                "rush_share": decayed_rate(g.get("carries", 0), g.get("team_carries", 0), wk, w),
                "rz_target_share": decayed_rate(g.get("rz_targets", 0), g.get("team_rz_targets", 0), wk, w),
                "rz_rush_share": decayed_rate(g.get("rz_carries", 0), g.get("team_rz_carries", 0), wk, w),
                "inside5_rush_share": decayed_rate(g.get("inside5_carries", 0), g.get("team_inside5_carries", 0), wk, w),
                "ypc": decayed_rate(g.get("rushing_yards", 0), g.get("carries", 0), wk, w),
                "yptarget": decayed_rate(g.get("receiving_yards", 0), g.get("targets", 0), wk, w),
                "adot": decayed_rate(g.get("air_yards", 0), g.get("targets", 0), wk, w),
                "catch_rate": decayed_rate(g.get("receptions", 0), g.get("targets", 0), wk, w),
                "ypa": decayed_rate(g.get("passing_yards", 0), g.get("pass_attempts", 0), wk, w),
                "pass_att_share": decayed_rate(g.get("pass_attempts", 0), g.get("team_dropbacks", 0), wk, w),
                "targets_per_game": decayed_mean(g.get("targets", pd.Series(dtype=float)), wk, w),
                "carries_per_game": decayed_mean(g.get("carries", pd.Series(dtype=float)), wk, w),
            }
            rows.append(r)
        return pd.DataFrame(rows)

    def current_team_offense(self) -> pd.DataFrame:
        gl = self.team_off_gamelog
        if not len(gl):
            return pd.DataFrame(columns=["team", "n_games_current"])
        w = self.week
        empty = pd.Series(dtype=float)
        rows = []
        for team, g in gl.groupby("team"):
            wk = g["week"]
            rows.append({
                "team": team,
                "n_games_current": g["week"].nunique(),
                "plays_per_game": decayed_mean(g.get("plays", empty), wk, w),
                "pace_sec_per_play": decayed_mean(g.get("pace_sec_per_play", empty), wk, w),
                "pass_rate": decayed_rate(g.get("dropbacks", empty), g.get("plays", empty), wk, w),
                "proe": decayed_mean(g.get("proe", empty), wk, w),
                "rz_pass_rate": decayed_rate(g.get("rz_dropbacks", empty), g.get("rz_plays", empty), wk, w),
                "team_pass_att_pg": decayed_mean(g.get("pass_attempts", empty), wk, w),
                "team_rush_att_pg": decayed_mean(g.get("carries", empty), wk, w),
                "off_td_per_game": decayed_mean(g.get("off_td", empty), wk, w),
                "points_per_game": decayed_mean(g.get("points", empty), wk, w),
                "rush_td_share": decayed_rate(g.get("rush_td", empty), g.get("off_td", empty), wk, w),
                "off_epa_per_play": decayed_mean(g.get("epa_per_play", empty), wk, w),
            })
        return pd.DataFrame(rows)

    def current_team_defense(self) -> pd.DataFrame:
        gl = self.team_def_gamelog
        if not len(gl):
            return pd.DataFrame(columns=["team", "n_games_current"])
        w = self.week
        empty = pd.Series(dtype=float)
        rows = []
        for team, g in gl.groupby("team"):
            wk = g["week"]
            rows.append({
                "team": team,
                "n_games_current": g["week"].nunique(),
                "def_epa_per_play_allowed": decayed_mean(g.get("epa_per_play_allowed", empty), wk, w),
                "def_epa_per_pass_allowed": decayed_mean(g.get("epa_per_pass_allowed", empty), wk, w),
                "def_epa_per_rush_allowed": decayed_mean(g.get("epa_per_rush_allowed", empty), wk, w),
                "def_ypa_allowed": decayed_rate(g.get("pass_yds_allowed", empty), g.get("pass_att_faced", empty), wk, w),
                "def_ypc_allowed": decayed_rate(g.get("rush_yds_allowed", empty), g.get("carries_faced", empty), wk, w),
                "def_rz_td_rate_allowed": decayed_rate(g.get("rz_td_allowed", empty), g.get("rz_trips_faced", empty), wk, w),
                "def_pass_td_pg_allowed": decayed_mean(g.get("pass_td_allowed", empty), wk, w),
                "def_rush_td_pg_allowed": decayed_mean(g.get("rush_td_allowed", empty), wk, w),
                "def_plays_pg_faced": decayed_mean(g.get("plays_faced", empty), wk, w),
                "def_vs_wr_out": decayed_mean(g.get("epa_vs_wr_out", empty), wk, w),
                "def_vs_wr_slot": decayed_mean(g.get("epa_vs_wr_slot", empty), wk, w),
                "def_vs_te": decayed_mean(g.get("epa_vs_te", empty), wk, w),
                "def_vs_rb_pass": decayed_mean(g.get("epa_vs_rb_pass", empty), wk, w),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def player_meta(self) -> pd.DataFrame:
        """player_id -> position, current team. Roster file wins over last season."""
        base = self.player_season[["player_id", "player_name", "position", "team"]].copy() \
            if len(self.player_season) else pd.DataFrame(
                columns=["player_id", "player_name", "position", "team"])
        if len(self.roster_changes) and "new_team" in self.roster_changes.columns:
            rc_cols = ["player_id", "new_team"]
            if "position" in self.roster_changes.columns:
                rc_cols.append("position")
            rc = self.roster_changes[rc_cols].dropna(subset=["player_id", "new_team"])
            base = base.merge(rc, on="player_id", how="outer", suffixes=("", "_rc"))
            base["team"] = base["new_team"].fillna(base["team"])
            if "position_rc" in base.columns:
                base["position"] = base["position_rc"].fillna(base["position"])
                base = base.drop(columns=["position_rc"])
            base = base.drop(columns=["new_team"])
        if len(self.player_gamelog):
            latest = (self.player_gamelog.sort_values("week")
                      .groupby("player_id").tail(1)[["player_id", "team", "position"]])
            base = base.merge(latest, on="player_id", how="outer", suffixes=("", "_gl"))
            base["team"] = base["team_gl"].fillna(base["team"])
            base["position"] = base["position_gl"].fillna(base["position"])
            base = base.drop(columns=[c for c in ("team_gl", "position_gl") if c in base])
        base["player_name"] = base["player_name"].fillna(base["player_id"])
        return base.drop_duplicates("player_id").reset_index(drop=True)

    def environment_for(self, team: str) -> dict:
        env = self.game_env
        if not len(env):
            return {}
        row = env[(env["season"] == self.season) & (env["week"] == self.week)
                  & (env["team"] == team)]
        if not len(row):
            return {}
        r = row.iloc[0].to_dict()
        if "implied_total" not in r or pd.isna(r.get("implied_total")):
            r["implied_total"] = r.get("total", 44.0) / 2 - r.get("spread", 0.0) / 2
        return r
