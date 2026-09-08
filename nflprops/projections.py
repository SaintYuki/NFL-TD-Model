"""
Projection engine. Ties the pipeline together:

    DataStore -> blend(prior, current, league) -> roster adjustments
      -> features -> four models -> line probabilities -> JSON

Public entry points:
    engine = ProjectionEngine(store)
    engine.project_player("00-0034796", markets=["passing_yards","anytime_td"])
    engine.project_slate()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .util import safe_num
from .config import ModelConfig, DEFAULT_CONFIG
from .data.blend import blend_value, blend_weights_report
from .data.roster import build_roster_context, apply_roster_adjustments
from .features import build_feature_row
from .models.anytime_td import AnytimeTDModel
from .models.yardage import MODEL_REGISTRY
from .probability import edge_and_stake, devig_two_way
from .output import assemble_projection_json

PLAYER_METRICS = [
    "snap_share", "route_participation", "target_share", "air_yards_share",
    "rush_share", "rz_target_share", "rz_rush_share", "inside5_rush_share",
    "inside10_target_share", "ypc", "yptarget", "adot", "catch_rate",
    "ypa", "pass_att_share", "sack_rate",
]
TEAM_OFF_METRICS = [
    "plays_per_game", "pace_sec_per_play", "pass_rate", "proe", "rz_pass_rate",
    "team_pass_att_pg", "team_rush_att_pg", "off_td_per_game", "rush_td_share",
    "points_per_game", "off_epa_per_play",
]
TEAM_DEF_METRICS = [
    "def_epa_per_play_allowed", "def_epa_per_pass_allowed", "def_epa_per_rush_allowed",
    "def_ypa_allowed", "def_ypc_allowed", "def_rz_td_rate_allowed",
    "def_pass_td_pg_allowed", "def_rush_td_pg_allowed", "def_plays_pg_faced",
    "def_vs_wr_out", "def_vs_wr_slot", "def_vs_te", "def_vs_rb_pass",
]
# static (non-blended) team attributes carried straight through
TEAM_STATIC = [
    "ol_passblock_grade", "ol_runblock_grade", "ol_snaps_returning",
    "wr_corps_grade", "qb_grade", "scheme_run", "scheme_pass",
    "offensive_coordinator", "head_coach",
]
DEF_STATIC = ["cb1_grade", "cb2_grade", "slot_cb_grade", "shadow_cb",
              "def_man_rate", "def_zone_rate", "def_light_box_rate"]


class ProjectionEngine:
    def __init__(self, store, cfg: ModelConfig = DEFAULT_CONFIG):
        self.store = store
        self.cfg = cfg
        self.week = store.week
        self.season = store.season

        self.meta = store.player_meta()
        self.roster_ctx = build_roster_context(store.roster_changes, self.meta, cfg)

        self.cur_usage = store.current_player_usage().set_index("player_id") \
            if len(store.current_player_usage()) else pd.DataFrame()
        self.cur_team_off = store.current_team_offense().set_index("team") \
            if len(store.current_team_offense()) else pd.DataFrame()
        self.cur_team_def = store.current_team_defense().set_index("team") \
            if len(store.current_team_defense()) else pd.DataFrame()

        self.prior_players = store.player_season.set_index("player_id") \
            if len(store.player_season) else pd.DataFrame()
        self.prior_off = store.team_off_prior.set_index("team") \
            if len(store.team_off_prior) else pd.DataFrame()
        self.prior_def = store.team_def_prior.set_index("team") \
            if len(store.team_def_prior) else pd.DataFrame()

        self.td_model = AnytimeTDModel(cfg)
        self.yard_models = {k: cls(cfg) for k, cls in MODEL_REGISTRY.items()}
        self._usage_cache = None

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------
    def _row(self, frame: pd.DataFrame, key) -> dict:
        if frame is None or not len(frame) or key not in frame.index:
            return {}
        r = frame.loc[key]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r.to_dict()

    def blended_player(self, player_id: str) -> tuple[dict, dict]:
        ctx = self.roster_ctx[self.roster_ctx.player_id == player_id]
        ctx = ctx.iloc[0].to_dict() if len(ctx) else {}
        pos = ctx.get("position") or "WR"
        rho = float(ctx.get("rho", 0.9))

        prior = self._row(self.prior_players, player_id)
        cur = self._row(self.cur_usage, player_id)
        n_cur = safe_num(cur.get("n_games_current"), 0)

        posbase = self.cfg.position.get(pos, self.cfg.position["WR"])
        blended, weights = {}, {}
        for m in PLAYER_METRICS:
            lg = posbase.get(m, self.cfg.league.get(m, np.nan))
            if pd.isna(lg):
                lg = 0.0
            blended[m] = blend_value(m, cur.get(m), prior.get(m), lg,
                                     n_cur, rho, self.week, self.cfg)
            if m in ("target_share", "rush_share", "ypc", "yptarget", "ypa"):
                weights[m] = blend_weights_report(m, n_cur, rho, self.week, self.cfg)

        blended["position"] = pos
        blended["player_id"] = player_id
        blended["n_games_current"] = n_cur
        # RZ share fallbacks: red zone samples are tiny, lean on overall share
        if not blended.get("inside5_rush_share"):
            blended["inside5_rush_share"] = 0.85 * blended.get("rz_rush_share", 0.0)
        if not blended.get("inside10_target_share"):
            blended["inside10_target_share"] = 0.9 * blended.get("rz_target_share", 0.0)
        return blended, {"blend_weights": weights, "n_games_current": n_cur}

    def blended_team_offense(self, team: str) -> dict:
        prior = self._row(self.prior_off, team)
        cur = self._row(self.cur_team_off, team)
        n = safe_num(cur.get("n_games_current"), 0)
        rho = self._team_rho(team, side="offense")
        out = {}
        for m in TEAM_OFF_METRICS:
            out[m] = blend_value(m, cur.get(m), prior.get(m),
                                 self.cfg.league.get(m, np.nan), n, rho, self.week, self.cfg)
        for m in TEAM_STATIC:
            out[m] = prior.get(m)
        out["team_rho"] = rho
        return out

    def blended_team_defense(self, team: str) -> dict:
        prior = self._row(self.prior_def, team)
        cur = self._row(self.cur_team_def, team)
        n = safe_num(cur.get("n_games_current"), 0)
        rho = self._team_rho(team, side="defense")
        out = {}
        for m in TEAM_DEF_METRICS:
            out[m] = blend_value(m, cur.get(m), prior.get(m),
                                 self.cfg.league.get(m, np.nan), n, rho, self.week, self.cfg)
        for m in DEF_STATIC:
            out[m] = prior.get(m)
        out["team_rho"] = rho
        return out

    def _team_rho(self, team: str, side: str) -> float:
        """
        Team-level continuity. A new coordinator makes last year's team
        tendencies less predictive of this year's, exactly like a player
        changing teams.
        """
        rho = 0.92
        rc = self.store.roster_changes
        if len(rc) and "new_team" in rc.columns:
            sub = rc[rc["new_team"] == team]
            if len(sub):
                if side == "offense":
                    if sub["new_oc"].fillna(False).any():
                        rho *= self.cfg.continuity["new_oc"]
                    if "new_hc_offense" in sub and sub["new_hc_offense"].fillna(False).any():
                        rho *= self.cfg.continuity["new_hc_offense"]
                    if "scheme_change_major" in sub and sub["scheme_change_major"].fillna(False).any():
                        rho *= self.cfg.continuity["scheme_change_major"]
                else:
                    if "new_dc" in sub and sub["new_dc"].fillna(False).any():
                        rho *= self.cfg.continuity["new_dc"]
        return float(np.clip(rho, 0.25, 1.0))

    # ------------------------------------------------------------------
    # Team-wide usage normalization (run once, cached)
    # ------------------------------------------------------------------
    def team_adjusted_usage(self) -> pd.DataFrame:
        if self._usage_cache is not None:
            return self._usage_cache
        rows = []
        for pid in self.meta["player_id"]:
            b, _ = self.blended_player(pid)
            rows.append(b)
        usage = pd.DataFrame(rows).merge(
            self.meta[["player_id", "team", "player_name"]], on="player_id", how="left")
        usage = apply_roster_adjustments(usage, self.roster_ctx, cfg=self.cfg)
        self._usage_cache = usage.set_index("player_id")
        return self._usage_cache

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def build_features(self, player_id: str) -> dict:
        usage = self.team_adjusted_usage()
        if player_id not in usage.index:
            raise KeyError(f"unknown player_id {player_id}")
        pb = usage.loc[player_id].to_dict()
        pb["player_id"] = player_id

        ctx_row = self.roster_ctx[self.roster_ctx.player_id == player_id]
        ctx = ctx_row.iloc[0].to_dict() if len(ctx_row) else {}
        team = pb.get("team") or ctx.get("team")

        env = self.store.environment_for(team)
        if not env:
            # No line/matchup row for this team this week. The projection still
            # runs on league-neutral game context, but it must be labeled as
            # such rather than passed off as a real matchup projection.
            env = {"spread": 0.0, "total": 44.0, "implied_total": 22.0,
                   "dome": False, "wind_mph": 0.0, "precip_prob": 0.0,
                   "_missing": True}
        opponent = env.get("opponent")
        team_off = self.blended_team_offense(team)
        team_off_prior = self._row(self.prior_off, team)
        opp_def = self.blended_team_defense(opponent) if opponent else {}

        f = build_feature_row(pb, team_off, team_off_prior, opp_def, env, ctx, self.cfg)
        f["team"] = team
        f["opponent"] = opponent
        f["missing_game_environment"] = bool(env.get("_missing", False))
        f["player_name"] = pb.get("player_name", player_id)
        f["sack_rate"] = pb.get("sack_rate", 0.065)
        return f

    def project_player(self, player_id: str, markets=None,
                       lines: dict | None = None) -> dict:
        markets = markets or ["anytime_td", "passing_yards",
                              "rushing_yards", "receiving_yards"]
        f = self.build_features(player_id)
        pos = f.get("position", "WR")
        _, meta = self.blended_player(player_id)

        market_lines = self._lines_for(player_id)
        booked = {r.get("market") for r in market_lines}

        results, recompute = {}, {}
        for m in markets:
            if m == "anytime_td":
                results[m] = self.td_model.predict(f)
                recompute[m] = lambda f2: self.td_model.intensity(f2)["lambda_total"]
            elif m in self.yard_models:
                if m == "passing_yards" and pos != "QB":
                    continue
                if m == "receiving_yards" and pos == "QB":
                    continue
                model = self.yard_models[m]
                # skip a market with no posted line when the position does not
                # normally carry it or the projected volume is negligible
                if m not in booked:
                    if m == "rushing_yards" and pos not in ("RB", "QB"):
                        continue
                    opp = model.opportunity(f)["expected_opportunities"]
                    if opp < 1.5:
                        continue
                lns = (lines or {}).get(m)
                lns = [lns] if isinstance(lns, (int, float)) else lns
                results[m] = model.predict(f, lines=lns)
                recompute[m] = (lambda mm: lambda f2: (
                    mm.opportunity(f2)["expected_opportunities"]
                    * mm.efficiency(f2)["expected_yards_per_opp"]))(model)

        return assemble_projection_json(
            player_id=player_id, features=f, results=results,
            blend_meta=meta, market_lines=market_lines,
            season=self.season, week=self.week, recompute=recompute)

    def _lines_for(self, player_id: str) -> list[dict]:
        pl = self.store.prop_lines
        if not len(pl):
            return []
        sub = pl[(pl["player_id"] == player_id) & (pl["season"] == self.season)
                 & (pl["week"] == self.week)]
        return sub.to_dict("records")

    def project_slate(self, player_ids=None, markets=None) -> list[dict]:
        ids = player_ids if player_ids is not None else list(self.meta["player_id"])
        out = []
        for pid in ids:
            try:
                out.append(self.project_player(pid, markets=markets))
            except Exception as e:  # keep the slate running
                out.append({"player_id": pid, "error": f"{type(e).__name__}: {e}"})
        return out
