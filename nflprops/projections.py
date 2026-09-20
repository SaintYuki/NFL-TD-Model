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
from .config import (ModelConfig, DEFAULT_CONFIG, EFFICIENCY_CREDIBILITY_K,
                     TEAM_VOLUME_PER_GAME)
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
        self.roster_ctx = build_roster_context(
            store.roster_changes, self.meta, cfg,
            depth_injury=getattr(store, "depth_injury", None))

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

        self._qb_change_cache = {}
        self._league_def_means = None
        self.td_model = AnytimeTDModel(cfg)
        self.yard_models = {k: cls(cfg) for k, cls in MODEL_REGISTRY.items()}
        self._usage_cache = None

        # How many games each team has already played this season. Needed so
        # that a player logging ZERO snaps while his team has played can be
        # treated as real evidence of replacement-level usage, rather than as
        # "no information" (which would leave him sitting on a prior-season
        # share earned in the handful of games he actually appeared in).
        self.team_games_played = {}
        if len(store.player_gamelog):
            gl = store.player_gamelog
            self.team_games_played = gl.groupby("team")["week"].nunique().to_dict()

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
        repbase = self.cfg.replacement.get(pos, self.cfg.replacement["WR"])

        # A player who has logged no snaps while his team has already played
        # is not an unknown -- he is an observed non-contributor. Count those
        # missed games as observations of replacement-level usage, so a
        # backup cannot coast on a share he earned in a couple of spot starts
        # last season.
        team = ctx.get("team")
        team_games = float(self.team_games_played.get(team, 0) or 0)
        inactive_evidence = team_games > 0 and n_cur == 0

        # RHO DECAY WITH EVIDENCE
        # ------------------------------------------------------------------
        # rho represents uncertainty about whether a player's prior profile
        # still describes him -- a team change, a new coordinator, a depth
        # chart move. Current-season snaps on the NEW team resolve exactly
        # that uncertainty, so rho must decay toward 1.0 as evidence arrives.
        #
        # Without this, Kenneth Walker (traded to KC, rho 0.55) took 23
        # carries for a 60.5% rush share in Week 1, and the blend still put
        # 34% of its weight on REPLACEMENT level -- returning 34.98%, below
        # both his prior (43.6%) and his observed new-team usage. A blend
        # should never land outside the range of its own inputs.
        RHO_EVIDENCE_K = 1.5
        if n_cur > 0:
            rho = 1.0 - (1.0 - rho) * (RHO_EVIDENCE_K / (n_cur + RHO_EVIDENCE_K))

        # Prior-season shares are computed over the games a player actually
        # appeared in, so they answer "what was his share WHEN ACTIVE" rather
        # than "what is his expected share in a random week". For a backup who
        # made two spot starts those are wildly different numbers. Shrink
        # opportunity shares toward replacement level by availability:
        #
        #   expected = share_when_active * avail + replacement * (1 - avail)
        #   avail    = games_played / 17
        #
        # Trade-off worth knowing: this also shrinks a STARTER who missed
        # most of last season to injury. Current-season data and the roster
        # context pull him back up as evidence arrives, but in Week 1 a
        # returning star will read low. Setting `returning_from_ir` in
        # roster_changes.csv raises his rho, which partly offsets it.
        prior_games = safe_num(prior.get("games"), 0.0)
        availability = float(np.clip(prior_games / 17.0, 0.0, 1.0)) if prior_games else 0.0
        # ...but ONLY when there is no current-season evidence. Availability
        # shrinkage exists to answer "will he be on the field?", and a snap
        # count from THIS season answers that far better than last season's
        # games-played ever could. Applying both double-counts, and does real
        # damage in the common case of a starter who missed time to injury:
        # Joe Burrow played 8 games last year, so shrinkage halved his prior
        # dropback share while simultaneously rewarding the backup who
        # replaced him -- leaving a healthy, Week-1-starting Burrow projected
        # at 21 attempts and 130 yards.
        apply_availability = (n_cur == 0)

        # Approximate how much opportunity the prior-season efficiency rests
        # on, so it can be regressed by CREDIBILITY rather than trusted flat.
        vol = TEAM_VOLUME_PER_GAME
        if pos == "QB":
            prior_opps = prior_games * vol["pass_att"] * safe_num(prior.get("pass_att_share"), 0.0)
            carry_opps = prior_games * vol["carries"] * safe_num(prior.get("rush_share"), 0.0)
        elif pos == "RB":
            prior_opps = prior_games * vol["targets"] * safe_num(prior.get("target_share"), 0.0)
            carry_opps = prior_games * vol["carries"] * safe_num(prior.get("rush_share"), 0.0)
        else:
            prior_opps = prior_games * vol["targets"] * safe_num(prior.get("target_share"), 0.0)
            carry_opps = prior_games * vol["carries"] * safe_num(prior.get("rush_share"), 0.0)
        opp_count = {"ypa": prior_opps if pos == "QB" else 0.0,
                     "ypc": carry_opps,
                     "yptarget": prior_opps,
                     "catch_rate": prior_opps,
                     "adot": prior_opps,
                     "sack_rate": prior_opps if pos == "QB" else 0.0}

        blended, weights = {}, {}
        for m in PLAYER_METRICS:
            # Opportunity metrics regress toward REPLACEMENT level, efficiency
            # metrics toward league average. An unknown player is assumed to
            # barely play, not to play like a starter -- but when he does
            # play, he gains yards at roughly a normal rate.
            if m in repbase:
                lg = repbase[m]
            else:
                lg = posbase.get(m, self.cfg.league.get(m, np.nan))
            if pd.isna(lg):
                lg = 0.0

            x_prior = prior.get(m)
            if (apply_availability and m in repbase and x_prior is not None
                    and not pd.isna(x_prior) and prior_games):
                x_prior = float(x_prior) * availability + lg * (1.0 - availability)

            # EFFICIENCY credibility regression: shrink toward the league mean
            # in proportion to how little opportunity the estimate rests on.
            # This is what stops a 48-attempt 12.1 YPA from surviving into a
            # league-leading projection.
            if (m in EFFICIENCY_CREDIBILITY_K and x_prior is not None
                    and not pd.isna(x_prior)):
                k_eff = EFFICIENCY_CREDIBILITY_K[m]
                n_opp = max(0.0, float(opp_count.get(m, 0.0)))
                lg_mean = self.cfg.league.get(m, posbase.get(m, lg))
                if lg_mean is not None and not pd.isna(lg_mean):
                    x_prior = ((float(x_prior) * n_opp + float(lg_mean) * k_eff)
                               / (n_opp + k_eff))

            x_cur, n_eff = cur.get(m), n_cur
            if inactive_evidence and m in repbase:
                # only opportunity metrics are affected; sitting on the bench
                # says nothing about how efficient he would be if he played
                x_cur, n_eff = lg, team_games
            elif m in repbase and n_cur > 0:
                # OPPORTUNITY CREDIBILITY: a game is not a fixed unit of
                # evidence. Chase seeing 4 targets and Chase seeing 14 both
                # counted as "one game", so a quiet afternoon moved an
                # established role as much as a heavy workload would. Scale
                # the current-season weight by observed volume relative to a
                # normal game for that role, capped so a genuine blowup still
                # counts fully.
                seen = safe_num(cur.get("targets_per_game") if m.endswith("target_share")
                                else cur.get("carries_per_game") if "rush" in m
                                else None, np.nan)
                expected_per_game = (
                    safe_num(prior.get("targets_per_game"), 5.0)
                    if m.endswith("target_share")
                    else safe_num(prior.get("carries_per_game"), 8.0))
                if not np.isnan(seen) and expected_per_game > 0:
                    # A heavy workload is MORE evidence than a normal one, not
                    # merely "one game". Walker's 23 carries is roughly two
                    # games of signal about his role; capping the ratio at 1.0
                    # threw that information away. Allow up to 2x, floor at
                    # 0.25 so a 2-target cameo still counts for little.
                    vol_ratio = float(np.clip(seen / expected_per_game, 0.25, 2.0))
                    n_eff = n_cur * vol_ratio

            blended[m] = blend_value(m, x_cur, x_prior, lg,
                                     n_eff, rho, self.week, self.cfg)
            if m in ("target_share", "rush_share", "ypc", "yptarget", "ypa"):
                weights[m] = blend_weights_report(m, n_eff, rho, self.week, self.cfg)

        blended["position"] = pos
        blended["player_id"] = player_id

        # NGS player traits (separation, CPOE, RYOE, ...) carried through as
        # descriptive features for the matchup engine. These are TRAITS, not
        # opportunity, so they are not blended or renormalized.
        ngs_tbl = getattr(self.store, "ngs", None)
        if ngs_tbl is not None and len(ngs_tbl):
            row = ngs_tbl[ngs_tbl["player_id"] == player_id]
            if len(row):
                for c, v in row.iloc[0].items():
                    if c != "player_id" and pd.notna(v):
                        blended[c] = v
        pfr_tbl = getattr(self.store, "pfr_player", None)
        if pfr_tbl is not None and len(pfr_tbl):
            row = pfr_tbl[pfr_tbl["player_id"] == player_id]
            if len(row):
                for c, v in row.iloc[0].items():
                    if c not in ("player_id", "pfr_id") and pd.notna(v):
                        blended[c] = v
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

        # real PFR pressure + coverage, replacing placeholder grades
        pd_def = getattr(self.store, "pfr_def", None)
        if pd_def is not None and len(pd_def):
            row = pd_def[pd_def["team"] == team]
            if len(row):
                r = row.iloc[0]
                for c in ("def_pressures", "def_blitzes", "def_hurries",
                          "def_qb_knockdowns", "def_cov_yds_per_target",
                          "def_cov_cmp_pct", "def_cov_rating"):
                    if c in r.index:
                        out[c] = r[c]
        cbs = getattr(self.store, "pfr_cbs", None)
        if cbs is not None and len(cbs):
            sub = cbs[cbs["team"] == team].sort_values("rank")
            for i, (_, r) in enumerate(sub.iterrows(), start=1):
                if i > 2:
                    break
                out[f"cb{i}_yds_tgt"] = r.get("yds_tgt")
                out[f"cb{i}_rating"] = r.get("rat")
                out[f"cb{i}_name"] = r.get("player")
        scheme = getattr(self.store, "team_scheme", None)
        if scheme is not None and len(scheme):
            row = scheme[scheme["team"] == team]
            if len(row):
                r = row.iloc[0]
                for c in ("def_blitz_rate", "def_avg_pass_rushers",
                          "def_avg_box", "def_heavy_box_rate"):
                    if c in r.index:
                        out[c] = r[c]
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
    def league_defense_means(self) -> dict:
        """
        Actual league means of the blended defensive metrics. Matchup
        multipliers are centered on these so they average 1.0, instead of on
        hardcoded constants that no longer match the data.
        """
        if self._league_def_means is not None:
            return self._league_def_means
        keys = ("def_epa_per_pass_allowed", "def_ypa_allowed",
                "def_epa_per_rush_allowed", "def_ypc_allowed")
        acc = {k: [] for k in keys}
        for t in self.meta["team"].dropna().unique():
            d = self.blended_team_defense(t)
            for k in keys:
                v = d.get(k)
                if v is not None and not pd.isna(v):
                    acc[k].append(float(v))
        self._league_def_means = {k: (float(np.mean(v)) if v else None)
                                  for k, v in acc.items()}
        return self._league_def_means

    def qb_change_for(self, team: str) -> dict:
        """Cached QB-continuity adjustment for a team's pass catchers."""
        if team in self._qb_change_cache:
            return self._qb_change_cache[team]
        from .qb_change import qb_change_multiplier
        try:
            res = qb_change_multiplier(
                team,
                self.store.player_season,
                self.store.player_gamelog,
                getattr(self.store, "depth_injury", None),
                getattr(self.store, "ngs", None))
        except Exception:
            res = {"qb_efficiency_mult": 1.0, "qb_changed": False}
        self._qb_change_cache[team] = res
        return res

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

        f = build_feature_row(pb, team_off, team_off_prior, opp_def, env, ctx,
                              self.cfg, league_means=self.league_defense_means())
        f["team"] = team
        f["opponent"] = opponent
        f["missing_game_environment"] = bool(env.get("_missing", False))
        f["player_name"] = pb.get("player_name", player_id)
        f["sack_rate"] = pb.get("sack_rate", 0.065)

        # QB continuity: a receiver's prior yards-per-target was earned with a
        # specific quarterback. If that changed, adjust.
        if f.get("position") in ("WR", "TE", "RB"):
            qc = self.qb_change_for(team)
            f["qb_efficiency_mult"] = qc.get("qb_efficiency_mult", 1.0)
            f["qb_changed"] = qc.get("qb_changed", False)
            f["qb_change_detail"] = qc.get("detail")
            f["current_qb"] = qc.get("current_qb")
            f["prior_qb"] = qc.get("prior_qb")
        else:
            f["qb_efficiency_mult"] = 1.0
        return f

    def project_player(self, player_id: str, markets=None,
                       lines: dict | None = None) -> dict:
        markets = markets or ["anytime_td", "passing_yards",
                              "rushing_yards", "receiving_yards"]
        f = self.build_features(player_id)
        pos = f.get("position", "WR")
        if pos not in ("QB", "RB", "WR", "TE"):
            # A non-skill position (OL, DL, LB, DB, K, P, LS, ...) has no
            # target/rush profile of its own. Silently falling through to
            # the WR baseline produces a plausible-looking but meaningless
            # number, which is worse than refusing -- callers that want a
            # full-roster sweep must filter to skill positions themselves.
            raise ValueError(
                f"{player_id} has position '{pos}', not a skill position "
                f"(QB/RB/WR/TE) -- prop projections do not apply.")
        _, meta = self.blended_player(player_id)

        market_lines = self._lines_for(player_id)
        booked = {r.get("market") for r in market_lines}

        results, recompute = {}, {}
        for m in markets:
            if m == "anytime_td":
                tdres = self.td_model.predict(f)
                try:
                    from .attribution import attribute_anytime_td
                    tdres["_attribution"] = attribute_anytime_td(f, tdres)
                except Exception:
                    pass
                results[m] = tdres
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
                res = model.predict(f, lines=lns)
                # ladder / outcome-distribution layer
                try:
                    from .ladder import build_ladder, ladder_score, percentiles
                    d = res.get("_dist")
                    if d is not None:
                        ls = ladder_score(d, m, f,
                                          res["opportunity"]["expected_opportunities"],
                                          res["projection"])
                        res["_ladder"] = {
                            "ladder": build_ladder(d, m),
                            "ladder_score": ls["ladder_score"],
                            "p_ceiling": ls["p_ceiling"],
                            "p_ceiling_vs_typical": ls["p_ceiling_vs_typical"],
                            "percentiles": percentiles(d),
                        }
                except Exception:
                    pass
                try:
                    from .attribution import attribute
                    res["_attribution"] = attribute(m, f, res)
                except Exception:
                    pass
                results[m] = res
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
