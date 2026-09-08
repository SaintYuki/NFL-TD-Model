"""
Output schema and assembly.

One JSON object per player-week. Contains the projection, the full predictive
distribution summary, probabilities against every posted line, the market edge,
and a ranked list of the drivers that produced the number.

The "key_drivers" block is not decoration. Each entry carries a signed impact
in the units of the market, computed by ablation: re-run the deterministic
projection with that one multiplier reset to neutral and record the difference.
That gives an auditable answer to "why is this number 12 yards above the book."
"""

from __future__ import annotations

import json
import numpy as np

from .probability import edge_and_stake, devig_two_way, line_probabilities

JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "NFLPlayerPropProjection",
    "type": "object",
    "required": ["player_id", "season", "week", "team", "opponent", "markets"],
    "properties": {
        "player_id": {"type": "string"},
        "player_name": {"type": "string"},
        "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE"]},
        "season": {"type": "integer"},
        "week": {"type": "integer"},
        "team": {"type": "string"},
        "opponent": {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "model_version": {"type": "string"},
        "game_environment": {
            "type": "object",
            "properties": {
                "spread": {"type": "number"},
                "total": {"type": "number"},
                "implied_team_total": {"type": "number"},
                "projected_team_plays": {"type": "number"},
                "projected_team_pass_attempts": {"type": "number"},
                "projected_team_carries": {"type": "number"},
                "expected_team_offensive_td": {"type": "number"},
            },
        },
        "data_blend": {
            "type": "object",
            "description": "How prior-season, current-season, and league data were weighted.",
            "properties": {
                "games_current_season": {"type": "number"},
                "continuity_rho": {"type": "number"},
                "role_uncertainty": {"type": "number"},
                "weights_by_metric": {"type": "object"},
            },
        },
        "markets": {
            "type": "object",
            "properties": {
                "anytime_td": {
                    "type": "object",
                    "properties": {
                        "expected_touchdowns": {"type": "number"},
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "p_two_plus": {"type": "number"},
                        "fair_odds_american": {"type": "integer"},
                        "components": {"type": "object"},
                        "market": {"$ref": "#/$defs/market_comparison"},
                    },
                },
                "passing_yards": {"$ref": "#/$defs/yardage_market"},
                "rushing_yards": {"$ref": "#/$defs/yardage_market"},
                "receiving_yards": {"$ref": "#/$defs/yardage_market"},
            },
        },
        "key_drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["feature", "value", "direction"],
                "properties": {
                    "feature": {"type": "string"},
                    "value": {"type": "number"},
                    "impact": {"type": "number",
                               "description": "Signed effect in market units."},
                    "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "note": {"type": "string"},
                },
            },
        },
        "flags": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "yardage_market": {
            "type": "object",
            "properties": {
                "projection": {"type": "number"},
                "median": {"type": "number"},
                "sd": {"type": "number"},
                "distribution": {"type": "string", "enum": ["normal", "gamma"]},
                "p_zero": {"type": "number"},
                "ci_50": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "ci_80": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "expected_opportunities": {"type": "number"},
                "expected_yards_per_opportunity": {"type": "number"},
                "lines": {"type": "array", "items": {"$ref": "#/$defs/line_eval"}},
            },
        },
        "line_eval": {
            "type": "object",
            "properties": {
                "book": {"type": "string"},
                "line": {"type": "number"},
                "p_over": {"type": "number"},
                "p_under": {"type": "number"},
                "fair_over_odds": {"type": "integer"},
                "fair_under_odds": {"type": "integer"},
                "book_over_odds": {"type": "integer"},
                "book_under_odds": {"type": "integer"},
                "market_devig_prob_over": {"type": "number"},
                "edge_over_pct_points": {"type": "number"},
                "edge_under_pct_points": {"type": "number"},
                "recommended_side": {"type": "string", "enum": ["over", "under", "pass"]},
                "kelly_recommended": {"type": "number"},
            },
        },
        "market_comparison": {
            "type": "object",
            "properties": {
                "book": {"type": "string"},
                "book_odds": {"type": "integer"},
                "book_implied_prob": {"type": "number"},
                "edge_pct_points": {"type": "number"},
                "recommended_side": {"type": "string"},
                "kelly_recommended": {"type": "number"},
            },
        },
    },
}

MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Driver attribution
# ---------------------------------------------------------------------------
DRIVER_SPECS = {
    "passing_yards": [
        ("pass_matchup_mult", 1.0, "Opponent pass defense (EPA + YPA allowed)"),
        ("proj_team_dropbacks", None, "Projected team dropback volume"),
        ("ypa_unit_delta", 0.0, "OL pass block + receiver corps change"),
        ("weather_penalty", 1.0, "Weather"),
        ("pass_rate_script_adj", 0.0, "Game script from the spread"),
    ],
    "rushing_yards": [
        ("rush_matchup_mult", 1.0, "Opponent run defense"),
        ("rush_share", None, "Projected carry share"),
        ("ypc_unit_delta", 0.0, "OL run block change"),
        ("rush_share_script_adj", 0.0, "Game script from the spread"),
        ("roster_usage_mult", 1.0, "Roster/depth chart change"),
    ],
    "receiving_yards": [
        ("target_share", None, "Projected target share"),
        ("matchup_eff_mult", 1.0, "WR/CB matchup"),
        ("pass_matchup_mult", 1.0, "Opponent pass defense"),
        ("yptarget_unit_delta", 0.0, "QB quality change"),
        ("roster_usage_mult", 1.0, "Roster/depth chart change"),
        ("weather_penalty", 1.0, "Weather"),
    ],
    "anytime_td": [
        ("e_team_off_td", None, "Implied team total to expected TDs"),
        ("inside5_rush_share", None, "Goal line carry share"),
        ("rz_target_share", None, "Red zone target share"),
        ("rush_td_share_of_off", None, "Team rush/pass TD split"),
        ("roster_usage_mult", 1.0, "Roster/depth chart change"),
    ],
}


def compute_drivers(market: str, features: dict, projection_value: float,
                    recompute_fn=None) -> list[dict]:
    """
    Ablation attribution: reset each multiplier to neutral, recompute, and
    report the delta. Falls back to reporting raw values when no recompute
    function is supplied.
    """
    specs = DRIVER_SPECS.get(market, [])
    drivers = []
    for name, neutral, note in specs:
        val = features.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        impact = None
        if recompute_fn is not None and neutral is not None:
            f2 = dict(features)
            f2[name] = neutral
            try:
                impact = float(projection_value - recompute_fn(f2))
            except Exception:
                impact = None
        if impact is None:
            direction = "neutral"
        else:
            direction = "positive" if impact > 0.15 else ("negative" if impact < -0.15 else "neutral")
        d = {"feature": name, "value": round(float(val), 4), "direction": direction, "note": note}
        if impact is not None:
            d["impact"] = round(impact, 2)
        drivers.append(d)
    drivers.sort(key=lambda d: abs(d.get("impact", 0.0)), reverse=True)
    return drivers[:6]


def _flags(features: dict) -> list[str]:
    out = []
    if features.get("role_uncertainty", 0) > 0.35:
        out.append("high_role_uncertainty")
    if features.get("continuity_rho", 1.0) < 0.55:
        out.append("major_roster_or_scheme_change")
    if features.get("depth_chart_volatility", 0) > 0.4:
        out.append("volatile_depth_chart")
    if features.get("n_games_current", 0) == 0:
        out.append("no_current_season_sample")
    if features.get("weather_penalty", 1.0) < 0.95:
        out.append("adverse_weather")
    if features.get("missing_game_environment"):
        out.append("no_game_environment_neutral_context")
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_projection_json(player_id, features, results, blend_meta,
                             market_lines, season, week,
                             min_edge=0.035, recompute=None) -> dict:
    from datetime import datetime, timezone

    recompute = recompute or {}
    markets_out, all_drivers = {}, []

    for market, res in results.items():
        book_rows = [r for r in market_lines if r.get("market") == market]

        if market == "anytime_td":
            m = {
                "expected_touchdowns": res["expected_touchdowns"],
                "probability": res["p_anytime_td"],
                "p_two_plus": res["p_two_plus_td"],
                "fair_odds_american": res["fair_odds"],
                "components": res["components"],
            }
            if book_rows:
                r = book_rows[0]
                p_dev, _ = devig_two_way(r.get("over_odds", -110), r.get("under_odds", 110))
                ev = edge_and_stake(res["p_anytime_td"], r.get("over_odds", -110), p_dev)
                side = "yes" if ev["edge_pct_points"] > min_edge else (
                    "no" if ev["edge_pct_points"] < -min_edge else "pass")
                m["market"] = {"book": r.get("book"), **ev, "recommended_side": side}
            markets_out[market] = m
            drivers = compute_drivers(market, features, res["expected_touchdowns"],
                                      recompute.get(market))
        else:
            dist = res.pop("_dist", None)
            m = {
                "projection": res["projection"],
                "median": res["median"],
                "sd": res["sd"],
                "distribution": res["distribution"],
                "p_zero": res["p_zero"],
                "ci_50": res.get("ci_50"),
                "ci_80": res.get("ci_80"),
                "expected_opportunities": res["opportunity"]["expected_opportunities"],
                "expected_yards_per_opportunity": res["efficiency"]["expected_yards_per_opp"],
                "opportunity_detail": res["opportunity"],
                "efficiency_detail": res["efficiency"],
                "lines": [],
            }
            for r in book_rows:
                line = float(r.get("line"))
                lp = line_probabilities(dist, line) if dist else {}
                over_odds = r.get("over_odds", -110)
                under_odds = r.get("under_odds", -110)
                p_dev_over, p_dev_under = devig_two_way(over_odds, under_odds)
                ev_o = edge_and_stake(lp["p_over"], over_odds, p_dev_over)
                ev_u = edge_and_stake(lp["p_under"], under_odds, p_dev_under)
                if ev_o["edge_pct_points"] > min_edge and ev_o["edge_pct_points"] >= ev_u["edge_pct_points"]:
                    side, kelly = "over", ev_o["kelly_recommended"]
                elif ev_u["edge_pct_points"] > min_edge:
                    side, kelly = "under", ev_u["kelly_recommended"]
                else:
                    side, kelly = "pass", 0.0
                m["lines"].append({
                    "book": r.get("book"),
                    **lp,
                    "book_over_odds": int(over_odds),
                    "book_under_odds": int(under_odds),
                    "market_devig_prob_over": round(p_dev_over, 4),
                    "edge_over_pct_points": ev_o["edge_pct_points"],
                    "edge_under_pct_points": ev_u["edge_pct_points"],
                    "recommended_side": side,
                    "kelly_recommended": kelly,
                })
            markets_out[market] = m
            drivers = compute_drivers(market, features, res["projection"],
                                      recompute.get(market))

        for d in drivers:
            d["market"] = market
        all_drivers.extend(drivers)

    return {
        "player_id": player_id,
        "player_name": features.get("player_name"),
        "position": features.get("position"),
        "season": int(season),
        "week": int(week),
        "team": features.get("team"),
        "opponent": features.get("opponent"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "game_environment": {
            "spread": features.get("spread"),
            "total": features.get("total"),
            "implied_team_total": round(float(features.get("team_total_implied", 0)), 2),
            "projected_team_plays": round(float(features.get("proj_team_plays", 0)), 1),
            "projected_team_pass_attempts": round(float(features.get("proj_team_pass_attempts", 0)), 1),
            "projected_team_carries": round(float(features.get("proj_team_carries", 0)), 1),
            "expected_team_offensive_td": round(float(features.get("e_team_off_td", 0)), 2),
        },
        "data_blend": {
            "games_current_season": blend_meta.get("n_games_current", 0),
            "continuity_rho": round(float(features.get("continuity_rho", 0)), 3),
            "role_uncertainty": round(float(features.get("role_uncertainty", 0)), 3),
            "weights_by_metric": blend_meta.get("blend_weights", {}),
        },
        "markets": markets_out,
        "key_drivers": all_drivers,
        "flags": _flags({**features, "n_games_current": blend_meta.get("n_games_current", 0)}),
    }


def write_json(objs, path: str):
    with open(path, "w") as fh:
        json.dump(objs, fh, indent=2, default=str)
    return path
