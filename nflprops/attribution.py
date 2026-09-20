"""
Projection attribution: show HOW the number was built, not just the range.

A range says "somewhere between 34 and 158". It does not say why 90 is the
central estimate rather than 60 or 120. This module decomposes a projection
into the chain that produced it, with each adjustment expressed in YARDS so
the pieces visibly sum to the total.

The chain for receiving yards:

    baseline    league-average yards/target x projected targets
    + role      this player's yards/target vs league average
    + matchup   opponent pass defense blended with the specific defender
    + QB        quarterback change since the prior numbers were earned
    + weather   wind / precipitation
    + depth     air-yards profile (deep threat vs possession)
    = projection

Each step is computed by holding everything else fixed and toggling that one
factor, so the attribution reflects the model's actual arithmetic rather than
a plausible-sounding story told after the fact.
"""

from __future__ import annotations

from .config import LEAGUE_BASELINE
from .util import safe_num


def _fmt(x, nd=1):
    return round(float(x), nd)


def attribute_receiving(f: dict, result: dict) -> dict:
    """Decompose a receiving-yards projection into yardage contributions."""
    opp = result["opportunity"]
    eff = result["efficiency"]
    targets = safe_num(opp.get("expected_opportunities"), 0.0)
    final_ypt = safe_num(eff.get("expected_yards_per_opp"), 0.0)

    lg_ypt = LEAGUE_BASELINE["yptarget"]
    base_ypt = safe_num(f.get("yptarget"), lg_ypt)   # blended, pre-adjustment

    combined_matchup = safe_num(eff.get("combined_matchup"), 1.0)
    qb_mult = safe_num(eff.get("qb_efficiency_mult"), 1.0)
    weather = safe_num(eff.get("weather_penalty"), 1.0)
    ay_corr = safe_num(eff.get("air_yards_correction"), 1.0)

    # walk the multiplier chain, converting each step into yards
    y0 = targets * lg_ypt                      # league-average receiver
    y1 = targets * base_ypt                    # this player's own efficiency
    y2 = y1 * combined_matchup
    y3 = y2 * qb_mult
    y4 = y3 * weather
    y5 = y4 * ay_corr

    steps = [
        {"label": "Baseline (league avg yds/target)",
         "detail": f"{_fmt(targets,2)} targets x {_fmt(lg_ypt,2)} yds",
         "yards": _fmt(y0), "delta": None},
        {"label": "Player efficiency",
         "detail": f"his {_fmt(base_ypt,2)} yds/target vs league {_fmt(lg_ypt,2)}",
         "yards": _fmt(y1), "delta": _fmt(y1 - y0)},
        {"label": "Matchup",
         "detail": (f"opp pass D + specific defender "
                    f"({_fmt(f.get('defender_allowed_ypt', 7.0),1)} yds/tgt allowed), "
                    f"x{_fmt(combined_matchup,3)}"),
         "yards": _fmt(y2), "delta": _fmt(y2 - y1)},
        {"label": "Quarterback",
         "detail": (f.get("qb_change_detail") or "no change")
                   + f", x{_fmt(qb_mult,3)}",
         "yards": _fmt(y3), "delta": _fmt(y3 - y2)},
        {"label": "Weather",
         "detail": f"x{_fmt(weather,3)}",
         "yards": _fmt(y4), "delta": _fmt(y4 - y3)},
        {"label": "Target depth profile",
         "detail": f"air-yards share vs target share, x{_fmt(ay_corr,3)}",
         "yards": _fmt(y5), "delta": _fmt(y5 - y4)},
    ]

    return {
        "market": "receiving_yards",
        "opportunity": {
            "projected_targets": _fmt(targets, 2),
            "team_pass_attempts": _fmt(opp.get("team_pass_attempts"), 1),
            "target_share": _fmt(opp.get("target_share"), 4),
        },
        "efficiency": {
            "yards_per_target": _fmt(final_ypt, 2),
            "league_average": _fmt(lg_ypt, 2),
        },
        "steps": steps,
        "projection": _fmt(result.get("projection")),
    }


def attribute_rushing(f: dict, result: dict) -> dict:
    opp = result["opportunity"]
    eff = result["efficiency"]
    carries = safe_num(opp.get("expected_opportunities"), 0.0)
    lg_ypc = LEAGUE_BASELINE["ypc"]
    base_ypc = safe_num(f.get("ypc"), lg_ypc)
    matchup = safe_num(eff.get("matchup_mult"), 1.0)
    unit = safe_num(eff.get("unit_delta"), 0.0)

    y0 = carries * lg_ypc
    y1 = carries * base_ypc
    y2 = y1 * matchup
    y3 = y2 + carries * unit

    steps = [
        {"label": "Baseline (league avg yds/carry)",
         "detail": f"{_fmt(carries,2)} carries x {_fmt(lg_ypc,2)} yds",
         "yards": _fmt(y0), "delta": None},
        {"label": "Player efficiency",
         "detail": f"his {_fmt(base_ypc,2)} yds/carry vs league {_fmt(lg_ypc,2)}",
         "yards": _fmt(y1), "delta": _fmt(y1 - y0)},
        {"label": "Matchup",
         "detail": f"opponent run defense, x{_fmt(matchup,3)}",
         "yards": _fmt(y2), "delta": _fmt(y2 - y1)},
        {"label": "Run blocking / continuity",
         "detail": f"{_fmt(unit,3)} yds/carry adjustment",
         "yards": _fmt(y3), "delta": _fmt(y3 - y2)},
    ]
    return {
        "market": "rushing_yards",
        "opportunity": {
            "projected_carries": _fmt(carries, 2),
            "team_carries": _fmt(opp.get("team_carries"), 1),
            "rush_share": _fmt(opp.get("rush_share"), 4),
        },
        "efficiency": {"yards_per_carry": _fmt(eff.get("expected_yards_per_opp"), 2),
                       "league_average": _fmt(lg_ypc, 2)},
        "steps": steps,
        "projection": _fmt(result.get("projection")),
    }


def attribute_passing(f: dict, result: dict) -> dict:
    opp = result["opportunity"]
    eff = result["efficiency"]
    att = safe_num(opp.get("expected_opportunities"), 0.0)
    lg_ypa = LEAGUE_BASELINE["ypa"]
    base_ypa = safe_num(f.get("ypa"), lg_ypa)
    matchup = safe_num(eff.get("matchup_mult"), 1.0)
    weather = safe_num(eff.get("weather_penalty"), 1.0)
    unit = safe_num(eff.get("unit_delta"), 0.0)

    y0 = att * lg_ypa
    y1 = att * base_ypa
    y2 = y1 * matchup
    y3 = y2 * weather
    y4 = y3 + att * unit

    steps = [
        {"label": "Baseline (league avg yds/attempt)",
         "detail": f"{_fmt(att,1)} attempts x {_fmt(lg_ypa,2)} yds",
         "yards": _fmt(y0), "delta": None},
        {"label": "QB efficiency",
         "detail": f"his {_fmt(base_ypa,2)} yds/attempt vs league {_fmt(lg_ypa,2)}",
         "yards": _fmt(y1), "delta": _fmt(y1 - y0)},
        {"label": "Matchup",
         "detail": f"opponent pass defense, x{_fmt(matchup,3)}",
         "yards": _fmt(y2), "delta": _fmt(y2 - y1)},
        {"label": "Weather", "detail": f"x{_fmt(weather,3)}",
         "yards": _fmt(y3), "delta": _fmt(y3 - y2)},
        {"label": "Pass protection / receivers",
         "detail": f"{_fmt(unit,3)} yds/attempt adjustment",
         "yards": _fmt(y4), "delta": _fmt(y4 - y3)},
    ]
    return {
        "market": "passing_yards",
        "opportunity": {
            "projected_attempts": _fmt(att, 1),
            "team_dropbacks": _fmt(opp.get("team_dropbacks"), 1),
            "dropback_share": _fmt(opp.get("dropback_share"), 4),
        },
        "efficiency": {"yards_per_attempt": _fmt(eff.get("expected_yards_per_opp"), 2),
                       "league_average": _fmt(lg_ypa, 2)},
        "steps": steps,
        "projection": _fmt(result.get("projection")),
    }


ATTRIBUTORS = {
    "receiving_yards": attribute_receiving,
    "rushing_yards": attribute_rushing,
    "passing_yards": attribute_passing,
}


def attribute(market: str, f: dict, result: dict) -> dict | None:
    fn = ATTRIBUTORS.get(market)
    return fn(f, result) if fn else None


def attribute_anytime_td(f: dict, td_result: dict) -> dict:
    """
    Why this TD probability. Shows the team scoring pool and the player's
    share of the high-value opportunities that actually produce touchdowns.
    """
    c = td_result.get("components", {})
    return {
        "market": "anytime_td",
        "team_pool": {
            "implied_team_total": _fmt(f.get("team_total_implied"), 1),
            "expected_offensive_td": _fmt(f.get("e_team_off_td"), 2),
            "expected_rushing_td": _fmt(f.get("e_team_rush_td"), 2),
            "expected_receiving_td": _fmt(f.get("e_team_rec_td"), 2),
        },
        "player_share": {
            "inside5_rush_share": _fmt(f.get("inside5_rush_share"), 3),
            "rz_rush_share": _fmt(f.get("rz_rush_share"), 3),
            "inside10_target_share": _fmt(f.get("inside10_target_share"), 3),
            "rz_target_share": _fmt(f.get("rz_target_share"), 3),
            "air_yards_share": _fmt(f.get("air_yards_share"), 3),
        },
        "lambda_rush": _fmt(c.get("lambda_rush"), 4),
        "lambda_rec": _fmt(c.get("lambda_rec"), 4),
        "lambda_total": _fmt(c.get("lambda_total"), 4),
        "probability": td_result.get("p_anytime_td"),
        "note": ("P(score) = 1 - exp(-lambda). Lambda is the team's expected "
                 "touchdowns multiplied by this player's share of the "
                 "goal-area work that produces them."),
    }
