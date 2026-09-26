"""
Runs the full slate and writes two files the static site reads directly:

    docs/data/props.json   flat array, one row per player-market-line
    docs/data/meta.json    season, week, generated_at, counts

Kept separate from run_week.py because the site wants a flat table
(player, market, line, edge, side) while run_week.py's output is the full
nested per-player JSON used for auditing and calibration.

    python scripts/build_site_data.py --root data --season 2026 --week 1 \
        --out docs/data
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflprops import DataStore, ProjectionEngine  # noqa: E402
from nflprops.ladder import build_ladder, ladder_score, percentiles  # noqa: E402


def _json_safe(obj):
    """
    Recursively replace NaN/Infinity with None (JSON null).

    Python's json.dump writes NaN as the bare token `NaN` by default, which
    is NOT valid JSON per spec -- Python's own json.load tolerates it (so it
    round-trips silently and never surfaces as a bug locally), but a
    browser's JSON.parse() rejects it outright with a SyntaxError. That
    mismatch is exactly what broke the live site: the data "looked" fine
    every way it was inspected except the one way that mattered.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def compute_def_ranks(store) -> dict:
    """
    League rank (1 = toughest/best defense) by def_epa_per_play_allowed,
    blended the same way the model itself blends it, so the drawer's
    "opponent defense" context matches what the projection actually used.
    """
    from nflprops.projections import ProjectionEngine
    engine = ProjectionEngine(store)
    rows = []
    for team in store.player_meta()["team"].dropna().unique():
        d = engine.blended_team_defense(team)
        rows.append((team, d.get("def_epa_per_play_allowed", 0.0)))
    rows.sort(key=lambda r: r[1])  # most negative EPA allowed = best defense = rank 1
    return {team: i + 1 for i, (team, _) in enumerate(rows)}


def flatten(projections: list[dict], def_ranks: dict | None = None) -> list[dict]:
    def_ranks = def_ranks or {}
    rows = []
    for p in projections:
        if "error" in p:
            continue
        env = p.get("game_environment", {})
        drivers = p.get("key_drivers", [])
        base = {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "position": p["position"],
            "team": p["team"],
            "opponent": p["opponent"],
            "flags": p.get("flags", []),
            "spread": env.get("spread"),
            "total": env.get("total"),
            "implied_team_total": env.get("implied_team_total"),
            "opp_def_rank": def_ranks.get(p["opponent"]),
        }
        for market, blk in p.get("markets", {}).items():
            top_driver = next((d for d in drivers if d.get("market") == market), None)
            if market == "anytime_td":
                mk = blk.get("market")
                rows.append({
                    **base, "market": market, "line": None,
                    "projection": blk["expected_touchdowns"],
                    "probability": blk["probability"],
                    "fair_odds": blk["fair_odds_american"],
                    "book_odds": mk.get("book_odds") if mk else None,
                    "book": mk.get("book") if mk else None,
                    "edge": mk.get("edge_pct_points") if mk else None,
                    "recommended_side": mk.get("recommended_side") if mk else None,
                    "kelly": mk.get("kelly_recommended") if mk else None,
                    "top_driver": top_driver.get("note") if top_driver else None,
                    "attribution": blk.get("attribution"),
                    "td_distribution": blk.get("td_distribution"),
                })
            else:
                lines = blk.get("lines") or [{}]
                for ln in lines:
                    side = ln.get("recommended_side", "pass")
                    edge = (ln.get("edge_over_pct_points") if side == "over"
                            else ln.get("edge_under_pct_points"))
                    rows.append({
                        **base, "market": market, "line": ln.get("line"),
                        "projection": blk["projection"],
                        "median": blk["median"],
                        "ci_80": blk.get("ci_80"),
                        "probability": ln.get("p_over"),
                        "book": ln.get("book"),
                        "book_over_odds": ln.get("book_over_odds"),
                        "book_under_odds": ln.get("book_under_odds"),
                        "edge": edge,
                        "recommended_side": side,
                        "kelly": ln.get("kelly_recommended"),
                        "top_driver": top_driver.get("note") if top_driver else None,
                        "ladder": blk.get("ladder"),
                        "ladder_score": blk.get("ladder_score"),
                        "p_ceiling": blk.get("p_ceiling"),
                        "percentiles": blk.get("percentiles"),
                        "attribution": blk.get("attribution"),
                    })
    return rows


def load_kalshi(out_dir: str) -> dict:
    """
    Load a previously fetched kalshi.json and index it by (player_id, market).

    Each player-market can carry MANY Kalshi contracts (one per strike). The
    dashboard row shows a single market comparison, so pick the contract
    closest to a coin flip -- that is the strike carrying the most
    information about where the market actually projects the player, and it
    is also the most liquid part of the ladder. The full set is kept under
    `all_contracts` so the drawer can show the whole ladder.
    """
    path = os.path.join(out_dir, "kalshi.json")
    if not os.path.exists(path):
        return {}
    try:
        rows = json.load(open(path))
    except Exception:
        return {}
    by_key: dict = {}
    for r in rows:
        key = (r.get("player_id"), r.get("market"))
        by_key.setdefault(key, []).append(r)

    out = {}
    out["__fetched_at__"] = None
    for r in rows:
        if r.get("fetched_at"):
            out["__fetched_at__"] = r["fetched_at"]
            break
    for key, contracts in by_key.items():
        liquid = [c for c in contracts if c.get("liquid")]
        pool = liquid or contracts
        best = min(pool, key=lambda c: abs((c.get("market_implied_probability") or 0.5) - 0.5))
        out[key] = {"primary": best, "all_contracts": sorted(
            contracts, key=lambda c: (c.get("market_strike") or 0))}
    return out


def confidence_score(row: dict) -> dict:
    """
    0-100 confidence, built ONLY from things we actually measure. No vibes.

      role certainty   inverse of role_uncertainty (rho, depth volatility)
      sample           games of current-season data behind the projection
      market quality   liquid two-sided quote with a tight spread
      agreement        model and market not wildly apart (a huge gap has
                       historically meant a model role gap, not an edge)

    Deliberately NOT a function of edge size. Confidence answers "how much
    do we trust this number", which is a different question from "how big is
    the disagreement" -- conflating them would make every wrong projection
    look confident precisely when it is most wrong.
    """
    role = 1.0 - min(max(float(row.get("role_uncertainty") or 0.15), 0.0), 1.0)
    n_games = float(row.get("games_current_season") or 0)
    sample = min(n_games / 4.0, 1.0)

    spread = row.get("market_spread")
    if row.get("market_ticker") and spread is not None:
        market_q = 1.0 - min(float(spread) / 0.12, 1.0)
    elif row.get("market_ticker"):
        market_q = 0.5
    else:
        market_q = 0.35          # no market to check against

    edge = row.get("edge")
    if edge is None:
        agreement = 0.6
    else:
        agreement = 1.0 - min(abs(float(edge)) / 0.35, 1.0)

    score = 100.0 * (0.34 * role + 0.26 * sample
                     + 0.22 * market_q + 0.18 * agreement)
    grade = ("A+" if score >= 85 else "A" if score >= 75 else "B+" if score >= 65
             else "B" if score >= 55 else "C" if score >= 45 else "D")
    return {"confidence": round(score, 1), "confidence_grade": grade,
            "confidence_parts": {"role": round(role, 2), "sample": round(sample, 2),
                                 "market_quality": round(market_q, 2),
                                 "agreement": round(agreement, 2)}}


def bettable_tag(row: dict) -> str:
    """
    Heuristic playability tag.

    IMPORTANT: these thresholds are NOT backtested. No graded results exist
    yet, so they are reasoned defaults, not validated ones. Once
    grade_results.py has a few hundred settled contracts, replace them with
    the edge buckets that actually showed positive ROI.
    """
    side = row.get("recommended_side")
    if side == "model_role_gap":
        return "EXCLUDED"
    if side not in ("yes", "no"):
        return "NO EDGE"
    edge = abs(float(row.get("edge") or 0))
    conf = float(row.get("confidence") or 0)
    if not row.get("market_liquid"):
        return "ILLIQUID"
    if edge >= 0.10 and conf >= 65:
        return "STRONG"
    if edge >= 0.06 and conf >= 55:
        return "PLAYABLE"
    if edge >= 0.035:
        return "MONITOR"
    return "NO EDGE"


def attach_market(row: dict, kalshi_index: dict) -> dict:
    k = kalshi_index.get((row.get("player_id"), row.get("market")))
    if not k:
        return row
    p = k["primary"]
    row["market_ticker"] = p.get("market_ticker")
    row["market_strike"] = p.get("market_strike")
    row["market_implied_probability"] = p.get("market_implied_probability")
    row["model_probability"] = p.get("model_probability")
    row["yes_bid"] = p.get("yes_bid")
    row["yes_ask"] = p.get("yes_ask")
    row["market_liquid"] = p.get("liquid")
    row["market_implied_projection"] = p.get("market_implied_projection")
    row["divergence"] = p.get("divergence")
    # overwrite the placeholder edge/side/kelly with the real market numbers
    if p.get("edge") is not None:
        row["edge"] = p.get("edge")
        row["recommended_side"] = p.get("recommended_side")
        row["kelly"] = p.get("kelly")
        row["line"] = p.get("market_strike")
    try:
        from nflprops.market_divergence import ladder_divergence
        row["ladder_divergence"] = ladder_divergence(
            k["all_contracts"], row.get("projection") or 0.0)
    except Exception:
        pass
    row["kalshi_ladder"] = [
        {"strike": c.get("market_strike"),
         "market_prob": c.get("market_implied_probability"),
         "model_prob": c.get("model_probability"),
         "edge": c.get("edge"),
         "side": c.get("recommended_side"),
         "liquid": c.get("liquid")}
        for c in k["all_contracts"]
    ]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default="docs/data")
    args = ap.parse_args()

    store = DataStore(args.root, args.season, args.week)
    engine = ProjectionEngine(store)
    def_ranks = compute_def_ranks(store)

    # Only project offensive skill positions. Without this, every position
    # on a real roster (OL, DL, LB, DB, K, P, LS) gets run through the model
    # too -- and since none of those positions have a real target/rush
    # profile, they all regress to nearly the same generic baseline and
    # show up as dozens of near-identical "phantom" rows.
    skill_ids = engine.meta[engine.meta.position.isin(["QB", "RB", "WR", "TE"])]["player_id"].tolist()
    projections = engine.project_slate(skill_ids)
    rows = flatten(projections, def_ranks)

    # merge Kalshi market data if it has been fetched
    kalshi_index = load_kalshi(args.out)
    kalshi_fetched_at = kalshi_index.pop("__fetched_at__", None) if kalshi_index else None
    if kalshi_index:
        rows = [attach_market(r, kalshi_index) for r in rows]
        n_mkt = sum(1 for r in rows if r.get("market_ticker"))
        print(f"attached Kalshi market data to {n_mkt} rows", file=sys.stderr)

    # score confidence and playability before writing
    for r in rows:
        r.update(confidence_score(r))
        r["bettable"] = bettable_tag(r)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "props.json"), "w") as fh:
        json.dump(_json_safe(rows), fh, indent=2, default=str)

    meta = {
        "kalshi_fetched_at": kalshi_fetched_at,
        "season": args.season,
        "week": args.week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_players": len([p for p in projections if "error" not in p]),
        "n_rows": len(rows),
        "n_errors": len([p for p in projections if "error" in p]),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Compact history snapshot, one file per week, for the site's trend
    # sparklines. Real data that accumulates week over week -- there is
    # only ever one or two points early in a season, and the site should
    # just show fewer points rather than fabricate a longer trend.
    hist_dir = os.path.join(args.out, "history")
    os.makedirs(hist_dir, exist_ok=True)
    compact = [
        {"player_id": r["player_id"], "market": r["market"],
         "projection": r["projection"], "line": r.get("line"),
         "probability": r.get("probability")}
        for r in rows
    ]
    with open(os.path.join(hist_dir, f"{args.season}_wk{args.week}.json"), "w") as fh:
        json.dump(_json_safe(compact), fh, separators=(",", ":"))

    index_path = os.path.join(hist_dir, "index.json")
    weeks = []
    if os.path.exists(index_path):
        try:
            weeks = json.load(open(index_path))
        except Exception:
            weeks = []
    if args.week not in weeks:
        weeks.append(args.week)
    weeks = sorted(set(weeks))
    with open(index_path, "w") as fh:
        json.dump(weeks, fh)

    print(f"wrote {len(rows)} rows for season {args.season} week {args.week} "
          f"to {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
