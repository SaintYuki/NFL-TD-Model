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


def _json_safe(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def flatten(projections: list[dict]) -> list[dict]:
    rows = []
    for p in projections:
        if "error" in p:
            continue
        base = {
            "player_id": p["player_id"],
            "player_name": p["player_name"],
            "position": p["position"],
            "team": p["team"],
            "opponent": p["opponent"],
            "flags": p.get("flags", []),
        }
        for market, blk in p.get("markets", {}).items():
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
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default="docs/data")
    args = ap.parse_args()

    store = DataStore(args.root, args.season, args.week)
    engine = ProjectionEngine(store)
    projections = engine.project_slate()
    rows = flatten(projections)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "props.json"), "w") as fh:
        json.dump(_json_safe(rows), fh, indent=2, default=str)

    meta = {
        "season": args.season,
        "week": args.week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_players": len([p for p in projections if "error" not in p]),
        "n_rows": len(rows),
        "n_errors": len([p for p in projections if "error" in p]),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"wrote {len(rows)} rows for season {args.season} week {args.week} "
          f"to {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
