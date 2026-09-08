"""
Weekly projection run.

    python scripts/run_week.py --root data --season 2026 --week 1 \
        --out out/week1.json

    python scripts/run_week.py --root data --season 2026 --week 5 \
        --player KC_WR1 --pretty
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflprops import DataStore, ProjectionEngine, write_json  # noqa: E402
from nflprops.config import MIN_EDGE_TO_FLAG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--player", default=None, help="single player_id")
    ap.add_argument("--markets", default=None, help="comma separated")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--edges-only", action="store_true",
                    help="print only plays clearing the edge threshold")
    args = ap.parse_args()

    store = DataStore(args.root, args.season, args.week)
    missing = store.validate()
    for tbl, cols in missing.items():
        if cols and cols != ["<empty>"]:
            print(f"[warn] {tbl} missing columns: {cols}", file=sys.stderr)

    engine = ProjectionEngine(store)
    markets = args.markets.split(",") if args.markets else None

    if args.player:
        out = [engine.project_player(args.player, markets=markets)]
    else:
        out = engine.project_slate(markets=markets)

    ok = [o for o in out if "error" not in o]
    errs = [o for o in out if "error" in o]
    print(f"projected {len(ok)} players, {len(errs)} errors", file=sys.stderr)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_json(out, args.out)
        print(f"wrote {args.out}", file=sys.stderr)

    if args.edges_only:
        rows = []
        for o in ok:
            for m, blk in o.get("markets", {}).items():
                if m == "anytime_td":
                    mk = blk.get("market")
                    if mk and mk.get("recommended_side") in ("yes", "no"):
                        rows.append((o["player_name"], m, "-", mk["recommended_side"],
                                     mk["edge_pct_points"], mk["kelly_recommended"]))
                else:
                    for ln in blk.get("lines", []):
                        if ln["recommended_side"] != "pass":
                            edge = (ln["edge_over_pct_points"] if ln["recommended_side"] == "over"
                                    else ln["edge_under_pct_points"])
                            rows.append((o["player_name"], m, ln["line"],
                                         ln["recommended_side"], edge, ln["kelly_recommended"]))
        rows.sort(key=lambda r: -r[4])
        print(f"\n{'player':<12}{'market':<18}{'line':>7}  {'side':<6}{'edge':>7}{'kelly':>8}")
        for r in rows[:40]:
            print(f"{str(r[0]):<12}{r[1]:<18}{str(r[2]):>7}  {r[3]:<6}{r[4]:>7.3f}{r[5]:>8.3f}")
    elif args.pretty:
        print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))


if __name__ == "__main__":
    main()
