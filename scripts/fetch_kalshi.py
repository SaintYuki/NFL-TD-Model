"""
Fetch Kalshi NFL player-prop markets and merge them into the site data.

    # verify connectivity and credentials before trusting anything else
    python scripts/fetch_kalshi.py --selftest

    # pull one game's props (event ticker from the Kalshi URL)
    python scripts/fetch_kalshi.py --event KXNFLGAME-26SEP17DETBUF \
        --season 2026 --week 2 --root data --out docs/data

    # pull every NFL prop series for the week
    python scripts/fetch_kalshi.py --all-series --season 2026 --week 2 \
        --root data --out docs/data

Credentials (only needed for authenticated endpoints; market reads generally
work without them):
    export KALSHI_API_KEY_ID="..."
    export KALSHI_PRIVATE_KEY_PATH="~/.kalshi/private_key.pem"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflprops import DataStore, ProjectionEngine  # noqa: E402
from nflprops.kalshi import (  # noqa: E402
    KalshiClient, SERIES_TO_MARKET, normalize_markets, match_quotes_to_players,
    parse_ticker,
)
from nflprops.kalshi_edge import (  # noqa: E402
    evaluate_quote, kalshi_yes_probability, td_probability_at_least,
)
from nflprops.ladder import divergence_report  # noqa: E402


def selftest(client: KalshiClient):
    print(f"base_url       : {client.base_url}")
    print(f"authenticated  : {client.authenticated}")
    if not client.authenticated:
        print("  (no credentials found -- market reads may still work)")
    try:
        data = client.get("/markets", {"limit": 3, "status": "open"})
    except Exception as e:
        print(f"\nFAILED to reach the API: {type(e).__name__}: {e}")
        print("\nCheck, in order:")
        print("  1. network access to the Kalshi host")
        print("  2. KALSHI_API_BASE_URL -- Kalshi has changed hostnames before;")
        print("     confirm the current one at https://docs.kalshi.com")
        print("  3. key id / private key path, if the endpoint needs auth")
        return 1
    markets = data.get("markets", [])
    print(f"\nOK -- received {len(markets)} market(s). Sample:")
    for m in markets[:3]:
        print(f"  {m.get('ticker')}  yes_bid={m.get('yes_bid')} yes_ask={m.get('yes_ask')}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="print raw tickers and price fields, then exit")
    ap.add_argument("--event", default=None, help="Kalshi event ticker")
    ap.add_argument("--all-series", action="store_true",
                    help="pull every NFL player-prop series")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    client = KalshiClient(base_url=args.base_url) if args.base_url else KalshiClient()

    if args.selftest:
        sys.exit(selftest(client))

    if not (args.season and args.week):
        ap.error("--season and --week are required unless --selftest")

    # 1. fetch
    raw = []
    if args.event:
        raw = client.get_markets(event_ticker=args.event)
    elif args.all_series:
        for series in SERIES_TO_MARKET:
            try:
                raw.extend(client.get_markets(series_ticker=series))
            except Exception as e:
                print(f"[warn] series {series} failed: {e}", file=sys.stderr)
    else:
        ap.error("pass --event or --all-series")

    quotes = normalize_markets(raw)
    print(f"fetched {len(raw)} raw markets -> {len(quotes)} recognized player props",
          file=sys.stderr)

    if args.debug:
        print("\n--- RAW MARKETS (first 25) ---")
        for m in raw[:25]:
            t = m.get("ticker", "")
            print(f"\nticker: {t}")
            print(f"  parsed by our regex: {parse_ticker(t)}")
            # show every field that might carry a price, since field names
            # are the thing most likely to differ from the documented spec
            price_fields = {k: v for k, v in m.items()
                            if any(s in k.lower() for s in
                                   ("bid", "ask", "price", "last", "volume", "interest"))}
            print(f"  price-ish fields: {price_fields}")
            print(f"  all keys: {sorted(m.keys())}")
        return

    if not quotes:
        print("no player props recognized; nothing to merge", file=sys.stderr)
        return

    # 2. match to internal players
    store = DataStore(args.root, args.season, args.week)
    engine = ProjectionEngine(store)
    _pcols = ["player_id", "player_name", "team", "position"]
    if "jersey_number" in engine.meta.columns:
        _pcols.append("jersey_number")
    players = engine.meta[engine.meta.position.isin(["QB", "RB", "WR", "TE"])][
        _pcols].to_dict("records")
    mapping = match_quotes_to_players(quotes, players)
    unmatched = [q.ticker for q in quotes if q.ticker not in mapping]
    print(f"matched {len(mapping)}/{len(quotes)} tickers to players", file=sys.stderr)
    if unmatched:
        print(f"[warn] {len(unmatched)} unmatched (left out rather than guessed):",
              file=sys.stderr)
        for t in unmatched[:10]:
            print("   ", t, file=sys.stderr)

    # 3. evaluate each quote against the model
    by_player = {}
    for q in quotes:
        pid = mapping.get(q.ticker)
        if pid:
            by_player.setdefault(pid, []).append(q)

    evaluations = []
    for pid, qs in by_player.items():
        try:
            f = engine.build_features(pid)
        except Exception as e:
            print(f"[warn] {pid}: {e}", file=sys.stderr)
            continue
        for q in qs:
            try:
                dist, res_proj = None, None
                if q.market == "anytime_td":
                    td = engine.td_model.predict(f)
                    k = int(q.strike)
                    if k <= 1:
                        p_model = td["p_anytime_td"]
                    else:
                        # strikes 2/3/4 are MULTI-TD markets, not anytime
                        p_model = td_probability_at_least(
                            td["expected_touchdowns"], k)
                else:
                    model = engine.yard_models.get(q.market)
                    if model is None:
                        continue
                    mres = model.predict(f)
                    dist = mres["_dist"]
                    res_proj = mres["projection"]
                    p_model = kalshi_yes_probability(dist, q.market, q.strike)
                ev = evaluate_quote(p_model, q)
                ev["player_id"] = pid
                ev["market"] = q.market
                ev["player_name"] = f.get("player_name")
                ev["team"] = f.get("team")
                ev["opponent"] = f.get("opponent")

                # Market divergence diagnostic. For yardage markets the
                # Kalshi strike is a probability threshold, not a median, so
                # the comparable "market line" is the strike where the market
                # prices ~50%. That is approximated by inverting the model
                # distribution at the market's own implied probability, which
                # yields the yardage the market is effectively projecting.
                if q.market != "anytime_td" and dist is not None:
                    pm = q.implied_prob
                    if pm is not None and 0.02 < pm < 0.98:
                        try:
                            implied_line = float(dist.quantile(1.0 - pm))
                            ev["market_implied_projection"] = round(implied_line, 1)
                            ev["model_projection"] = res_proj
                            ev["divergence"] = divergence_report(
                                res_proj, implied_line, f,
                                0.0, 0.0, q.market)
                        except Exception:
                            pass
                evaluations.append(ev)
            except Exception as e:
                print(f"[warn] {q.ticker}: {type(e).__name__}: {e}", file=sys.stderr)

    # 4. write
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "kalshi.json")
    with open(path, "w") as fh:
        json.dump(evaluations, fh, indent=2, default=str)

    playable = [e for e in evaluations if e.get("recommended_side") in ("yes", "no")]
    print(f"wrote {len(evaluations)} evaluations ({len(playable)} actionable) to {path}",
          file=sys.stderr)
    for e in sorted(playable, key=lambda x: -abs(x.get("edge") or 0))[:15]:
        print(f"  {e['market']:16} {e['market_ticker']:46} "
              f"side={e['recommended_side']:4} edge={e['edge']:+.3f} kelly={e['kelly']:.3f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
