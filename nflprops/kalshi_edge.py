"""
Join Kalshi market prices to model projections and compute edge.

THE KEY DETAIL: "X OR MORE"
---------------------------
Kalshi player props settle YES on "X or more", not on a half-point over/under.
KXNFLRECYDS-...-40 pays if the player gains 40 or more receiving yards.

Yardage is integer-valued, so:

    P(yes) = P(X >= 40) = P(X > 39.5)

Evaluating the model's continuous predictive distribution at 40.0 instead of
39.5 systematically understates every YES probability. The error is small per
market (roughly half a yard of probability mass, about 0.5 to 1.5 points) but
it is one-directional, so across a full slate it would bias every edge the
same way and manufacture phantom UNDER value. The correction below is
deliberate, not incidental.

For anytime TD, strike 1 means "1 or more touchdowns", which is exactly the
model's P(ATD), so no correction applies.

EDGE
----
    edge = p_model - p_market

Kalshi quotes in cents that already ARE probabilities (a YES contract settles
at $1), so the bid/ask midpoint needs no de-vig step -- unlike American
sportsbook odds, where the two sides sum to more than 100% and must be
normalized first. Using the midpoint is the equivalent of a de-vigged price.

Expected value uses the ASK for a YES buy (what you would actually pay), not
the midpoint, because you cannot transact at the midpoint:

    cost     = ask / 100
    payoff   = 1.0 on YES
    ev       = p_model * (1 - cost) - (1 - p_model) * cost
             = p_model - cost

    kelly    = (p_model - cost) / (1 - cost)      # binary contract
"""

from __future__ import annotations

from .config import KELLY_FRACTION, MIN_EDGE_TO_FLAG
from .kalshi import KalshiQuote


def kalshi_yes_probability(dist, market: str, strike: float) -> float:
    """
    Model probability that a Kalshi YES contract settles, given the market's
    predictive distribution. Applies the integer continuity correction for
    yardage markets.
    """
    if market == "anytime_td":
        # handled by the caller (ATD probability is not a yardage distribution)
        raise ValueError("use the model's p_anytime_td directly for anytime_td")
    threshold = float(strike) - 0.5   # "X or more" on an integer stat
    return float(dist.sf(threshold))


def evaluate_quote(p_model: float, quote: KalshiQuote,
                   min_edge: float = MIN_EDGE_TO_FLAG,
                   kelly_fraction: float = KELLY_FRACTION) -> dict:
    """
    Compare a model probability against one Kalshi market. Returns the block
    that gets merged into the dashboard JSON for that prop.
    """
    p_market = quote.implied_prob
    out = {
        "market_ticker": quote.ticker,
        "market_strike": quote.strike,
        "yes_bid": quote.yes_bid,
        "yes_ask": quote.yes_ask,
        "market_spread": None if quote.spread is None else round(quote.spread, 4),
        "market_volume": quote.volume,
        "market_open_interest": quote.open_interest,
        "market_implied_probability": None if p_market is None else round(p_market, 4),
        "model_probability": round(float(p_model), 4),
        "liquid": quote.is_liquid,
    }

    if p_market is None or not quote.is_liquid:
        # An illiquid or one-sided market implies almost nothing. Reporting an
        # "edge" against it would be noise dressed up as signal.
        out.update({"edge": None, "expected_value": None, "kelly": None,
                    "recommended_side": "no_market"})
        return out

    edge = float(p_model) - float(p_market)
    out["edge"] = round(edge, 4)

    # cost to actually transact: you pay the ask for YES, or (1 - bid) for NO.
    # Using the midpoint here would overstate every edge, since you cannot
    # actually trade at the midpoint.
    yes_cost = quote.yes_ask if quote.yes_ask is not None else 1.0
    no_cost = 1.0 - (quote.yes_bid if quote.yes_bid is not None else 0.0)

    ev_yes = float(p_model) - yes_cost
    ev_no = (1.0 - float(p_model)) - no_cost

    if ev_yes >= ev_no and edge > min_edge and yes_cost < 1.0:
        kelly = (float(p_model) - yes_cost) / (1.0 - yes_cost)
        out.update({
            "recommended_side": "yes",
            "expected_value": round(ev_yes, 4),
            "kelly": round(max(kelly, 0.0) * kelly_fraction, 4),
            "entry_price": quote.yes_ask,
        })
    elif ev_no > ev_yes and (-edge) > min_edge and no_cost < 1.0:
        kelly = ((1.0 - float(p_model)) - no_cost) / (1.0 - no_cost)
        out.update({
            "recommended_side": "no",
            "expected_value": round(ev_no, 4),
            "kelly": round(max(kelly, 0.0) * kelly_fraction, 4),
            "entry_price": round(no_cost, 4),
        })
    else:
        out.update({"recommended_side": "pass",
                    "expected_value": round(max(ev_yes, ev_no), 4),
                    "kelly": 0.0})
    return out


def td_probability_at_least(lam: float, k: int, kappa: float = 0.965) -> float:
    """
    P(at least k touchdowns) from the model's Poisson intensity.

    Kalshi lists TD markets at strikes 1, 2, 3 and 4 -- only strike 1 is the
    "anytime" market. Feeding an anytime probability into a 2+ TD contract
    would badly overprice it (a 25% anytime scorer is maybe 4% to score twice),
    so multi-TD strikes get their own tail probability instead of being
    silently mismatched or dropped.
    """
    import math
    lam = max(float(lam), 0.0) * kappa
    if k <= 0:
        return 1.0
    # P(X >= k) = 1 - sum_{i<k} e^-lam lam^i / i!
    cdf = 0.0
    term = math.exp(-lam)
    for i in range(k):
        if i > 0:
            term *= lam / i
        cdf += term
    return float(max(0.0, min(1.0, 1.0 - cdf)))


def attach_to_projection(proj: dict, quotes_by_player: dict,
                         dists_by_market: dict | None = None) -> dict:
    """
    Merge Kalshi evaluations into one player's projection dict (the nested
    per-player structure from ProjectionEngine.project_player).

    quotes_by_player: {player_id: [KalshiQuote, ...]}
    dists_by_market:  {market_name: YardageDistribution} for that player
    """
    pid = proj.get("player_id")
    quotes = quotes_by_player.get(pid, [])
    if not quotes:
        return proj

    for q in quotes:
        blk = proj.get("markets", {}).get(q.market)
        if blk is None:
            continue
        if q.market == "anytime_td":
            p_model = blk.get("probability")
            if p_model is None:
                continue
            evaluation = evaluate_quote(p_model, q)
        else:
            dist = (dists_by_market or {}).get(q.market)
            if dist is None:
                continue
            p_model = kalshi_yes_probability(dist, q.market, q.strike)
            evaluation = evaluate_quote(p_model, q)
        blk.setdefault("kalshi", []).append(evaluation)
    return proj
