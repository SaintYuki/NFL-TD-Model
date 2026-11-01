"""
Blend the model's probability with the market's before computing edge.

Why this exists
---------------
Graded results through Week 6 showed the model's YES picks overconfident at
every probability level (Week 6: 41.5% predicted, 18.4% realized), and edge
size predicting nothing inside them. Meanwhile closing-line value was
positive two weeks running, so the model does carry information, just far
less than the raw edge column claims (about 0.3pp of market movement versus
claimed edges of 5-20pp).

Both facts point the same way: the model's opinion should be combined with
the market's, not substituted for it. This applies a per-market logistic
blend fitted by scripts/fit_blend.py:

    logit(p) = (1 - lambda) * logit(p_market) + lambda * logit(p_model)

lambda is a single trust weight in [0, 1]. A blended edge therefore always
has the same sign as the model's own edge and is a fraction of it, so the
blend can only shrink the model's opinion toward the market, never invent an
edge from patterns in the market price. (An earlier unconstrained version
did exactly that: with a free intercept and market weight, it "found"
passing and receiving edges while the model's own weight was near zero.)

If the model adds nothing in a market, lambda fits near zero, edges shrink
toward zero, and that market stops producing plays.

Contracts in the role-gap zone (model near 0 or 1 against a confident
market) are passed through RAW, so the role-gap guard in kalshi_edge still
sees them and excludes them instead of turning them into weak plays.
"""

from __future__ import annotations

import json
import math
import os

MIN_CLUSTERS_TO_APPLY = 100


def _logit(p: float) -> float:
    p = min(max(float(p), 0.01), 0.99)
    return math.log(p / (1.0 - p))


def load_blend_params(root: str = "data") -> dict:
    path = os.path.join(root, "blend_params.json")
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path)).get("markets", {})
    except Exception:
        return {}


def blend_probability(market: str, p_model, p_market, params: dict):
    """Return the blended probability, or the raw model value if no fit applies."""
    if p_model is None or p_market is None:
        return p_model
    pm, pk = float(p_model), float(p_market)
    # leave role-gap cases raw so the guard downstream can catch them
    if (pm < 0.03 and pk > 0.25) or (pm > 0.97 and pk < 0.75):
        return pm
    pr = params.get(market)
    if not pr or pr.get("n_clusters", 0) < MIN_CLUSTERS_TO_APPLY or "lambda" not in pr:
        # older unconstrained fits are ignored: they could invent edges from
        # patterns in the market price alone, which is not the model's edge
        return pm
    lam = min(max(float(pr["lambda"]), 0.0), 1.0)
    z = (1.0 - lam) * _logit(pk) + lam * _logit(pm)
    return 1.0 / (1.0 + math.exp(-z))
