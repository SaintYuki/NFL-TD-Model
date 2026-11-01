"""
Fit how much to trust the model versus the market, per market.

    python scripts/fit_blend.py --season 2026 --weeks 3,4,5,6 --root data

Fits, separately for each market,

    logit(P(yes)) = (1 - trust) * logit(p_market) + trust * logit(p_model)

with a single trust weight in [0, 1], so it can only shrink the model's
opinion toward the market and never invent edges from the market price.

on EVERY graded liquid contract (both sides, all strikes), not only the YES
picks. Fitting on the picks alone would bake the selection bias into the fit.

It then checks the fit honestly with leave-one-week-out validation: each
week is predicted by a blend fitted on the other weeks, and log loss is
compared against the market price alone and the raw model alone. A blend
that cannot beat the market out of sample is a sign the model adds nothing
in that market, and the report says so.

Writes data/blend_params.json, which fetch_kalshi.py applies automatically.
Refit weekly: the model itself has changed most weeks this season, so older
weeks describe older versions of it.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

_spec = importlib.util.spec_from_file_location("gr", os.path.join(HERE, "grade_results.py"))
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def collect(season, weeks, root, hist):
    actuals = gr.load_actuals(root, season)
    if not len(actuals) or "week" not in actuals.columns:
        print(f"no game logs under {root}/current -- run fetch_nflverse_data current first")
        sys.exit(1)
    kick = gr.load_kickoffs(season)
    recs = []
    for wk in weeks:
        contracts, n = gr.load_week_contracts(hist, season, wk, kick)
        if contracts is None:
            path = os.path.join(hist, f"kalshi_{season}_wk{wk}.json")
            if not os.path.exists(path):
                continue
            contracts = json.load(open(path))
        wkdf = actuals[actuals["week"] == wk]
        by_pid = {r["player_id"]: r for _, r in wkdf.iterrows()}
        for c in contracts:
            if not c.get("liquid") or c.get("recommended_side") == "model_role_gap":
                continue
            # fit on the RAW model probability; once blending is live, the
            # stored model_probability already contains the market
            pm = c.get("model_probability_raw", c.get("model_probability"))
            pk = c.get("market_implied_probability")
            if pm is None or pk is None:
                continue
            row = by_pid.get(c.get("player_id"))
            if row is None:
                continue
            y = gr.settle(c, row)
            if y is None:
                continue
            recs.append({"week": wk, "market": c.get("market"),
                         "p_model": float(pm), "p_mkt": float(pk), "y": int(y),
                         "cluster": f"{wk}|{c.get('player_id')}|{c.get('market')}"})
    return pd.DataFrame(recs)


LAMBDAS = np.linspace(0.0, 1.0, 201)


def predict(lam, pm, pk):
    return sigmoid((1.0 - lam) * logit(pk) + lam * logit(pm))


def fit(df):
    """Single trust weight in [0, 1] minimising log loss."""
    y = df.y.values
    lm, lk = logit(df.p_model), logit(df.p_mkt)
    best, best_ll = 0.0, None
    for lam in LAMBDAS:
        p = np.clip(sigmoid((1.0 - lam) * lk + lam * lm), 1e-4, 1 - 1e-4)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if best_ll is None or ll < best_ll:
            best, best_ll = float(lam), float(ll)
    return best


def logloss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_w_model(df, n_boot=200, seed=7):
    """Cluster bootstrap CI for w_model: strikes on one player settle together."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby("cluster")}
    keys = list(groups)
    ws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        bdf = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        if bdf.y.nunique() < 2:
            continue
        ws.append(fit(bdf))
    if len(ws) < 20:
        return None, None
    return float(np.percentile(ws, 2.5)), float(np.percentile(ws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True, help="comma list, e.g. 3,4,5,6")
    ap.add_argument("--root", default="data")
    ap.add_argument("--hist", default="docs/data/history")
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    df = collect(args.season, weeks, args.root, args.hist)
    if not len(df):
        print("no gradeable contracts found")
        sys.exit(1)
    print(f"graded contracts used: {len(df)} across {df.cluster.nunique()} clusters, "
          f"weeks {sorted(df.week.unique())}\n")

    out = {}
    print(f"{'market':16} {'n':>5} {'clust':>5}  {'trust':>6} {'95% CI':>14}  "
          f"{'CV logloss: market / model / blend':>36}  verdict")
    for mkt, g in df.groupby("market"):
        if g.y.nunique() < 2 or g.cluster.nunique() < 30:
            print(f"{mkt:16} too little data, skipped")
            continue
        lam = fit(g)
        lo, hi = bootstrap_w_model(g)

        # leave-one-week-out
        ll = {"market": [], "model": [], "blend": []}
        n_te = []
        for w in sorted(g.week.unique()):
            tr, te = g[g.week != w], g[g.week == w]
            if tr.y.nunique() < 2 or not len(te):
                continue
            p = fit(tr)
            ll["market"].append(logloss(te.y, te.p_mkt))
            ll["model"].append(logloss(te.y, te.p_model))
            ll["blend"].append(logloss(te.y, predict(p, te.p_model, te.p_mkt)))
            n_te.append(len(te))
        cv = {k: float(np.average(v, weights=n_te)) if v else None for k, v in ll.items()}

        # Require a real margin AND a trust weight distinguishable from zero.
        # A 0.0001 log-loss improvement is noise, not information.
        beats = (cv["blend"] is not None and cv["market"] is not None
                 and cv["market"] - cv["blend"] >= 0.002)
        if beats and lo is not None and lo > 0:
            verdict = "MODEL ADDS INFORMATION"
        elif beats:
            verdict = "suggestive, not established (trust CI includes 0)"
        else:
            verdict = "no demonstrated edge over the market"

        ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "n/a"
        cvs = (f"{cv['market']:.4f} / {cv['model']:.4f} / {cv['blend']:.4f}"
               if cv["blend"] is not None else "n/a")
        print(f"{mkt:16} {len(g):5} {g.cluster.nunique():5}  {lam:6.2f} {ci:>14}  {cvs:>36}  {verdict}")

        out[mkt] = {"lambda": lam, "lambda_ci": [lo, hi], "n": int(len(g)),
                    "n_clusters": int(g.cluster.nunique()),
                    "cv_logloss": cv, "verdict": verdict}

    print("\nHow to read it: trust is the share of the model's disagreement with the")
    print("market that survives (0 = ignore the model, 1 = use it unchanged). Blended")
    print("edges are roughly trust x raw edge, always with the same sign. In markets")
    print("where trust is ~0, edges shrink toward zero and stop producing plays.")

    path = os.path.join(args.root, "blend_params.json")
    json.dump({"season": args.season, "weeks": weeks, "markets": out},
              open(path, "w"), indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
