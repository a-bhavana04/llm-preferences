"""
fit_llm.py -- fit the preference models to LLM-judge labels (from query_llm.py)
and compare them with the human labels on the SAME pairs.

Each pair was judged twice, once in each display order (AB and BA). Position
bias is handled by AVERAGING the two answers (in the A-frame, y_A = P(A preferred)):
    both say A        -> 1.0        both say B       -> 0.0
    A one way, B the other (full reversal)           -> 0.5
    A one way, "equal" the other                     -> 0.75   (etc.)
Averaged labels are used as soft targets, exactly like the human ties.
Pairs with an unparseable answer in either order are dropped.

Two reports per label file:
  1. Order-CONSISTENT pairs (same answer in both orders): share agreeing with the
     noiseless partial-return vs regret prediction, with 95% Wilson intervals.
     On discriminating pairs these two are complements (for non-"equal" answers).
  2. Model fits on the averaged labels, using the validated code from
     fit_human.py / fit_models.py (V* from value_iteration, not the buggy
     calc_value): NLL, AIC, BIC, 10-fold CV cross-entropy, free (a,b) direction,
     and a pair bootstrap of b/a. The same fits on the HUMAN labels for the same
     pairs are printed alongside for comparison.

Run from the repo root:
    venv/bin/python fit_llm.py data/llm_labels/llama3.1-8b_discall.jsonl
    venv/bin/python fit_llm.py data/llm_labels/*_disc60.jsonl --n_boot 200
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

from fit_models import MODELS, features, neg_log_lik, info_criteria, to_direction, angle_deg
from fit_human import fit_model, agreement, cv_cross_entropy, bootstrap_direction
from load_human_prefs import load_human_prefs

NAMES = ["random", "partial_return", "regret", "outcome_only", "free"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def fmt_share(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} = {k / n if n else float('nan'):.3f}  [{lo:.2f}, {hi:.2f}]"


def load_llm_labels(path):
    """Return {pair_idx: (y_A from AB, y_A from BA)}, keeping only fully parsed pairs."""
    by_pair = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_pair[r["pair_idx"]][r["order"]] = r["y_A"]
    return {i: (o["AB"], o["BA"]) for i, o in by_pair.items()
            if o.get("AB") is not None and o.get("BA") is not None}


def predictions(X):
    """Noiseless winner under each model: 1 = A, 0 = B, 0.5 = exact tie."""
    out = {}
    for m in ("partial_return", "regret"):
        z = np.round(X @ np.array(MODELS[m]), 9)
        out[m] = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))
    return out


def fit_table(X, y, label):
    """Fit all models to (X, y); print the table; return the rows."""
    N = len(y)
    rows = {}
    for name in NAMES:
        w, k = fit_model(name, X, y)
        nll = neg_log_lik(w, X, y)
        aic, bic = info_criteria(nll, k, N)
        cv, cv_sd = cv_cross_entropy(name, X, y)
        rows[name] = dict(w=w, k=k, nll=nll, aic=aic, bic=bic, cv=cv, cv_sd=cv_sd,
                          agree=agreement(w, X, y))
    best_aic = min(r["aic"] for r in rows.values())
    best_bic = min(r["bic"] for r in rows.values())
    print(f"\n  {label}: N = {N} pairs")
    print(f"  {'model':<15} {'k':>2} {'tau':>7} {'NLL':>7} {'dAIC':>7} {'dBIC':>7} "
          f"{'10-fold CV CE':>15} {'agree':>6}")
    for name in NAMES:
        r = rows[name]
        # Signed tau for fixed models: NEGATIVE means the labels run against that
        # model (e.g. "anti-regret"). For the free model, tau = max|w| (always > 0).
        if r["k"] == 0:
            tau = "-"
        elif name == "free":
            tau = f"{np.max(np.abs(r['w'])):.3f}"
        else:
            j = int(np.argmax(np.abs(MODELS[name])))
            tau = f"{r['w'][j] / MODELS[name][j]:+.3f}"
        print(f"  {name:<15} {r['k']:>2} {tau:>7} {r['nll']:7.1f} {r['aic'] - best_aic:7.1f} "
              f"{r['bic'] - best_bic:7.1f} {r['cv']:9.3f} ±{r['cv_sd']:.3f} {r['agree']:6.3f}")
    ab, tau = to_direction(rows["free"]["w"])
    angles = ", ".join(f"{m} {angle_deg(rows['free']['w'], np.array(d)):.1f}°"
                       for m, d in MODELS.items())
    print(f"  free direction (a, b) = ({ab[0]:+.3f}, {ab[1]:+.3f}), b/a = {ab[1] / ab[0]:+.3f}; "
          f"angle to {angles}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="label files written by query_llm.py")
    ap.add_argument("--n_boot", type=int, default=1000, help="pair-bootstrap refits of b/a (0 = skip)")
    args = ap.parse_args()

    d = load_human_prefs()
    X_all = features(d["G"], d["dV"])
    human_all = d["y"]

    for path in args.files:
        pairs = load_llm_labels(path)
        idx = np.array(sorted(pairs))
        ab = np.array([pairs[i] for i in idx])          # (n, 2): y_A in AB and BA order
        X, human = X_all[idx], human_all[idx]
        preds = predictions(X)
        print("=" * 90)
        print(f"{path}: {len(idx)} pairs with both orderings parsed")
        print("=" * 90)

        # ---- 1. order-consistent pairs -------------------------------------------
        cons = ab[:, 0] == ab[:, 1]
        y_c = ab[cons, 0]
        strict = y_c != 0.5                                # consistent and not "equal"
        print(f"Order-consistent pairs: {cons.sum()}/{len(idx)} "
              f"({np.sum(y_c == 0.5)} of them 'equal' both ways)")
        for m in ("partial_return", "regret"):
            ok = preds[m][cons][strict] == y_c[strict]
            print(f"  agree with {m:<15} {fmt_share(ok.sum(), strict.sum())}")
        h = human[cons][strict]
        ok = (h == y_c[strict])[h != 0.5]
        print(f"  agree with human labels      {fmt_share(ok.sum(), len(ok))}   (non-tied human labels)")
        hs = human != 0.5
        for m in ("partial_return", "regret"):
            okh = preds[m][hs] == human[hs]
            print(f"  [humans, all {hs.sum()} non-tied pairs here] agree with {m:<15} "
                  f"{fmt_share(okh.sum(), hs.sum())}")

        # ---- 2. model fits on averaged labels ------------------------------------
        y = ab.mean(axis=1)                                 # averaging both orderings
        if np.all(X[:, 0] > 0) and np.all(X.sum(axis=1) < 0):
            print("\nWARNING: on every pair here A has the higher partial return AND B the lower")
            print("regret, so 'partial return' = 'pick A' and 'regret' = 'pick B' throughout.")
            print("The fits below cannot separate partial return from anti-regret; use the")
            print("consistent-pair shares above, and fit on all 1,793 pairs instead.")
        print(f"\nAveraged labels: 1.0 = {np.mean(y == 1):.2f}, 0.0 = {np.mean(y == 0):.2f}, "
              f"0.5 = {np.mean(y == 0.5):.2f}, other (0.25/0.75) = {np.mean((y != 0) & (y != 1) & (y != 0.5)):.2f}")
        fit_table(X, y, "LLM labels (averaged over both orders)")
        fit_table(X, human, "HUMAN labels, same pairs")

        if args.n_boot:
            # Resample pairs (each pair = one cluster holding its averaged label).
            clusters = [np.array([i]) for i in range(len(y))]
            for who, labels in (("LLM", y), ("human", human)):
                ratios, _ = bootstrap_direction(X, labels, clusters, n_boot=args.n_boot)
                lo, hi = np.percentile(ratios, [2.5, 97.5])
                print(f"  pair bootstrap ({args.n_boot}x) of b/a, {who:<5}: 95% CI [{lo:+.3f}, {hi:+.3f}]"
                      f"   (partial return = 0, regret = 1)")
        print()


if __name__ == "__main__":
    main()
