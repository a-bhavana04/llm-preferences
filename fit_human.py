"""
fit_human.py -- Phase 2, step 5: fit the preference models to the Knox et al.
(2022) human labels and compare them.

Models (P(A > B) = sigmoid(tau * [a*dG + b*d(dV)])):
    random           P = 0.5 always                       0 free params
    partial_return   (a,b) = (1,0), tau fitted            1 free param
    regret           (a,b) = (1,1), tau fitted            1 free param
    outcome_only     (a,b) = (0,1), tau fitted            1 free param
    free             w = tau*(a,b) both fitted            2 free params

Reported per model:
    NLL, AIC, BIC                    on all labels (fit = evaluation set)
    CE / label                       mean cross-entropy on all labels
    10-fold CV CE                    mean held-out cross-entropy (compare with the
                                     paper: regret 0.57, partial return 0.62, random 0.69)
    agreement                        fraction of non-tied human labels where the
                                     model gives the human's choice prob > 0.5
                                     (pairs the model calls exactly 50/50 count as 0.5)

Ties (human label 0.5) are kept as soft targets: they add
0.5*log P(A) + 0.5*log P(B) to the log-likelihood.

Run from the repo root:
    venv/bin/python fit_human.py
"""

import numpy as np

from fit_models import (
    MODELS, features, fit_free, fit_fixed_direction, neg_log_lik,
    info_criteria, to_direction, angle_deg,
)
from load_human_prefs import load_human_prefs

N_FOLDS = 10
N_BOOT = 1000
SEED = 0


def fit_model(name, X, y):
    """Fit one model; return its weight vector w (logit = X @ w) and #params."""
    if name == "random":
        return np.zeros(2), 0
    if name == "free":
        w, _, _ = fit_free(X, y, seed=SEED)
        return w, 2
    tau, _ = fit_fixed_direction(X, y, MODELS[name])
    return tau * np.array(MODELS[name]), 1


def agreement(w, X, y):
    """Agreement with non-tied human labels under the model's prediction."""
    strict = y != 0.5
    z = np.round(X[strict] @ w, 9)
    pred = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))
    return np.mean(np.where(pred == 0.5, 0.5, (pred == y[strict]).astype(float)))


def cv_cross_entropy(name, X, y, n_folds=N_FOLDS, seed=SEED):
    """Mean held-out cross-entropy per label over k random folds."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_folds)
    ces = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        w, _ = fit_model(name, X[train], y[train])
        ces.append(neg_log_lik(w, X[test], y[test]) / len(test))
    return np.mean(ces), np.std(ces)


def bootstrap_direction(X, y, clusters, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap of the free model's direction.

    clusters: list of index arrays; each bootstrap draws len(clusters) clusters
    with replacement and refits on all labels in them.
    Returns b/a samples and angles (deg) to each hypothesis direction.
    """
    rng = np.random.default_rng(seed)
    ratios, angles = [], {m: [] for m in MODELS}
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[j] for j in pick])
        w, _, _ = fit_free(X[idx], y[idx], n_inits=1, seed=seed)
        ratios.append(w[1] / w[0])
        for m, d in MODELS.items():
            angles[m].append(angle_deg(w, np.array(d)))
    return np.array(ratios), {m: np.array(a) for m, a in angles.items()}


def main():
    d = load_human_prefs()
    X = features(d["G"], d["dV"])
    y = d["y"]
    N = len(y)
    print(f"Human labels: N = {N} (ties = {np.mean(y == 0.5):.1%}, kept as 0.5 soft targets)\n")

    names = ["random", "partial_return", "regret", "outcome_only", "free"]
    rows = {}
    for name in names:
        w, k = fit_model(name, X, y)
        nll = neg_log_lik(w, X, y)
        aic, bic = info_criteria(nll, k, N)
        cv_mean, cv_sd = cv_cross_entropy(name, X, y)
        rows[name] = dict(w=w, k=k, nll=nll, aic=aic, bic=bic,
                          ce=nll / N, cv=cv_mean, cv_sd=cv_sd, agree=agreement(w, X, y))

    best_aic = min(r["aic"] for r in rows.values())
    best_bic = min(r["bic"] for r in rows.values())
    hdr = (f"{'model':<15} {'k':>2} {'tau':>6} {'NLL':>8} {'AIC':>8} {'dAIC':>7} "
           f"{'BIC':>8} {'dBIC':>7} {'CE/label':>8} {'10-fold CV CE':>15} {'agree':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        r = rows[name]
        tau = "-" if r["k"] == 0 else f"{np.max(np.abs(r['w'])):.3f}"
        print(f"{name:<15} {r['k']:>2} {tau:>6} {r['nll']:8.1f} {r['aic']:8.1f} "
              f"{r['aic'] - best_aic:7.1f} {r['bic']:8.1f} {r['bic'] - best_bic:7.1f} "
              f"{r['ce']:8.3f} {r['cv']:9.3f} ±{r['cv_sd']:.3f} {r['agree']:6.3f}")

    # Likelihood-ratio style comparison of the two headline models (same k)
    d_nll = rows["partial_return"]["nll"] - rows["regret"]["nll"]
    print(f"\nRegret vs partial return: dNLL = {d_nll:.1f}  "
          f"(likelihood ratio e^{d_nll:.0f} in favor of "
          f"{'regret' if d_nll > 0 else 'partial return'})")
    print("Paper (Knox et al. 2022), 10-fold CV CE: regret 0.57, partial return 0.62, random 0.69")

    # Free model: where does the human direction point?
    ab, tau = to_direction(rows["free"]["w"])
    print(f"\nFree model: (a, b) = ({ab[0]:+.3f}, {ab[1]:+.3f}), tau = {tau:.3f}, "
          f"b/a = {ab[1] / ab[0]:.3f}")
    for m, dirn in MODELS.items():
        print(f"  angle to {m:<15} {angle_deg(rows['free']['w'], np.array(dirn)):6.1f}°")

    # The released data has NO subject IDs, so a true subject-level bootstrap is
    # impossible. We compare three resampling units instead:
    #   labels      - each label independent (too optimistic)
    #   pairs       - unique segment pairs (341 pairs were labelled 2-13 times)
    #   blocks      - 50 contiguous blocks of ~36 labels: a subject-level proxy that
    #                 is only right IF the file is ordered by subject (unverified)
    pair_key = np.round(np.concatenate([X, d["ses"].reshape(N, -1)], axis=1), 4)
    _, pair_id = np.unique(pair_key, axis=0, return_inverse=True)
    pair_id = pair_id.ravel()
    schemes = {
        "labels": [np.array([i]) for i in range(N)],
        "unique pairs": [np.where(pair_id == g)[0] for g in np.unique(pair_id)],
        "50 blocks (proxy)": np.array_split(np.arange(N), 50),
    }
    print(f"\nBootstrap of b/a ({N_BOOT}x). PR = 0, regret = 1, outcome-only = inf.")
    for name, clusters in schemes.items():
        ratios, angles = bootstrap_direction(X, y, clusters)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        nearest = np.argmin(np.column_stack([angles[m] for m in MODELS]), axis=1)
        closest = ", ".join(f"{m} {np.mean(nearest == i):.2f}" for i, m in enumerate(MODELS))
        print(f"  {name:<18} ({len(clusters):4d} clusters)  95% CI [{lo:.3f}, {hi:.3f}]"
              f"   closest-model fractions: {closest}")


if __name__ == "__main__":
    main()
