"""
fit_models.py -- Phase 1: validate a pipeline that fits a preference model to
pairwise segment labels.

Preference model (for a pair of segments A, B):

    P(A > B) = sigmoid( tau * [ a * (G_A - G_B) + b * (dV_A - dV_B) ] )

    G   = partial return of a segment (sum of rewards)
    dV  = V*(end_state) - V*(start_state)

    (a, b) = (1, 0)  -> partial return
    (a, b) = (1, 1)  -> regret   (same formula as the repo's get_gt_regret, but with correct V*)
    (a, b) = (0, 1)  -> outcome-only

Only the products tau*a and tau*b are identified (scaling (a,b) up and tau down
gives the same likelihood). So we fit the two weights w = (w_G, w_V) = tau*(a, b)
by maximum likelihood, then report the DIRECTION normalized so that
max(|a|, |b|) = 1, plus tau = max(|w_G|, |w_V|).

Recovery check: sample labels from each known model, fit, and confirm the
recovered direction is right.

V* comes from the repo's value_iteration, NOT its calc_value (which drops the
reward for entering a terminal state; see compute_value_table).

Run from the repo root:
    venv/bin/python fit_models.py            # MDP 101, 5000 pairs
    venv/bin/python fit_models.py --mdp 105 --n_pairs 20000
"""

import argparse
import gzip
import pickle
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.algorithms.rl_algos import value_iteration
from learn_advantage.env.generate_random_policies import calc_value  # only for the bug check

MDP_DIR = "data/input/random_MDPs/"

# The three hypotheses, as (a, b) directions.
MODELS = {
    "partial_return": (1.0, 0.0),
    "regret": (1.0, 1.0),
    "outcome_only": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# 1. Load an MDP and its segment pairs (mirrors advantage_learning_exp_handler.py)
# ---------------------------------------------------------------------------
def load_mdp(mdp_id):
    segs = np.load(gzip.GzipFile(f"{MDP_DIR}MDP_{mdp_id}all_trajs.npy.gz", "r"))
    gt_rew_vec = np.load(gzip.GzipFile(f"{MDP_DIR}MDP_{mdp_id}gt_rew_vec.npy.gz", "r"))
    with open(f"{MDP_DIR}MDP_{mdp_id}env.pickle", "rb") as f:
        env = pickle.load(f)
    # Same setup the repo's experiment handler does for random MDPs.
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)
    return env, gt_rew_vec, segs


# Minimal stand-in for the repo's argparse namespace: only the fields that
# get_extended_features reads.
def make_repo_args(preference_model="regret"):
    return SimpleNamespace(
        use_extended_SF=False,  # 6-dim reward features -> partial return
        generalize_SF=False,
        gamma=0.999,
        preference_model=preference_model,
        preference_assum=preference_model,
        include_actions=False,  # deterministic MDPs: use start/end-state regret
        keep_ties=True,  # keep every pair so labels stay aligned with our arrays
        n_prob_samples=1,  # one label per pair
    )


# ---------------------------------------------------------------------------
# 2. Compute G and dV for every segment
# ---------------------------------------------------------------------------
def compute_value_table(env, gt_rew_vec):
    """V*(x, y) for every grid cell, via the repo's value_iteration.

    NOTE: we deliberately do NOT use the repo's calc_value here. calc_value
    computes V* from successor features that omit the reward of the transition
    INTO a terminal state, so whenever the goal reward is nonzero its V* is
    ~goal_reward too low at every non-terminal cell (e.g. ~50 too low in the
    human-study delivery MDP). That makes "regret" = G + dV come out at about
    +50 for every segment that enters the goal, which is impossible (true regret
    is always <= 0). value_iteration gives the correct V*, with V*(terminal) = 0.
    See calc_value_offset() for a check.
    """
    V, _ = value_iteration(rew_vec=np.array(gt_rew_vec), gamma=0.999, env=env)
    return np.array(V)


def calc_value_offset(env, gt_rew_vec):
    """Max |V_value_iteration - V_calc_value| over non-terminal, non-blocked
    cells, to document the calc_value bug described above."""
    V = compute_value_table(env, gt_rew_vec)
    worst = 0.0
    first = True  # first calc_value call computes + caches; later calls read the cache
    for x in range(env.height):
        for y in range(env.width):
            if env.is_terminal(x, y) or env.is_blocked(x, y):
                continue
            v_calc = calc_value(not first, (x, y), gt_rew_vec, env)
            first = False
            worst = max(worst, abs(V[x, y] - v_calc))
    return worst


def compute_stats(env, gt_rew_vec, segs):
    """Returns G (N,2), dV (N,2), and ses (start/end states, N x 2 x 2 x 2)."""
    args = make_repo_args()
    _, all_r, all_ses, _ = get_extended_features(args, segs, env, gt_rew_vec, seg_length=3)
    G = np.array(all_r, dtype=float)  # (N, 2)
    ses = np.array(all_ses, dtype=int)  # (N, seg, start/end, xy)

    V = compute_value_table(env, gt_rew_vec)
    V_start = V[ses[:, :, 0, 0], ses[:, :, 0, 1]]  # (N, 2)
    V_end = V[ses[:, :, 1, 0], ses[:, :, 1, 1]]  # (N, 2)
    dV = V_end - V_start
    return G, dV, ses, V


def max_regret(G, dV):
    """Sanity check: G + dV is the (negated) regret of a segment, which can
    never be positive (no segment beats optimal). Returns its maximum, which
    should be ~0 (optimal segments). Small positive slack (<~0.2) can come from
    discounting conventions in G."""
    return float(np.max(G + dV))


# ---------------------------------------------------------------------------
# 3. The model and maximum-likelihood fitting
# ---------------------------------------------------------------------------
def features(G, dV):
    """Per-pair differences (A minus B): columns are [dG, d(dV)]."""
    return np.column_stack([G[:, 0] - G[:, 1], dV[:, 0] - dV[:, 1]])


def neg_log_lik(w, X, y):
    """Bernoulli negative log-likelihood of logistic model with logit X @ w.

    Uses log_expit for numerical stability. y may contain 0.5 for ties,
    which contributes half of each outcome's log-likelihood.
    """
    z = X @ w
    return -np.sum(y * log_expit(z) + (1 - y) * log_expit(-z))


def fit_free(X, y, n_inits=5, seed=0):
    """Fit both weights w = tau*(a, b) from several random inits.

    The NLL is convex in w, so all inits should land on the same optimum;
    we report the spread as a check.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_inits):
        w0 = rng.normal(scale=1.0, size=2)
        res = minimize(neg_log_lik, w0, args=(X, y), method="BFGS")
        results.append(res)
    best = min(results, key=lambda r: r.fun)
    spread = max(np.max(np.abs(r.x - best.x)) for r in results)
    return best.x, best.fun, spread


def fit_fixed_direction(X, y, direction):
    """Fit only tau for a fixed (a, b) direction (one free parameter)."""
    d = np.asarray(direction, dtype=float)
    x1 = X @ d  # scalar feature per pair
    res = minimize(lambda t: neg_log_lik(t, x1[:, None], y), x0=[0.1], method="BFGS")
    return res.x[0], res.fun


def to_direction(w):
    """Split w = tau*(a,b) into a normalized direction (max |.| = 1) and tau."""
    tau = np.max(np.abs(w))
    return w / tau, tau


def angle_deg(u, v):
    """Angle between two 2-D vectors, in degrees."""
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ---------------------------------------------------------------------------
# 4. Recovery check
# ---------------------------------------------------------------------------
def synth_labels(model_name, X, tau=1.0, rng=None):
    """Sample stochastic labels from the model itself:
    y ~ Bernoulli(sigmoid(tau * X @ (a, b))), with (a, b) = MODELS[model_name].

    (We don't use the repo's generate_synthetic_prefs here: for regret it calls
    get_gt_regret -> calc_value, which has the terminal-reward bug described in
    compute_value_table. With tau=1 this matches the repo's sigmoid labeling.)
    """
    rng = rng or np.random.default_rng()
    p_A = expit(tau * X @ np.array(MODELS[model_name]))
    return (rng.random(len(X)) < p_A).astype(float)  # 1 = A preferred


def info_criteria(nll, k, n):
    """AIC and BIC from a negative log-likelihood with k params and n data points."""
    return 2 * k + 2 * nll, k * np.log(n) + 2 * nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", type=int, default=101)
    parser.add_argument("--n_pairs", type=int, default=5000,
                        help="random subsample of segment pairs (0 = use all)")
    parser.add_argument("--seed", type=int, default=0)
    cli = parser.parse_args()

    np.random.seed(cli.seed)  # for the pair subsample

    # --- load + compute statistics
    env, gt_rew_vec, segs = load_mdp(cli.mdp)
    if cli.n_pairs and cli.n_pairs < len(segs):
        idx = np.random.choice(len(segs), cli.n_pairs, replace=False)
        segs = segs[idx]
    G, dV, ses, V = compute_stats(env, gt_rew_vec, segs)
    X = features(G, dV)

    print(f"MDP {cli.mdp}: {len(G)} segment pairs, grid {env.height}x{env.width}")
    print(f"gt_rew_vec = {gt_rew_vec}")
    print(f"G  shape {G.shape}, range [{G.min():.2f}, {G.max():.2f}]")
    print(f"dV shape {dV.shape}, range [{dV.min():.2f}, {dV.max():.2f}]")
    print(f"corr(dG, d(dV)) across pairs = {np.corrcoef(X.T)[0, 1]:+.3f}")
    print(f"max over segments of G + dV (should be ~0, never >> 0) = {max_regret(G, dV):.3f}")
    print(f"max |V*_value_iteration - V_calc_value| (repo calc_value bug if ~goal reward) "
          f"= {calc_value_offset(env, gt_rew_vec):.2f}")

    # --- recovery check
    rng = np.random.default_rng(cli.seed)
    print("\nRecovery check (labels sampled from each model's sigmoid, true tau=1)")
    header = (f"{'true model':<15} {'true (a,b)':<11} {'recovered (a,b)':<17} "
              f"{'tau':>6} {'angle err':>9} {'init spread':>11}   "
              f"{'AIC: PR / regret / outcome':<30} best")
    print(header)
    print("-" * len(header))
    all_ok = True
    for name, true_ab in MODELS.items():
        y = synth_labels(name, X, tau=1.0, rng=rng)
        w, _, spread = fit_free(X, y, seed=cli.seed)
        ab, tau = to_direction(w)
        err_deg = angle_deg(w, np.array(true_ab))

        # Compare the three fixed-direction models (1 param each: tau)
        aics = {}
        for m, d in MODELS.items():
            _, nll_m = fit_fixed_direction(X, y, d)
            aics[m] = info_criteria(nll_m, 1, len(y))[0]
        best = min(aics, key=aics.get)

        ok = err_deg < 10 and best == name
        all_ok &= ok
        aic_str = " / ".join(f"{aics[m]:.0f}" for m in MODELS)
        print(f"{name:<15} {str(true_ab):<11} ({ab[0]:+.3f}, {ab[1]:+.3f})  "
              f"{tau:6.3f} {err_deg:8.2f}° {spread:11.1e}   {aic_str:<30} "
              f"{best}{'' if ok else '  <-- MISMATCH'}")

    print("\nRECOVERY", "PASSED" if all_ok else "FAILED",
          "(criterion: direction within 10° of truth AND lowest-AIC fixed model is the true one)")


if __name__ == "__main__":
    main()
