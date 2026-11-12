"""
load_human_prefs.py -- Phase 2, step 4: load the Knox et al. (2022) human
preference dataset (Texas Data Repository, DOI 10.18738/T8/S4WTWR) and turn it
into the same (G, dV, y) arrays that fit_models.py works with.

Expected files in data/human_prefs/ (the dataset's delivery_mdp folder):
    DELIVERY_MDP_env.pickle                              GridWorld for the 10x10 delivery task
    DELIVERY_MDP_gt_rew_vec.npy                          [-1, 50, -50, 1, -1, -2]
    DELIVERY_MDP_human_prefs.npy                         (N,)  0 = 1st seg preferred, 1 = 2nd, 0.5 = tie
    DELIVERY_MDP_segment_pair_features_regret_form.npy   (N, 2, 10): 6 reward features (discounted,
                                                          gamma=0.999) + [start_x, start_y, end_x, end_y]

Notes on the data (checked by running this file):
  * N = 1793 labels (the dataset's PDF report says 1812).
  * The segments' actions are NOT included (DELIVERY_MDP_trajs.npy holds only
    start/end states), so G is taken from the provided reward features:
    G = features . gt_rew_vec. These features are discounted with gamma=0.999,
    which is the convention the original paper used.
  * The label convention is the REVERSE of the repo's [1,0]=A preferred.
    We convert to y = P(A preferred) = 1 - pref.

Run from the repo root to print a summary:
    venv/bin/python load_human_prefs.py
"""

import pickle
import sys

import numpy as np

import learn_advantage.env.grid_world as grid_world
from fit_models import compute_value_table, features, max_regret

HUMAN_DIR = "data/human_prefs/"


def load_delivery_env():
    """Unpickle the delivery-task GridWorld.

    The pickle refers to the class as `grid_world.GridWorldEnv` (the module's
    old top-level name), so we alias that name to this repo's module first.
    (Checked with pickletools: the file only reconstructs a GridWorldEnv.)
    """
    sys.modules.setdefault("grid_world", grid_world)
    with open(HUMAN_DIR + "DELIVERY_MDP_env.pickle", "rb") as f:
        return pickle.load(f)


def load_human_prefs():
    """Returns a dict with:
        G   (N, 2)  partial return of each segment
        dV  (N, 2)  V*(end) - V*(start) of each segment
        y   (N,)    P(segment A preferred): 1, 0, or 0.5 for ties
        ses (N, 2, 2, 2) start/end states  [pair, segment, start/end, xy]
        env, gt_rew_vec, V (the V* table)
    """
    # Plain numeric arrays -> allow_pickle=False so loading can't run code.
    def load(name):
        return np.load(HUMAN_DIR + f"DELIVERY_MDP_{name}.npy", allow_pickle=False)

    gt_rew_vec = load("gt_rew_vec")
    prefs = load("human_prefs")
    feats = load("segment_pair_features_regret_form")  # (N, 2, 10)
    assert len(prefs) == len(feats), "labels and segment pairs don't line up"
    assert set(np.unique(prefs)) <= {0.0, 0.5, 1.0}

    env = load_delivery_env()

    # Partial return: reward features (first 6 columns) dotted with the reward vector.
    G = feats[:, :, :6] @ gt_rew_vec  # (N, 2)

    # Start/end states (last 4 columns) -> V*(end) - V*(start).
    ses = feats[:, :, 6:10].astype(int).reshape(-1, 2, 2, 2)
    V = compute_value_table(env, gt_rew_vec)
    V_start = V[ses[:, :, 0, 0], ses[:, :, 0, 1]]
    V_end = V[ses[:, :, 1, 0], ses[:, :, 1, 1]]
    dV = V_end - V_start

    # Dataset: 0 = first preferred, 1 = second. Ours: y = P(A preferred).
    y = 1.0 - prefs

    return dict(G=G, dV=dV, y=y, ses=ses, env=env, gt_rew_vec=gt_rew_vec, V=V)


def main():
    d = load_human_prefs()
    G, dV, y, ses = d["G"], d["dV"], d["y"], d["ses"]
    X = features(G, dV)
    N = len(y)

    print(f"Human preference dataset: {N} labels, grid {d['env'].height}x{d['env'].width}")
    print(f"gt_rew_vec = {d['gt_rew_vec']}")
    print(f"labels: A preferred {np.sum(y == 1)}, B preferred {np.sum(y == 0)}, "
          f"ties {np.sum(y == 0.5)} ({np.mean(y == 0.5):.1%})")
    n_unique = len(np.unique(np.column_stack([G.round(4), ses.reshape(N, -1)]), axis=0))
    print(f"unique segment pairs (by G + start/end states): {n_unique}")
    print(f"G  range [{G.min():.2f}, {G.max():.2f}],  dV range [{dV.min():.2f}, {dV.max():.2f}]")
    print(f"corr(dG, d(dV)) across pairs = {np.corrcoef(X.T)[0, 1]:+.3f}")
    print(f"max over segments of G + dV (should be ~0) = {max_regret(G, dV):.3f}")

    # How often each model's noiseless prediction agrees with the human (ties excluded)
    print("\nNoiseless agreement with human labels (non-tied labels; pairs the model calls a tie count as 0.5):")
    strict = y != 0.5
    for name, stat in [("partial_return", X[:, 0]), ("regret", X[:, 0] + X[:, 1]),
                       ("outcome_only", X[:, 1])]:
        s = np.round(stat[strict], 6)
        pred = np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))
        agree = np.mean(np.where(pred == 0.5, 0.5, pred == y[strict]))
        print(f"  {name:<15} {agree:.3f}   (model ties: {np.mean(pred == 0.5):.1%})")


if __name__ == "__main__":
    main()
