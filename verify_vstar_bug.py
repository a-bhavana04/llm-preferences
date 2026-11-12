"""
verify_vstar_bug.py -- independent check of the V* bug in the repo's calc_value.

Bug: learn_advantage/algorithms/rl_algos.py, learn_successor_feature_iter,
lines 85-86. When a transition ENDS the episode (done=True), the successor
feature is set to zero, dropping that transition's own reward features phi:

        else:
            psi_sas = np.zeros(env.feature_size)   # should be prob * phi
            psi_q   = np.zeros(env.feature_size)   # should be phi

calc_value (learn_advantage/env/generate_random_policies.py:114) computes
V*(s) = psi(s) . w from these features, so V* misses the reward collected on
entering a terminal state (+50 for the goal in the delivery task).
value_iteration (rl_algos.py:356, `Q = reward` when done) does it correctly.

This script prints, for the human-study delivery MDP:
  1. max |V*_value_iteration - V_calc_value| over non-terminal cells
  2. max of G + dV (true regret is never positive) under each V
  3. 10-fold CV cross-entropy of the regret model under each V
     (same folds/seed as fit_human.py)

Run from the repo root:   venv/bin/python verify_vstar_bug.py
"""

import numpy as np

from fit_models import features, compute_value_table, calc_value_offset
from fit_human import cv_cross_entropy
from load_human_prefs import load_human_prefs
from learn_advantage.env.generate_random_policies import calc_value


def calc_value_table(env, w):
    """V for every cell using the repo's (buggy) calc_value."""
    V = np.zeros((env.height, env.width))
    first = True  # first call computes + caches the successor features
    for x in range(env.height):
        for y in range(env.width):
            V[x, y] = calc_value(not first, (x, y), w, env)
            first = False
    return V


def main():
    d = load_human_prefs()  # uses value_iteration V* (correct)
    env, w, ses, G, y = d["env"], d["gt_rew_vec"], d["ses"], d["G"], d["y"]

    def dV_from(V):
        return V[ses[:, :, 1, 0], ses[:, :, 1, 1]] - V[ses[:, :, 0, 0], ses[:, :, 0, 1]]

    tables = {"value_iteration (fixed)": compute_value_table(env, w),
              "calc_value (buggy)": calc_value_table(env, w)}

    print(f"1. max |V_value_iteration - V_calc_value| on non-terminal cells: "
          f"{calc_value_offset(env, w):.2f}   (goal reward = {w[1]})\n")
    print(f"{'V* source':<26} {'max(G+dV)':>9} {'#segs G+dV>1':>13} {'regret 10-fold CV CE':>21}")
    for name, V in tables.items():
        dV = dV_from(V)
        R = G + dV
        cv, sd = cv_cross_entropy("regret", features(G, dV), y)
        print(f"{name:<26} {R.max():9.2f} {np.sum(R > 1):13d} {cv:15.3f} ±{sd:.3f}")


if __name__ == "__main__":
    main()
