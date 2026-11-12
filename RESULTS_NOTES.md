# Results notes: which preference model do humans follow?

Model: `P(A ≻ B) = sigmoid(τ·[a·(G_A − G_B) + b·(ΔV_A − ΔV_B)])`, where G is a segment's
partial return and ΔV = V\*(end) − V\*(start).
(a,b) = (1,0) partial return, (1,1) regret, (0,1) outcome-only. Only τ·(a,b) is identified.

Scripts (run from the repo root with `venv/bin/python`):

| script | what it does |
|---|---|
| `fit_models.py` | Phase 1: fits the model to synthetic labels on a random MDP and checks that it recovers the model that generated them |
| `load_human_prefs.py` | Loads the Knox et al. (2022) human labels into (G, ΔV, y) arrays |
| `fit_human.py` | Phase 2: fits all models to the human labels; prints AIC/BIC, cross-validated cross-entropy, agreement and a bootstrap |
| `verify_vstar_bug.py` | Shows the V\* bug and its effect (see the last section) |

## Phase 1: recovering known models (MDP_101, 5,000 pairs, true τ = 1)

| true model | true (a,b) | recovered (a,b) | angle error | lowest-AIC fixed model |
|---|---|---|---|---|
| partial return | (1, 0) | (1.000, 0.021) | 1.2° | partial return |
| regret | (1, 1) | (1.000, 0.982) | 0.5° | regret |
| outcome-only | (0, 1) | (−0.013, 1.000) | 0.8° | outcome-only |

Also passes on MDP_120, and on all 66,908 pairs of MDP_101.

## Phase 2: human labels (Knox et al. 2022, DOI 10.18738/T8/S4WTWR)

N = 1793 labels (19.7% ties, kept as soft 0.5 targets), 10×10 delivery task.

| model | k | τ | NLL | ΔAIC | ΔBIC | CE/label | 10-fold CV CE | agreement |
|---|---|---|---|---|---|---|---|---|
| random | 0 | – | 1242.8 | 468.1 | 457.1 | 0.693 | 0.693 | 0.500 |
| partial return | 1 | 0.059 | 1110.9 | 206.2 | 200.7 | 0.620 | 0.620 | 0.605 |
| **regret** | 1 | 0.234 | 1027.9 | 40.2 | 34.7 | 0.573 | **0.574** | 0.721 |
| outcome-only | 1 | 0.017 | 1225.4 | 435.2 | 429.7 | 0.683 | 0.684 | 0.636 |
| free (a,b,τ) | 2 | 0.257 | 1006.8 | 0.0 | 0.0 | 0.561 | 0.562 | 0.735 |

- **Reproduces Knox et al. (2022):** they report 10-fold CV cross-entropy of 0.57 for regret,
  0.62 for partial return and 0.69 for random; we get 0.574, 0.620 and 0.693.
- **Regret beats partial return decisively:** ΔNLL = 83 with the same number of parameters, i.e. a
  likelihood ratio of about e^83.
- **Free fit:** (a,b) = (1, 0.874). It is 3.9° from regret, 41° from partial return and 49° from
  outcome-only. Humans weight ΔV slightly less than pure regret does, and the free model beats
  regret by ΔAIC = 40.
- **Uncertainty on b/a (1,000 bootstrap refits each; `fit_human.py`):**

  | resampling unit | clusters | b/a 95% CI | fits closest to regret |
  |---|---|---|---|
  | individual labels | 1793 | [0.828, 0.910] | 100% |
  | unique segment pairs | 1147 | [0.827, 0.908] | 100% |
  | 50 contiguous blocks of ~36 labels (subject proxy) | 50 | [0.804, 0.921] | 100% |

  A true subject-level bootstrap is **not possible with the released data**: the files have no
  subject IDs (checked all 7 files and `MANIFEST.TXT`). The 50-block scheme only approximates
  subjects if the label file is ordered by subject, which we cannot verify, so treat it as a rough
  check. All three intervals exclude both 1 (pure regret) and 0 (partial return), and they differ
  little, so the conclusion does not depend on the resampling unit.
- Agreement = fraction of non-tied human labels where the model gives the human's choice
  probability > 0.5.

### Data caveats

- **Counts:** the dataset's PDF report says 1,812 preferences over 1,245 unique pairs; the
  downloaded files contain 1,793 labels over about 1,149 unique pairs (by G + start/end states).
- **Label direction:** the human labels are 0 = first segment preferred and 1 = second, the reverse of
  the repo's `[1,0]` = first preferred. The loader converts them to y = P(A preferred) = 1 − label.
- **No actions:** the dataset gives only start/end states per segment, not actions, so G comes from
  the provided reward features (discounted at γ = 0.999, as in the paper).

## Bug in the repo's V\* (`calc_value`): how to verify it

**What the buggy code does.** `learn_successor_feature_iter` in
`learn_advantage/algorithms/rl_algos.py` (lines 85–86) computes successor features ψ under the
optimal policy. When a transition ends the episode (`done=True`), it sets that transition's
contribution to zero:

```python
else:                                          # transition enters a terminal state
    psi_sas = np.zeros(env.feature_size)       # drops phi; should be  prob * phi
    psi_q = np.zeros(env.feature_size)         # drops phi; should be  phi
```

`calc_value` (`learn_advantage/env/generate_random_policies.py:114`) returns V\*(s) = ψ(s)·w, so it
misses the reward collected *on entering* a terminal state: +50 for the goal in the human-study
task. `get_gt_regret` (`learn_advantage/utils/utils.py`) uses `calc_value`, and so does
`generate_synthetic_prefs(preference_model="regret")`. By contrast, `value_iteration`
(`rl_algos.py:356`, `Q = reward` when `done`) keeps that reward and gives the correct V\*.

**Our fix.** We do not modify repo code. Our scripts take V\* from `value_iteration`
(`fit_models.compute_value_table`), which gives V\*(terminal) = 0.

**Verify it yourself:** `venv/bin/python verify_vstar_bug.py` (needs `data/human_prefs/`). Expected output:

```
1. max |V_value_iteration - V_calc_value| on non-terminal cells: 50.00   (goal reward = 50)

V* source                  max(G+dV)  #segs G+dV>1  regret 10-fold CV CE
value_iteration (fixed)         0.15             0           0.574 ±0.021
calc_value (buggy)             50.00           464           0.600 ±0.026
```

How to read it:
1. V from `calc_value` is too low by up to the goal reward (50) at every non-terminal cell.
2. G + ΔV equals −regret, so it can never be positive. With the buggy V, all 464 segments that
   enter the goal get G + ΔV ≈ +46 to +50, which is impossible. With the fix the max is 0.15, which
   is slack from the γ = 0.999 discounted features.
3. Regret-model 10-fold CV cross-entropy on the human labels is **0.600 with the buggy V vs 0.574
   fixed**, using the same folds and seed as `fit_human.py`. The paper reports 0.57, so its
   published human analysis matches the *fixed* V\*; the bug is in this repo's code path.
   (An earlier draft quoted 0.598, which was the in-sample value, not CV.)

**Scope.** It affects any MDP with a nonzero goal reward: MDP_101 shows an offset of 0.00 (goal
reward 0) and MDP_120 shows 50.00 (both print this from `fit_models.py --mdp N`).

**Proposed one-line repo fix (tested, not applied).** Replacing those two lines with
`psi_sas = prob * phi` and `psi_q = phi` brings the successor-feature V within 0.5 of
`value_iteration` everywhere on the delivery task, down from 50. The remaining ≤0.5 is in cells
next to the mud-coin cell (7,7), where the policy splits 50/50 between tied actions. Not
investigated further. A possibly related quirk: in `GridWorldEnv.get_reward_feature` the
`MUD_COIN` case appears twice, so the first copy (`pass`) always wins.
