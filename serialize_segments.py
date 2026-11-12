"""
serialize_segments.py -- turn a human-study segment pair into text an LLM can judge.

Each segment = start cell + 3 moves in the 10x10 delivery task. The text has:
  1. TASK_RULES: the game's rules, as explained to the human subjects in
     Knox et al. 2022 (from the dataset's report). No per-segment totals are
     given, so the LLM must work out which segment is better itself.
  2. A map of the whole grid (one symbol per cell, with a legend).
  3. For each segment, a step-by-step list of moves and the cell entered.

Reconstructing the path: the dataset stores only each segment's start state,
end state, and DISCOUNTED reward features (gamma=0.999, so a coin collected on
step k contributes 0.999**(k-1)). We try all 4^3 = 64 action sequences from the
start and keep those whose end state and features match exactly, using the
repo's own find_reward_features / get_sa_list. Often several routes match
(e.g. right-then-down vs down-then-right over plain road); they then have the
same per-step rewards, start and end, so the same G and dV for every model.
We pick a canonical one (fewest bumps, then a fixed action order) and report
how many candidates there were.

Run from the repo root to print example serializations:
    venv/bin/python serialize_segments.py            # 3 default examples
    venv/bin/python serialize_segments.py 0 17 42    # specific pair indices
"""

import itertools
import sys

import numpy as np

from learn_advantage.env.grid_world import GridWorldEnv
from learn_advantage.utils.segment_feats_utils import find_reward_features, get_sa_list
from load_human_prefs import load_delivery_env, HUMAN_DIR

GAMMA = 0.999
DIRECTION_NAMES = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}

# Board codes (see get_state_feature in segment_feats_utils.py) ->
# (map symbol, description in words)
CELL_INFO = {
    0: (".", "white road"),
    1: ("G", "the DESTINATION"),
    2: ("H", "a house (cannot be entered)"),
    3: ("S", "a SHEEP"),
    4: ("c", "white road with a coin"),
    5: ("r", "white road with a roadblock"),
    6: ("b", "brick road"),
    7: ("G", "the DESTINATION (on brick)"),
    8: ("H", "a house (cannot be entered)"),
    9: ("S", "a SHEEP (on brick)"),
    10: ("C", "brick road with a coin"),
    11: ("R", "brick road with a roadblock"),
}

TASK_RULES = """\
You are judging a simple delivery game played on a 10x10 grid. A delivery vehicle moves one
cell per step: up, down, left, or right. The player's goal is to maximize the total reward.

Rewards for each step, based on the cell the vehicle moves into:
- White road: -1. Brick road: -2.
- A coin: +1 on top of the road cost. A roadblock: an extra -1 on top of the road cost.
- Reaching the destination: +50, and the game ends.
- Running into a sheep: -50, and the game ends.
- Trying to move into a house or off the edge of the grid: the vehicle stays where it is and
  pays only its current cell's road cost (no coin or roadblock reward).
Coins do not disappear after being collected."""

# Prompt-robustness variant: same game, but NO reward numbers, NO "maximize total
# reward" objective. Cells are described qualitatively (what a person would know
# about driving), so the judge has to decide for itself what "better" means.
NEUTRAL_RULES = """\
You are watching short clips from a delivery game played on a 10x10 grid. A delivery vehicle
moves one cell per step: up, down, left, or right. Its job is to deliver a package to the
destination.

What is on the map:
- Roads are either white road or brick road. Brick road is rougher and slower to drive on.
- Some cells have a coin (a small bonus for the driver) or a roadblock (a small hindrance).
- Some cells have sheep. Hitting a sheep is very bad, and the game ends.
- Reaching the destination completes the delivery, and the game ends.
- The vehicle cannot drive into a house or off the edge of the grid; trying to leaves it where
  it is."""

RULES = {"rewards": TASK_RULES, "neutral": NEUTRAL_RULES}

MAP_LEGEND = ("Legend: . white road | b brick road | c coin on white road | C coin on brick road | "
              "r roadblock on white road | R roadblock on brick road | S sheep | G destination | H house\n"
              "Coordinates are (row, column); row 0 is the top row, column 0 the left column.")


def render_map(env):
    """ASCII map of the whole board with row/column indices."""
    b = np.array(env.board, dtype=int)
    lines = ["     " + " ".join(str(c) for c in range(b.shape[1]))]
    for r in range(b.shape[0]):
        lines.append(f"  {r}  " + " ".join(CELL_INFO[v][0] for v in b[r]))
    return "\n".join(lines)


def reconstruct_path(env, feats_row):
    """Find action sequences consistent with one segment's stored data.

    feats_row: length-10 array = 6 discounted reward features + [sx, sy, ex, ey].
    Returns (canonical_actions, n_distinct_routes). canonical_actions is a list
    of 3 (dx, dy) tuples.
    """
    start = [int(feats_row[6]), int(feats_row[7])]
    end = [int(feats_row[8]), int(feats_row[9])]
    matches = {}  # distinct route (sequence of cells) -> list of action sequences
    for acts in itertools.product(GridWorldEnv.actions, repeat=3):
        seg = [start] + [list(a) for a in acts]
        phi_dis, _ = find_reward_features(seg, env, use_extended_SF=False, gamma=GAMMA)
        sa_list, end_state = get_sa_list(seg, env)
        if list(end_state) == end and np.allclose(phi_dis, feats_row[:6], atol=1e-5):
            route = tuple(tuple(s[:2]) for s in sa_list) + (tuple(end_state),)
            matches.setdefault(route, []).append(acts)
    if not matches:
        raise ValueError(f"no action sequence reproduces segment {feats_row}")

    def n_bumps(route):  # steps where the vehicle did not move
        return sum(route[k] == route[k + 1] for k in range(len(route) - 1))

    # Canonical choice: fewest bumps, then the order itertools produced (fixed).
    best_route = min(matches, key=n_bumps)
    return list(matches[best_route][0]), len(matches)


def describe_segment(env, start, actions):
    """Step-by-step text for one segment, following the env's transition rules."""
    lines = [f"Starts at {tuple(start)}, on {CELL_INFO[int(env.board[start[0]][start[1]])][1]}."]
    x, y = start
    for k, (dx, dy) in enumerate(actions, start=1):
        if env.is_terminal(x, y):
            lines.append(f"Step {k}: (the game has already ended)")
            continue
        nx, ny = x + dx, y + dy
        direction = DIRECTION_NAMES[(dx, dy)]
        if not (0 <= nx < env.height and 0 <= ny < env.width):
            lines.append(f"Step {k}: tries to move {direction} off the edge of the grid; "
                         f"stays at {(x, y)}.")
        elif env.is_blocked(nx, ny):
            lines.append(f"Step {k}: tries to move {direction} into a house at {(nx, ny)}; "
                         f"stays at {(x, y)}.")
        else:
            x, y = nx, ny
            text = f"Step {k}: moves {direction} to {(x, y)}: {CELL_INFO[int(env.board[x][y])][1]}."
            if env.is_terminal(x, y):
                text += " The game ends."
            lines.append(text)
    return "\n".join(lines)


def serialize_pair(env, pair_feats, names=("Segment 1", "Segment 2"), variant="rewards"):
    """Full text for one pair: rules, map, and both segments (in the given order).

    pair_feats: (2, 10) array for the two segments, already in display order.
    variant: "rewards" (rules with reward numbers, as taught to humans) or
             "neutral" (qualitative description only; see NEUTRAL_RULES).
    Returns (text, n_routes) where n_routes gives the candidate count per segment.
    """
    parts = [RULES[variant], "", "Map:", render_map(env), MAP_LEGEND, ""]
    n_routes = []
    for name, f in zip(names, pair_feats):
        actions, n = reconstruct_path(env, f)
        n_routes.append(n)
        start = [int(f[6]), int(f[7])]
        parts += [f"{name}:", describe_segment(env, start, actions), ""]
    return "\n".join(parts).rstrip(), n_routes


def main():
    env = load_delivery_env()
    feats = np.load(HUMAN_DIR + "DELIVERY_MDP_segment_pair_features_regret_form.npy",
                    allow_pickle=False)
    prefs = np.load(HUMAN_DIR + "DELIVERY_MDP_human_prefs.npy", allow_pickle=False)
    w = np.load(HUMAN_DIR + "DELIVERY_MDP_gt_rew_vec.npy", allow_pickle=False)

    idx = [int(a) for a in sys.argv[1:]] or [0, 3, 12]
    for i in idx:
        text, n_routes = serialize_pair(env, feats[i])
        G = feats[i, :, :6] @ w
        human = {0.0: "Segment 1", 1.0: "Segment 2", 0.5: "tie"}[prefs[i]]
        print("=" * 100)
        print(f"PAIR {i}   [not shown to the LLM: G = {G.round(2).tolist()}, "
              f"human preferred: {human}, candidate routes per segment: {n_routes}]")
        print("=" * 100)
        print(text)
        print()


if __name__ == "__main__":
    main()
