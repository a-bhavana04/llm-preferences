# LLM Preferences: Which Preference Model Do LLM Judges Follow?

CS 690S (AI Alignment, UMass) final project.
Anjali Kanvinde, Bhavana Anand, Sharvi Endait.

## What this is

We test whether LLM judges compare short trajectory clips by **partial return**
(total reward collected) or **regret** (how good each decision was relative to the
best option), and whether they match humans.

**Main finding (H1):** in the Knox et al. delivery gridworld, LLM judges
(Llama-3.1-8B, Qwen-2.5-7B) follow partial return, while humans follow regret,
on identical clips.

Judge model:

    P(A > B) = sigmoid( tau * [ a*(G_A - G_B) + b*(dV_A - dV_B) ] )

- `(a,b) = (1,0)` partial return, `(1,1)` regret, `(0,1)` outcome-only
- `G` = partial return of a clip, `dV = V*(end) - V*(start)`
- Under deterministic dynamics, regret = `G + dV`.

## Key result

651 discriminating pairs (partial return and regret predict opposite winners),
order-consistent answers:

| Judge            | Partial return | Regret |
|------------------|----------------|--------|
| Llama-3.1-8B     | 0.98           | 0.02   |
| Qwen-2.5-7B      | 0.81           | 0.19   |
| Humans           | 0.40           | 0.60   |

Full-set fits agree: partial return wins for both LLMs (dAIC 176 Llama, 111 Qwen
over regret); regret wins for humans (dAIC 166 over partial return).
Full tables, Wilson intervals and caveats are in `RESULTS_NOTES.md`.

## This repo contains only our work

The environment, gridworld code and human labels are from the Knox et al. repo
and are **not** redistributed here:
https://github.com/Stephanehk/Learning-OA-From-Prefs

Our files:

- `serialize_segments.py` - turn a clip pair into a text prompt
- `query_llm.py` - query an LLM judge (Ollama / OpenAI-compatible), both orderings, with re-ask + resume
- `fit_human.py` - reproduce the human baseline fit
- `fit_llm.py` - fit the three models to LLM labels, plus the game-ending-segment split
- `fit_models.py` - simulated-judge recovery check (validates the fitter)
- `load_human_prefs.py` - load the Knox human labels
- `verify_vstar_bug.py` - documents a V* bug in the Knox repo (see note below)
- `RESULTS_NOTES.md` - full results, tables, data caveats
- `data/llm_labels/*.jsonl` - the LLM judgments we collected (ours to share)

## Setup (teammates start here)

The scripts import `learn_advantage` from the Knox repo, so run them **inside a
Knox clone**:

    git clone https://github.com/Stephanehk/Learning-OA-From-Prefs
    cd Learning-OA-From-Prefs
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    pip install torch

    # add our work into the Knox repo root
    git clone <this-repo-url> llm-preferences
    cp llm-preferences/*.py llm-preferences/RESULTS_NOTES.md .
    mkdir -p data/llm_labels && cp llm-preferences/data/llm_labels/*.jsonl data/llm_labels/

Human preference data (`data/human_prefs/`) is **not** in git. Get it from the
Knox release and place it there before running `fit_human.py`.

LLM judges run locally via Ollama (https://ollama.com):

    ollama pull llama3.1:8b
    ollama pull qwen2.5:7b

## Reproduce

    venv/bin/python fit_human.py                                    # human baseline (regret CV 0.574)
    venv/bin/python query_llm.py --model llama3.1:8b --pairs all    # collect labels (long: ~14h local)
    venv/bin/python fit_llm.py data/llm_labels/llama3.1-8b_full.jsonl

## Status

- **H1** (LLM vs. human preference model): done, supported.
- **H2** (steering via chain-of-thought / rubric): neutral-prompt check done; full runs pending.
- **H3** (policy degradation from mismatch): not started.
- **H4** (transfer to discretized MountainCar): not started.

Biggest open item: the **capability-vs-preference confound**. The 8B models may
follow partial return because they cannot compute regret's lookahead. The clean
test is one larger judge (e.g. Qwen2.5-72B) on the 60 discriminating pairs.

## Note on the V* bug

The Knox repo's value iteration zeroes successor features on the terminal
transition; the published human result reproduces only with the corrected V*.
We use the corrected version. `verify_vstar_bug.py` documents it. Not yet
reported upstream.
