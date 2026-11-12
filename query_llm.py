"""
query_llm.py -- ask an LLM judge which segment of each human-study pair is better.

For every pair (the SAME 1,793 pairs as the human baseline), the pair is shown
in BOTH orderings:
    "AB": dataset segment A shown as "Segment 1", B as "Segment 2"
    "BA": swapped
The judge may answer 1, 2, or equal. Each answer is converted back to the
dataset's A/B frame and saved as
    label  = [1, 0] (A preferred), [0, 1] (B preferred), [0.5, 0.5] (equal)
    y_A    = 1, 0, or 0.5   (= P(A preferred), the format fit_human.py uses)
together with the raw response, one JSON object per line in --out.
Already-saved (pair, order) entries are skipped, so a run can be resumed.

Parsing: we look for a final "Answer: 1/2/equal" line (ignoring any
<think>...</think> block from reasoning models). If there is none, we re-ask
ONCE in the same conversation: "Reply with only: 1, 2, or equal". Both replies
are saved; `parsed_from` says which one the label came from.

Talks to any OpenAI-compatible chat endpoint (e.g. Ollama) using only the
Python standard library (urllib), so nothing extra needs to be installed.

Pilot (default: 20 random pairs, fixed seed):
    venv/bin/python query_llm.py
Full run (all 1,793 pairs; only after the pilot is reviewed):
    venv/bin/python query_llm.py --model mistral --n_pairs 0
Other model (writes data/llm_labels/llama3.1-8b_pilot.jsonl):
    venv/bin/python query_llm.py --model llama3.1:8b
Neutral prompt variant (writes data/llm_labels/llama3.1-8b_neutral_pilot.jsonl):
    venv/bin/python query_llm.py --model llama3.1:8b --prompt_variant neutral
"""

import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

from load_human_prefs import load_delivery_env, load_human_prefs, HUMAN_DIR
from serialize_segments import serialize_pair

SYSTEM_PROMPT = "You are a careful judge of short gameplay clips."

QUESTION = """\
Question: Which of the two segments is better?
Explain your reasoning briefly, then finish with exactly one final line of the form
"Answer: 1", "Answer: 2", or "Answer: equal"."""

ANSWER_RE = re.compile(r"answer\s*[:=]?\s*\**\s*(1|2|equal)\b", re.IGNORECASE)


def chat(base_url, api_key, model, messages, temperature, seed, max_tokens=2000, timeout=1800):
    """One chat-completion call to an OpenAI-compatible endpoint.
    messages: list of {"role", "content"} dicts (system message is added here).
    Returns (text, finish_reason); finish_reason "length" means the reply was cut off."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": temperature,
        "seed": seed,
        # Hard cap on reply length. Ollama's default context is 4096 tokens and our
        # prompt is ~1000, so an uncapped reply can overflow it and ramble (this
        # happened with deepseek-r1). A cut-off reply has finish_reason "length".
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        choice = json.load(resp)["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason")


def strip_think(text):
    """Remove a reasoning model's <think>...</think> block (e.g. deepseek-r1), so we
    only parse the final answer. An unclosed <think> (cut-off reply) leaves nothing."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def parse_choice(text):
    """Last 'Answer: 1/2/equal' in the final answer -> '1', '2', 'equal', or None."""
    found = ANSWER_RE.findall(strip_think(text))
    return found[-1].lower() if found else None


REASK = "Reply with only: 1, 2, or equal"
BARE_RE = re.compile(r"^\W*(1|2|equal)\W*$", re.IGNORECASE)


def parse_reask(text):
    """Parse the short reply to the re-ask: the whole reply must be 1, 2 or equal
    (optionally with punctuation), or an 'Answer: ...' line."""
    final = strip_think(text)
    m = BARE_RE.match(final)
    return m.group(1).lower() if m else parse_choice(final)


def ask_judge(args, prompt):
    """Ask once; if no answer line is found, re-ask ONCE in the same conversation.
    Returns a dict of fields to store (raw response, re-ask reply, parsed choice...)."""
    msgs = [{"role": "user", "content": prompt}]
    raw, finish = chat(args.base_url, args.api_key, args.model, msgs, args.temperature,
                       args.seed, args.max_tokens)
    out = dict(raw_response=raw, finish_reason=finish, reask_response=None,
               choice=parse_choice(raw), parsed_from="answer_line")
    if out["choice"] is None:
        # Send back only the final answer text (no <think> block), so the re-ask
        # conversation stays well inside the context window.
        msgs += [{"role": "assistant", "content": strip_think(raw)},
                 {"role": "user", "content": REASK}]
        reply, _ = chat(args.base_url, args.api_key, args.model, msgs, args.temperature,
                        args.seed, args.max_tokens)
        out.update(reask_response=reply, choice=parse_reask(reply),
                   parsed_from="reask" if parse_reask(reply) else "unparsed")
    return out


def to_A_frame(choice, order):
    """Convert the judge's choice (about displayed positions) to y_A = P(A preferred)."""
    if choice is None:
        return None
    if choice == "equal":
        return 0.5
    shown_first_is_A = (order == "AB")
    picked_first = (choice == "1")
    return 1.0 if picked_first == shown_first_is_A else 0.0


def load_done(path):
    """Already-saved records, keyed by (pair_idx, order)."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[(r["pair_idx"], r["order"])] = r
    return done


def summarize(records, human_y, model_preds, n_show=3):
    """Agreement with humans and with each model's noiseless prediction, order-flip
    rate, position bias; print raw responses.
    model_preds: {name: array of y_A predicted by that preference model (no noise)}."""
    by_pair = {}
    for r in records:
        by_pair.setdefault(r["pair_idx"], {})[r["order"]] = r
    pairs = sorted(p for p, o in by_pair.items() if {"AB", "BA"} <= set(o))
    calls = [by_pair[p][o] for p in pairs for o in ("AB", "BA")]

    unparsed = sum(r["y_A"] is None for r in calls)
    valid = [r for r in calls if r["y_A"] is not None]
    reasked = sum(r.get("reask_response") is not None for r in calls)
    rescued = sum(r.get("parsed_from") == "reask" for r in calls)
    cut_off = sum(r.get("finish_reason") == "length" for r in calls)
    secs = [r.get("seconds", 0) for r in calls]
    print(f"\nPairs with both orderings: {len(pairs)}  calls: {len(calls)}  "
          f"mean {np.mean(secs):.1f} s/call")
    print(f"Unparseable after re-ask: {unparsed}/{len(calls)} ({unparsed / len(calls):.1%})   "
          f"re-asked: {reasked}, rescued by re-ask: {rescued}   cut off (finish_reason=length): {cut_off}")

    # Answer distribution by displayed position (position bias check)
    pos = {c: sum(r["choice"] == c for r in valid) for c in ("1", "2", "equal")}
    print(f"Answers by displayed position: Segment 1 = {pos['1']}, Segment 2 = {pos['2']}, "
          f"equal = {pos['equal']}")

    # Agreement with human labels, per call (same definition as fit_human.py:
    # only non-tied human labels; an LLM 'equal' counts as 0.5)
    strict = [r for r in valid if human_y[r["pair_idx"]] != 0.5]
    agree = [0.5 if r["y_A"] == 0.5 else float(r["y_A"] == human_y[r["pair_idx"]]) for r in strict]
    exact = [r["y_A"] == human_y[r["pair_idx"]] for r in valid]
    print(f"Agreement with human (non-tied human labels, LLM equal = 0.5): "
          f"{np.mean(agree):.3f}  over {len(agree)} calls")
    print(f"Exact 3-way match with human (A / B / equal), all labels: "
          f"{np.mean(exact):.3f}  over {len(exact)} calls")

    # Which preference model do the labels track? Agreement with each model's
    # noiseless prediction (pairs the model calls a tie are skipped; an
    # 'equal' label counts 0.5). Humans on the same pairs are shown for reference.
    print("Agreement with noiseless model predictions (same pairs):")
    for m, pred in model_preds.items():
        rel = [r for r in valid if pred[r["pair_idx"]] != 0.5]
        a = [0.5 if r["y_A"] == 0.5 else float(r["y_A"] == pred[r["pair_idx"]]) for r in rel]
        hp = [p for p in pairs if pred[p] != 0.5]
        h = [0.5 if human_y[p] == 0.5 else float(human_y[p] == pred[p]) for p in hp]
        print(f"  {m:<15} LLM {np.mean(a):.3f} (n={len(a)})   human {np.mean(h):.3f} (n={len(h)})")

    # Order flips: in the A/B frame, the two orderings should give the same label.
    both = [(by_pair[p]["AB"]["y_A"], by_pair[p]["BA"]["y_A"]) for p in pairs
            if by_pair[p]["AB"]["y_A"] is not None and by_pair[p]["BA"]["y_A"] is not None]
    flips = [a != b for a, b in both]
    hard_flips = [{a, b} == {0.0, 1.0} for a, b in both]
    print(f"Order-flip rate (label changes when order is swapped): {np.mean(flips):.3f}  "
          f"({sum(flips)}/{len(both)} pairs; full A<->B reversals: {sum(hard_flips)})")

    # Order-consistent pairs: same non-equal answer in both orderings, so the choice
    # can't be explained by position bias. Agreement of those answers with each
    # model's prediction and with the human label.
    cons = {p: by_pair[p]["AB"]["y_A"] for p in pairs
            if by_pair[p]["AB"]["y_A"] in (0.0, 1.0) and by_pair[p]["AB"]["y_A"] == by_pair[p]["BA"]["y_A"]}
    print(f"Order-consistent pairs: {len(cons)}/{len(pairs)}")
    refs = dict(model_preds, human=human_y)
    for m, ref in refs.items():
        ps = [p for p in cons if ref[p] != 0.5]
        if ps:
            k = sum(cons[p] == ref[p] for p in ps)
            print(f"  consistent answers agreeing with {m:<15} {k}/{len(ps)} = {k / len(ps):.3f}")

    print("\nPer pair (y_A: 1 = A preferred, 0 = B, 0.5 = equal):")
    print(f"  {'pair':>5} {'human':>6} {'LLM AB':>7} {'LLM BA':>7}")
    for p in pairs:
        print(f"  {p:5d} {human_y[p]:6.1f} {str(by_pair[p]['AB']['y_A']):>7} "
              f"{str(by_pair[p]['BA']['y_A']):>7}")

    for r in calls[:n_show]:
        print("\n" + "=" * 90)
        print(f"RAW RESPONSE  pair {r['pair_idx']}  order {r['order']}  parsed choice: {r['choice']}  "
              f"-> y_A = {r['y_A']}  (human y_A = {human_y[r['pair_idx']]})")
        print("=" * 90)
        print(r["raw_response"])
        if r.get("reask_response") is not None:
            print(f"--- re-ask: {REASK!r} -> reply: {r['reask_response']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:11434/v1")
    # Key comes from the HF_TOKEN env var if set (hosted runs), so it never appears on the
    # command line or in the saved records; "ollama" is a placeholder Ollama ignores.
    ap.add_argument("--api_key", default=os.environ.get("HF_TOKEN", "ollama"))
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--n_pairs", type=int, default=20, help="random pilot size (0 = all pairs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2000, help="cap on each reply's length")
    ap.add_argument("--pairs", choices=["all", "discriminating"], default="all",
                    help="pool to sample from: all pairs, or only pairs where partial return "
                         "and regret predict opposite winners")
    ap.add_argument("--prompt_variant", choices=["rewards", "neutral"], default="rewards",
                    help="rewards: rules with reward numbers (as taught to humans); "
                         "neutral: qualitative rules, no numbers, no 'maximize reward'")
    ap.add_argument("--out", default=None,
                    help="default: data/llm_labels/<model>_pilot.jsonl (or _full when --n_pairs 0)")
    args = ap.parse_args()
    if args.out is None:  # one file per model, so earlier pilots are never overwritten
        tag = "full" if args.n_pairs == 0 else "pilot"
        if args.pairs == "discriminating":
            tag = f"disc{args.n_pairs or 'all'}"
        variant = "" if args.prompt_variant == "rewards" else f"_{args.prompt_variant}"
        args.out = f"data/llm_labels/{args.model.replace(':', '-').replace('/', '_')}{variant}_{tag}.jsonl"

    env = load_delivery_env()
    feats = np.load(HUMAN_DIR + "DELIVERY_MDP_segment_pair_features_regret_form.npy", allow_pickle=False)
    human_y = 1.0 - np.load(HUMAN_DIR + "DELIVERY_MDP_human_prefs.npy", allow_pickle=False)

    # Noiseless predictions of partial return and regret, in y_A form (0.5 = tie)
    d = load_human_prefs()
    def pred(stat):
        s = np.round(stat, 6)
        return np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))
    model_preds = {"partial_return": pred(d["G"][:, 0] - d["G"][:, 1]),
                   "regret": pred((d["G"] + d["dV"])[:, 0] - (d["G"] + d["dV"])[:, 1])}

    # Which pairs to query. "discriminating" = pairs where partial return and regret
    # predict OPPOSITE winners (neither predicts a tie): the only pairs whose
    # labels can tell the two models apart (651 of the 1793).
    if args.pairs == "discriminating":
        pool = np.where((model_preds["partial_return"] != 0.5) & (model_preds["regret"] != 0.5)
                        & (model_preds["partial_return"] != model_preds["regret"]))[0]
    else:
        pool = np.arange(len(feats))
    if args.n_pairs and args.n_pairs < len(pool):
        idx = np.sort(np.random.default_rng(args.seed).choice(pool, args.n_pairs, replace=False))
    else:
        idx = pool

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out)
    print(f"{len(idx)} pairs x 2 orderings with {args.model} at {args.base_url}; "
          f"{len(done)} calls already saved in {args.out}")

    with open(args.out, "a") as f:
        for n, i in enumerate(idx, 1):
            for order in ("AB", "BA"):
                if (int(i), order) in done:
                    continue
                shown = feats[i] if order == "AB" else feats[i][::-1]
                text, n_routes = serialize_pair(env, shown, variant=args.prompt_variant)
                prompt = text + "\n\n" + QUESTION
                t0 = time.time()
                ans = ask_judge(args, prompt)
                y_A = to_A_frame(ans["choice"], order)
                label = None if y_A is None else [y_A, 1.0 - y_A]
                rec = dict(pair_idx=int(i), order=order, y_A=y_A, label=label, **ans,
                           model=args.model, prompt_variant=args.prompt_variant,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           seed=args.seed, n_routes=n_routes, seconds=round(time.time() - t0, 1))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(int(i), order)] = rec
            print(f"  [{n}/{len(idx)}] pair {i}: AB -> {done[(int(i), 'AB')]['choice']}, "
                  f"BA -> {done[(int(i), 'BA')]['choice']}", flush=True)

    summarize([done[(int(i), o)] for i in idx for o in ("AB", "BA")], human_y, model_preds)


if __name__ == "__main__":
    main()
