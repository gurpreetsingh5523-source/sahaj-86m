"""SEHAJ v7 evaluation harness.

Usage:
  python3 evaluate.py --checkpoint checkpoint_best.pt
  python3 evaluate.py --checkpoint random          # random-init smoke test

Battery:
  1. Val loss + bits-per-byte on val.bin (uint16 tokens). BPB uses the UTF-8
     byte length of the decoded val text. CAVEAT: the old SEHAJ 30M base
     scored 1.22 BPB on ITS OWN held-out set; ours is measured on OUR
     held-out (val.bin), so the comparison is indicative, not exact.
  2. Generation battery: 10 fixed prompts (5 Gurmukhi, 3 instruction-style,
     2 English), each with greedy AND sampled (t=0.7, top-p 0.9) decoding,
     max 80 new tokens, verbatim outputs recorded.
  3. EOS stop rate: fraction of generations terminating at EOS (id 3) before
     the token cap — the metric SEVAK failed at 0% (EOS was never in its
     training targets; see encode_sft_example in build_sft.py).
  4. Ground-truth QA on sevak_factual_ground_truth.jsonl, decontaminated
     against every prompt in the SFT data and local SFT source files (a
     previous eval was inflated by 14/32 leaked items).

Writes eval_results_<timestamp>.json and eval_report.md.
"""
import argparse
import datetime
import json
import math
import os
import re
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

from SEHAJ_v7_FINAL import Config, SEHAJ
from build_sft import PROMPT_TEMPLATE  # single source of truth for the chat format

WORK = os.path.dirname(os.path.abspath(__file__))
KIT = "/Users/gurpreetdhillon/MacBook_Testing_Kit"

MODEL_CFG = Config(vocab_size=6000, d_model=512, n_layers=6, n_heads=8,
                   d_state=64, d_ff=2048, max_seq_len=512, dropout=0.1,
                   tie_embeddings=True)
BOS_ID, EOS_ID = 2, 3

GEN_PROMPTS = [
    # 5 Gurmukhi (greetings / simple questions)
    "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
    "ਸਵਾਲ: ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?\nਜਵਾਬ:",
    "ਸਵਾਲ: ਸ੍ਰੀ ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ ਕੌਣ ਸਨ?\nਜਵਾਬ:",
    "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
    "ਸਵਾਲ: ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਕਿਹੜੀ ਲਿਪੀ ਵਿੱਚ ਲਿਖੀ ਜਾਂਦੀ ਹੈ?\nਜਵਾਬ:",
    # 3 instruction-style
    "ਨਿਰਦੇਸ਼: ਸਿਹਤਮੰਦ ਰਹਿਣ ਲਈ ਤਿੰਨ ਸੁਝਾਅ ਦਿਓ।\nਜਵਾਬ:",
    "ਨਿਰਦੇਸ਼: ਪੰਜਾਬ ਦੀਆਂ ਤਿੰਨ ਮੁੱਖ ਫਸਲਾਂ ਦੇ ਨਾਮ ਦੱਸੋ।\nਜਵਾਬ:",
    "ਨਿਰਦੇਸ਼: 'ਸੱਚ' ਸ਼ਬਦ ਦੀ ਵਰਤੋਂ ਕਰਦਿਆਂ ਇੱਕ ਵਾਕ ਲਿਖੋ।\nਜਵਾਬ:",
    # 2 English
    "Question: What is the capital of Punjab?\nAnswer:",
    "Hello, how are you?",
]

OLD_SEHAJ_BPB = 1.22  # old SEHAJ 30M base, on its OWN held-out set (caveat)


# ----------------------------------------------------------------------------- model
def load_model(checkpoint, device):
    if checkpoint == "random":
        torch.manual_seed(0)
        model = SEHAJ(MODEL_CFG)
        meta = {"checkpoint": "random-init (seed 0)", "step": 0}
    else:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = Config(**ck["config"]) if isinstance(ck, dict) and "config" in ck else MODEL_CFG
        model = SEHAJ(cfg)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        meta = {"checkpoint": os.path.abspath(checkpoint),
                "step": ck.get("step"), "tokens_seen": ck.get("tokens_seen"),
                "best_val": ck.get("best_val")}
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    meta["params"] = n_params
    print(f"model: {n_params:,} params on {device} ({meta['checkpoint']})")
    return model, meta


@torch.no_grad()
def generate(model, prompt_ids, max_new=80, temperature=0.0, top_p=1.0):
    """Returns (generated_ids, stopped_at_eos). temperature==0 -> greedy."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    logits, states, caches = model(ids)
    out, stopped = [], False
    for _ in range(max_new):
        z = logits[:, -1].float()
        if temperature <= 0:
            nxt = int(z.argmax(-1))
        else:
            z = z / max(temperature, 1e-5)
            probs = F.softmax(z, -1)
            sp_, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp_, -1)
            keep = cum - sp_ < top_p           # top-p nucleus (always keeps top-1)
            sp_[~keep] = 0.0
            sp_ /= sp_.sum()
            nxt = int(si.gather(-1, torch.multinomial(sp_, 1)))
        if nxt == EOS_ID:
            stopped = True
            break
        out.append(nxt)
        logits, states, caches = model(torch.tensor([[nxt]], dtype=torch.long,
                                                    device=ids.device), states, caches)
    return out, stopped


# ----------------------------------------------------------------------------- 1. val loss + BPB
@torch.no_grad()
def eval_val(model, sp, val_path, device, seq, max_windows, batch=8):
    ids = np.memmap(val_path, dtype=np.uint16, mode="r").astype(np.int64)
    n_win_total = (len(ids) - 1) // seq
    if max_windows and max_windows < n_win_total:
        win_starts = np.linspace(0, (n_win_total - 1) * seq, max_windows).astype(np.int64)
    else:
        win_starts = np.arange(0, n_win_total * seq, seq, dtype=np.int64)
    losses = []
    for i in range(0, len(win_starts), batch):
        chunk = win_starts[i:i + batch]
        x = torch.from_numpy(np.stack([ids[s:s + seq] for s in chunk])).to(device)
        y = torch.from_numpy(np.stack([ids[s + 1:s + seq + 1] for s in chunk])).to(device)
        logits, _, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      y.reshape(-1)).item())
    mean_loss = sum(losses) / len(losses)
    text = sp.decode(ids.tolist())
    n_bytes = len(text.encode("utf-8"))
    bpb = mean_loss * len(ids) / math.log(2) / n_bytes
    return {"val_loss": mean_loss, "windows_scored": len(win_starts),
            "val_tokens": int(len(ids)), "val_bytes_utf8": n_bytes, "bpb": bpb,
            "old_sehaj_30m_bpb_own_heldout": OLD_SEHAJ_BPB,
            "beats_old_sehaj": bool(bpb < OLD_SEHAJ_BPB),
            "caveat": ("BPB comparison is indicative only: old SEHAJ 1.22 BPB was "
                       "measured on its own held-out set, ours on val.bin.")}


# ----------------------------------------------------------------------------- 4. QA decontamination
def _norm_key(s):
    """NOTE (2026-09-18): `\\w` in Python's `re` does NOT include Unicode
    combining marks (category Mn/Mc) — Gurmukhi matras (ਿ ੀ ੁ ੂ ੇ ੈ ੋ ੌ ਾ ਂ
    ਼ etc.) are exactly that category, so a plain `[^\\w\\s]` strip shreds every
    multi-matra Gurmukhi word into bare-consonant fragments (e.g. "ਪਹਿਲਾ" ->
    "ਪਹ ਲ"), which then fail the len>=3 content-word filter below. Keep any
    character whose Unicode category is Letter/Mark/Number instead."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def load_sft_prompts():
    prompts = set()
    for path in [os.path.join(WORK, "sft_data.jsonl"), os.path.join(WORK, "sft_val.jsonl"),
                 os.path.join(KIT, "sevak_sovereign_bilingual_sft_master.jsonl"),
                 os.path.join(KIT, "sevak_bilingual_pa_en_master_sft.jsonl"),
                 os.path.join(KIT, "sevak_master_sovereign_knowledge_full.jsonl"),
                 os.path.join(KIT, "sevak_massive_sovereign_knowledge_5k.jsonl")]:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = d.get("prompt") or d.get("instruction") or ""
                if p:
                    prompts.add(_norm_key(p))
    return prompts


def answer_correct(expected, output):
    """Substring/keyword match: exact normalized substring, or >=60% of the
    expected answer's content words (Gurmukhi/Latin words, len>=3) present."""
    e, o = _norm_key(expected), _norm_key(output)
    if e and e in o:
        return True
    words = [w for w in e.split() if len(w) >= 3]
    if not words:
        return False
    hits = sum(1 for w in words if w in o)
    return hits / len(words) >= 0.6


@torch.no_grad()
def eval_ground_truth(model, sp, device, max_new=80):
    path = os.path.join(KIT, "sevak_factual_ground_truth.jsonl")
    items = [json.loads(l) for l in open(path) if l.strip()]
    sft_prompts = load_sft_prompts()
    results, leaked = [], 0
    for it in items:
        q, expected = it["instruction"], it["response"]
        k = _norm_key(q)
        contaminated = any(k and (k in p or p in k) for p in sft_prompts)
        if contaminated:
            leaked += 1
            continue
        prompt_ids = [BOS_ID] + list(sp.encode(PROMPT_TEMPLATE.format(prompt=q)))
        out_ids, stopped = generate(model, prompt_ids, max_new=max_new, temperature=0.0)
        output = sp.decode(out_ids)
        results.append({"question": q, "expected": expected, "output": output,
                        "stopped_at_eos": stopped,
                        "correct": answer_correct(expected, output)})
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0.0
    return {"items_total": len(items), "excluded_contaminated": leaked,
            "items_scored": n, "accuracy": acc, "per_item": results}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                        help="path to checkpoint .pt, or 'random' for a random-init model")
    ap.add_argument("--val-bin", default=os.path.join(WORK, "val.bin"))
    ap.add_argument("--tokenizer", default=os.path.join(WORK, "sevak_bpe_6k.model"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--val-windows", type=int, default=512,
                    help="evenly-spaced val windows to score (0 = all)")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--output-dir", default=WORK)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())

    model, meta = load_model(args.checkpoint, device)

    print("[1/4] val loss + BPB ...")
    val_res = eval_val(model, sp, args.val_bin, device, args.seq_len, args.val_windows)
    print(f"  val_loss={val_res['val_loss']:.4f} bpb={val_res['bpb']:.4f} "
          f"(old SEHAJ 30M: {OLD_SEHAJ_BPB} on its own held-out)")

    print("[2/4] generation battery ...")
    gens = []
    for p in GEN_PROMPTS:
        pids = [BOS_ID] + list(sp.encode(p))
        g_ids, g_stop = generate(model, pids, max_new=args.max_new, temperature=0.0)
        s_ids, s_stop = generate(model, pids, max_new=args.max_new,
                                 temperature=0.7, top_p=0.9)
        gens.append({"prompt": p,
                     "greedy": sp.decode(g_ids), "greedy_stopped_at_eos": g_stop,
                     "sampled_t0.7_p0.9": sp.decode(s_ids), "sampled_stopped_at_eos": s_stop})

    eos_rate = sum(g["greedy_stopped_at_eos"] + g["sampled_stopped_at_eos"]
                   for g in gens) / (2 * len(gens))
    print(f"[3/4] EOS stop rate: {eos_rate:.2%} (SEVAK failed at 0%)")

    print("[4/4] ground-truth QA (decontaminated) ...")
    qa = eval_ground_truth(model, sp, device, max_new=args.max_new)
    print(f"  {qa['items_scored']} scored ({qa['excluded_contaminated']} excluded as "
          f"contaminated), accuracy={qa['accuracy']:.2%}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"timestamp": ts, "model": meta, "device": device,
               "val": val_res, "eos_stop_rate": eos_rate,
               "generations": gens, "ground_truth_qa": qa}
    json_path = os.path.join(args.output_dir, f"eval_results_{ts}.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    md_path = os.path.join(args.output_dir, "eval_report.md")
    with open(md_path, "w") as fh:
        fh.write(f"# SEHAJ v7 eval report — {ts}\n\n")
        fh.write(f"model: `{meta['checkpoint']}` (step={meta.get('step')}, "
                 f"params={meta['params']:,}, device={device})\n\n")
        fh.write(f"## 1. Val loss + BPB\n\n- val_loss: **{val_res['val_loss']:.4f}** "
                 f"({val_res['windows_scored']} windows x {args.seq_len} tokens)\n"
                 f"- BPB: **{val_res['bpb']:.4f}** ({val_res['val_tokens']:,} tokens, "
                 f"{val_res['val_bytes_utf8']:,} UTF-8 bytes)\n"
                 f"- old SEHAJ 30M base: {OLD_SEHAJ_BPB} BPB on its own held-out set "
                 f"→ beats it: **{val_res['beats_old_sehaj']}**\n"
                 f"- caveat: {val_res['caveat']}\n\n")
        fh.write(f"## 2/3. Generation battery + EOS stop rate\n\n"
                 f"EOS stop rate: **{eos_rate:.2%}** over {2 * len(gens)} generations "
                 f"(SEVAK failed at 0%).\n\n")
        for i, g in enumerate(gens, 1):
            fh.write(f"### Prompt {i}\n```\n{g['prompt']}\n```\n"
                     f"- greedy (eos={g['greedy_stopped_at_eos']}):\n```\n{g['greedy']}\n```\n"
                     f"- sampled t=0.7 p=0.9 (eos={g['sampled_stopped_at_eos']}):\n"
                     f"```\n{g['sampled_t0.7_p0.9']}\n```\n\n")
        fh.write(f"## 4. Ground-truth QA\n\n- {qa['items_scored']} items scored, "
                 f"**{qa['excluded_contaminated']} excluded** (question text appears in "
                 f"SFT sources — decontamination)\n- accuracy: **{qa['accuracy']:.2%}**\n\n")
        for r in qa["per_item"]:
            fh.write(f"- [{'OK' if r['correct'] else 'X'}] {r['question']}\n"
                     f"  - expected: {r['expected']}\n  - output: {r['output']}\n")
    print(f"\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
