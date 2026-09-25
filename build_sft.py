"""SEHAJ v7 — SFT dataset builder.

Builds sft_data.jsonl / sft_val.jsonl ({"prompt": ..., "response": ...}) from:

  1. HF cache HydraIndicLM/punjabi_alpaca_52K (parquet). NOTE: the identical
     Sk4467/punjabi_alpaca_52K copy is deliberately NOT loaded (dedup).
  2. HF cache DebasishDhal99/punjabi-instruction-dataset (CSV; drop rows with
     '<' template placeholders, drop empty outputs, dedup on full text).
  3. HF cache CohereForAI/aya_dataset — language == 'Panjabi' rows only.
  4. Local sevak_sovereign_bilingual_sft_master.jsonl and
     sevak_bilingual_pa_en_master_sft.jsonl — keep only rows whose Gurmukhi
     fraction (Gurmukhi letters / (Gurmukhi + Latin letters)) is >= 0.60.
  5. Local sevak_master_sovereign_knowledge_full.jsonl and
     sevak_massive_sovereign_knowledge_5k.jsonl (template QA, good Gurmukhi).
  6. HF cache Nam-toon-studio/Sehaj-Gurmukhi-Frontier-Reasoning (train.jsonl).

Global filters (all sources): NFC normalize; response length in [10, 4000]
chars; drop responses with >30% Latin script unless inherently bilingual
(code/math); global exact dedup on (prompt, response); near-dedup on
normalized prompt only (later duplicates dropped). Shuffle seed 0; 2% held
out as sft_val.jsonl (disjoint).

================================================================================
CRITICAL FORMAT RULE — EOS SUPERVISION (encode_sft_example below)
================================================================================
The previous model (SEVAK) failed catastrophically because EOS was never in
the training targets: it never learned to stop generating. Therefore EVERY
SFT example, when tokenized for training, MUST end with the EOS token (id 3)
and the loss MUST cover that final EOS position. encode_sft_example()
guarantees this: the sequence is always

    [BOS] + tok("ਸਵਾਲ: {prompt}\nਜਵਾਬ: ") + tok(response) + [EOS]

truncated so the last token is ALWAYS EOS, and labels supervise every
response token INCLUDING the final EOS (prompt tokens are masked to -100).
The SFT trainer must reuse this function verbatim.
================================================================================
"""
import glob
import json
import os
import random
import re
import unicodedata

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")
KIT = "/Users/gurpreetdhillon/MacBook_Testing_Kit"

VOCAB_SIZE = 6000
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MAX_SEQ_LEN = 512

PROMPT_TEMPLATE = "ਸਵਾਲ: {prompt}\nਜਵਾਬ: "  # shared with evaluate.py

VAL_FRAC = 0.02
SEED = 0

GURMUKHI_LO, GURMUKHI_HI = 0x0A00, 0x0A7F


# ----------------------------------------------------------------------------- text utils
def norm_text(s):
    return unicodedata.normalize("NFC", str(s)).strip()


def script_fracs(s):
    """Return (gurmukhi_frac, latin_frac) over Gurmukhi+Latin letters."""
    g = l = 0
    for ch in s:
        o = ord(ch)
        if GURMUKHI_LO <= o <= GURMUKHI_HI:
            g += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            l += 1
    tot = g + l
    return (g / tot if tot else 0.0, l / tot if tot else 0.0)


CODE_MARKERS = ("```", "def ", "import ", "print(", "class ", "return ",
                "#include", "http://", "https://", "SELECT ", "function")
_MATH_CHARS = set("0123456789=+-*/%^(){}[]<>|&;")


def inherently_bilingual(resp):
    """Code/math responses legitimately contain lots of Latin."""
    if any(m in resp for m in CODE_MARKERS):
        return True
    if resp and sum(c in _MATH_CHARS for c in resp) / len(resp) >= 0.20:
        return True
    return False


def global_filter(prompt, response):
    """Returns drop-reason string, or None to keep."""
    if not prompt:
        return "empty_prompt"
    n = len(response)
    if n < 10:
        return "response_too_short"
    if n > 4000:
        return "response_too_long"
    _, latin = script_fracs(response)
    if latin > 0.30 and not inherently_bilingual(response):
        return "latin_heavy_response"
    return None


def prompt_key(p):
    p = unicodedata.normalize("NFC", p).lower()
    p = re.sub(r"[^\w\s]", " ", p, flags=re.UNICODE)
    return re.sub(r"\s+", " ", p).strip()


# ----------------------------------------------------------------------------- sources
def _snapshot(base, pattern):
    hits = sorted(glob.glob(os.path.join(HF_HUB, base, "snapshots", "*", pattern),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"{base}: no snapshot files match {pattern}")
    return hits


def load_alpaca_52k():
    """HydraIndicLM only — the identical Sk4467 copy is NOT loaded (dedup)."""
    counts = {"raw": 0}
    rows = []
    for p in _snapshot("datasets--HydraIndicLM--punjabi_alpaca_52K", "data/*.parquet"):
        df = pd.read_parquet(p)
        counts["raw"] += len(df)
        for ins, inp, out in zip(df["instruction"], df["input"], df["output"]):
            ins, inp, out = norm_text(ins), norm_text(inp), norm_text(out)
            prompt = f"{ins}\n\n{inp}" if inp else ins
            rows.append((prompt, out))
    return rows, counts


def load_debasish():
    counts = {"raw": 0, "drop_placeholder": 0, "drop_empty_output": 0,
              "drop_fulltext_dup": 0}
    rows, seen = [], set()
    for p in _snapshot("datasets--DebasishDhal99--punjabi-instruction-dataset",
                       "data/*.csv"):
        df = pd.read_csv(p).fillna("")
        counts["raw"] += len(df)
        for ins, inp, out in zip(df["instruction"], df["input"], df["output"]):
            ins, inp, out = norm_text(ins), norm_text(inp), norm_text(out)
            if "<" in ins or "<" in inp or "<" in out:
                counts["drop_placeholder"] += 1
                continue
            if not out:
                counts["drop_empty_output"] += 1
                continue
            key = f"{ins}{inp}{out}"
            if key in seen:
                counts["drop_fulltext_dup"] += 1
                continue
            seen.add(key)
            prompt = f"{ins}\n\n{inp}" if inp else ins
            rows.append((prompt, out))
    return rows, counts


def load_aya():
    counts = {"raw": 0}
    rows = []
    for p in _snapshot("datasets--CohereForAI--aya_dataset", "data/train-*.parquet"):
        df = pd.read_parquet(p, columns=["inputs", "targets", "language"])
        df = df[df["language"] == "Panjabi"]
        counts["raw"] += len(df)
        for ins, out in zip(df["inputs"], df["targets"]):
            rows.append((norm_text(ins), norm_text(out)))
    return rows, counts


def _load_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_bilingual(fname):
    """Keep only rows whose Gurmukhi fraction of the text is >= 0.60."""
    counts = {"raw": 0, "drop_low_gurmukhi": 0}
    rows = []
    for d in _load_jsonl(os.path.join(KIT, fname)):
        ins, out = norm_text(d.get("instruction", "")), norm_text(d.get("response", ""))
        counts["raw"] += 1
        gurm, _ = script_fracs(ins + " " + out)
        if gurm < 0.60:
            counts["drop_low_gurmukhi"] += 1
            continue
        rows.append((ins, out))
    return rows, counts


def load_knowledge(fname):
    counts = {"raw": 0}
    rows = []
    for d in _load_jsonl(os.path.join(KIT, fname)):
        ins, out = norm_text(d.get("instruction", "")), norm_text(d.get("response", ""))
        counts["raw"] += 1
        rows.append((ins, out))
    return rows, counts


def load_teacher_factory():
    """Local teacher-factory output, tier 1 (agy/frontier) only — see teacher_factory/generate.py.
    Tier 2 (weak local-fallback teacher) is deliberately NOT loaded here; it's kept as a
    separate file for its own ablation rather than silently blended with frontier-teacher data."""
    path = os.path.join(HERE, "teacher_factory", "sehaj_teacher_factory_tier1_agy.jsonl")
    counts = {"raw": 0}
    rows = []
    if not os.path.exists(path):
        return rows, counts
    for d in _load_jsonl(path):
        ins, out = norm_text(d.get("instruction", "")), norm_text(d.get("response", ""))
        counts["raw"] += 1
        rows.append((ins, out))
    return rows, counts


def load_frontier_reasoning():
    counts = {"raw": 0}
    rows = []
    for p in _snapshot("datasets--Nam-toon-studio--Sehaj-Gurmukhi-Frontier-Reasoning",
                       "*.jsonl"):
        for d in _load_jsonl(p):
            ins, inp, out = (norm_text(d.get("instruction", "")),
                             norm_text(d.get("input", "")),
                             norm_text(d.get("output", "")))
            counts["raw"] += 1
            prompt = f"{ins}\n\n{inp}" if inp else ins
            rows.append((prompt, out))
    return rows, counts


# ----------------------------------------------------------------------------- tokenization
def encode_sft_example(prompt, response, sp, max_len=MAX_SEQ_LEN):
    """Tokenize one SFT example for training.

    CRITICAL EOS RULE (see module docstring): the returned sequence ALWAYS
    ends with EOS (id 3) and the labels ALWAYS supervise it. Layout:

        input_ids = [BOS] + prompt_ids + response_ids + [EOS]
        labels    = [-100]*(1+len(prompt_ids)) + response_ids + [EOS]

    If the example exceeds max_len, response tokens are truncated so the
    final token is still EOS. Examples whose prompt alone fills max_len are
    rejected (return None). The SFT trainer must use this function so every
    training target ends with, and supervises, EOS.
    """
    prompt_ids = list(sp.encode(PROMPT_TEMPLATE.format(prompt=prompt)))
    response_ids = list(sp.encode(response))
    prefix_len = 1 + len(prompt_ids)  # BOS + prompt
    if prefix_len + 1 >= max_len:     # no room for even 1 response token + EOS
        return None
    room = max_len - prefix_len - 1   # response tokens that fit, EOS reserved
    response_ids = response_ids[:room]
    input_ids = [BOS_ID] + prompt_ids + response_ids + [EOS_ID]
    labels = [-100] * prefix_len + response_ids + [EOS_ID]
    assert input_ids[-1] == EOS_ID and labels[-1] == EOS_ID
    return {"input_ids": input_ids, "labels": labels}


# ----------------------------------------------------------------------------- main
def main():
    sp = None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=os.path.join(HERE, "sevak_bpe_6k.model"))
        assert sp.vocab_size() == VOCAB_SIZE and sp.eos_id() == EOS_ID
        print(f"tokenizer OK: vocab={sp.vocab_size()} eos={sp.eos_id()}")
    except Exception as e:
        print(f"tokenizer check skipped/failed: {e}")

    sources = [
        ("hydra_alpaca_52k", load_alpaca_52k),
        ("debasish_punjabi_instruct", load_debasish),
        ("aya_panjabi", load_aya),
        ("sevak_sovereign_bilingual", lambda: load_bilingual("sevak_sovereign_bilingual_sft_master.jsonl")),
        ("sevak_bilingual_pa_en", lambda: load_bilingual("sevak_bilingual_pa_en_master_sft.jsonl")),
        ("sovereign_knowledge_full", lambda: load_knowledge("sevak_master_sovereign_knowledge_full.jsonl")),
        ("sovereign_knowledge_5k", lambda: load_knowledge("sevak_massive_sovereign_knowledge_5k.jsonl")),
        ("sehaj_frontier_reasoning", load_frontier_reasoning),
        ("sehaj_teacher_factory_tier1", load_teacher_factory),
    ]

    seen_pairs, seen_prompts = set(), set()
    kept_all, report = [], {}
    for name, loader in sources:
        rows, counts = loader()
        counts.update({"drop_empty_prompt": 0, "drop_response_too_short": 0,
                       "drop_response_too_long": 0, "drop_latin_heavy_response": 0,
                       "drop_global_exact_dup": 0, "drop_prompt_near_dup": 0,
                       "kept": 0})
        for prompt, response in rows:
            reason = global_filter(prompt, response)
            if reason:
                counts[f"drop_{reason}"] += 1
                continue
            pair = (prompt_key(prompt), prompt_key(response))
            if pair in seen_pairs:
                counts["drop_global_exact_dup"] += 1
                continue
            pk = prompt_key(prompt)
            if pk in seen_prompts:
                counts["drop_prompt_near_dup"] += 1
                continue
            seen_pairs.add(pair)
            seen_prompts.add(pk)
            kept_all.append({"prompt": prompt, "response": response, "source": name})
            counts["kept"] += 1
        report[name] = counts
        print(f"{name}: raw={counts['raw']} kept={counts['kept']}")

    # verify EOS-supervised tokenization on a sample (trainer contract)
    if sp is not None:
        enc = encode_sft_example(kept_all[0]["prompt"], kept_all[0]["response"], sp)
        assert enc["input_ids"][-1] == EOS_ID and enc["labels"][-1] == EOS_ID
        print("encode_sft_example sanity: last input_id == last label == EOS(3) OK")

    rng = random.Random(SEED)
    rng.shuffle(kept_all)
    n_val = max(1, int(len(kept_all) * VAL_FRAC))
    val, train = kept_all[:n_val], kept_all[n_val:]

    def dump(path, rows):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"prompt": r["prompt"], "response": r["response"]},
                                    ensure_ascii=False) + "\n")

    dump(os.path.join(HERE, "sft_data.jsonl"), train)
    dump(os.path.join(HERE, "sft_val.jsonl"), val)

    g_tot = l_tot = 0
    for r in train:
        for ch in r["prompt"] + " " + r["response"]:
            o = ord(ch)
            if GURMUKHI_LO <= o <= GURMUKHI_HI:
                g_tot += 1
            elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                l_tot += 1
    gurm_ratio = g_tot / max(1, g_tot + l_tot)

    print("\n===== SFT BUILD REPORT =====")
    for name, c in report.items():
        drops = {k: v for k, v in c.items() if k.startswith("drop_") and v}
        print(f"{name}: raw={c['raw']} kept={c['kept']} drops={drops}")
    print(f"\nTOTAL: train={len(train)} val={len(val)} (disjoint, seed={SEED})")
    print(f"final-set Gurmukhi ratio (Gurmukhi / Gurmukhi+Latin letters): {gurm_ratio:.4f}")
    print("\n----- 5 random final examples (verbatim) -----")
    for r in rng.sample(train, 5):
        print(json.dumps({"prompt": r["prompt"], "response": r["response"]},
                         ensure_ascii=False, indent=2))
        print("---")

    with open(os.path.join(HERE, "sft_build_report.json"), "w") as fh:
        json.dump({"per_source": report, "train": len(train), "val": len(val),
                   "gurmukhi_ratio": gurm_ratio, "seed": SEED}, fh,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
