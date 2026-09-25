# SEHAJ / Sahaj v8 (86M) — Punjabi Language Model

An 86M-parameter byte/BPE-level Punjabi language model (hybrid Selective-SSM +
causal-attention architecture), trained from scratch on a curated Punjabi
corpus, with a retrieval-augmented (RAG) factual-QA layer and a trained
confidence/abstention probe.

Weights: https://huggingface.co/Nam-toon-studio/sahaj-86m

## What this is

- **Text-only.** Input text, output text. No audio, no vision, no code
  generation — none of those were trained on.
- **Punjabi-first.** Some SFT examples use English *instructions* asking for
  a Punjabi answer, but the model was not trained for English conversation.
- **Small and fast**, meant for local/on-device use, not a general-purpose
  assistant.

## Honest numbers (fixed scorer, see `evaluate.py`)

| | value |
|---|---|
| params | 86,036,224 |
| pretrain val_loss | 2.86 (86M) vs 3.10 (previous 30M model) |
| raw factual QA accuracy (no RAG) | 12% |
| factual QA accuracy (with RAG, `rag_qa.py`) | ~40% (retrieval is checkpoint-agnostic, measured on the shared corpus) |
| EOS stop rate | 50% (varies by prompt template — see note below) |
| confidence-head AUROC (predicts whether its own answer is correct) | 0.70, held-out |

**Prompt-format sensitivity, worth knowing:** the model was trained on the
literal template `"ਸਵਾਲ: {prompt}\nਜਵਾਬ: "`. Prompts sent through that exact
template get meaningfully more coherent, less repetitive answers than bare/
informal prompts. `server.py` applies this template automatically.

**A general capacity note, not a defect:** at 86M parameters this model
cannot memorize broad world knowledge — the RAG layer (retrieval over a local
Wikipedia-pa index) is what carries factual QA, not the model's own weights.
Treat it as a fast, private, small conversational layer with a grounded
lookup tool behind it, not an encyclopedia.

## Run it

```bash
pip install -r requirements.txt
python3 server.py --checkpoint sft_checkpoint_best.pt --port 8001
# POST /v1/chat {"prompt": "ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?"}
```

Or directly:

```bash
python3 inference.py --checkpoint sft_checkpoint_best.pt --prompt "ਸਵਾਲ: ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ\nਜਵਾਬ:"
```

Download `sft_checkpoint_best.pt`, the tokenizer (`sevak_bpe_6k.model`), and
config from the [HF repo](https://huggingface.co/Nam-toon-studio/sahaj-86m)
and place them alongside this code. (`model_fp16.safetensors` is also on HF,
a smaller weights-only export, but the current `inference.py`/`server.py`
load the `.pt` checkpoint format — use that one to run out of the box.)

## What's *not* included here

Training code, the RAG index builder, and the confidence-head trainer live in
the author's private research repo — this is the inference/serving code only.
