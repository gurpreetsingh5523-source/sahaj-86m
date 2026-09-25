"""SEHAJ v7 Punjabi — inference CLI.
Loads a training checkpoint + sevak_bpe_6k tokenizer, generates with proper EOS stopping.

Usage:
  python3 inference.py --checkpoint checkpoint_best.pt --prompt "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"
  python3 inference.py --checkpoint checkpoint_best.pt --chat
"""
import argparse
import torch
import sentencepiece as spm
from SEHAJ_v7_FINAL import SEHAJ, Config

WORK = "/Users/gurpreetdhillon/sehaj_v7_punjabi"
EOS_ID, BOS_ID, PAD_ID = 3, 2, 0


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    model = SEHAJ(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    sp = spm.SentencePieceProcessor(model_file=f"{WORK}/sevak_bpe_6k.model")
    return model, sp, ck


@torch.no_grad()
def generate(model, sp, prompt, max_new=120, temperature=0.7, top_p=0.9,
             greedy=False, device="cpu", rep_penalty=1.15):
    ids = [BOS_ID] + sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    states = caches = None
    logits, states, caches = model(x)
    stopped = False
    for _ in range(max_new):
        z = logits[:, -1].float()
        seen = torch.tensor([list(set(ids))], device=z.device)
        z.scatter_(1, seen, z.gather(1, seen) / rep_penalty)
        if greedy or temperature <= 0:
            nxt = z.argmax(-1, keepdim=True)
        else:
            z = z / max(temperature, 1e-5)
            s, _ = torch.sort(z, descending=True)
            cum = torch.cumsum(torch.softmax(s, -1), -1)
            cut = (cum > top_p).float().argmax(-1)
            thresh = s.gather(-1, cut[:, None])
            z = torch.where(z < thresh, torch.full_like(z, float("-inf")), z)
            nxt = torch.multinomial(torch.softmax(z, -1), 1)
        if nxt.item() in (EOS_ID, PAD_ID):
            stopped = True
            break
        ids.append(nxt.item())
        logits, states, caches = model(nxt, states, caches)
    return sp.decode(ids[len(sp.encode(prompt)) + 1:]), stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=f"{WORK}/checkpoint_best.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    model, sp, ck = load_model(args.checkpoint)
    print(f"loaded {args.checkpoint} | step={ck.get('step')} "
          f"tokens_seen={ck.get('tokens_seen', 0):,} best_val={ck.get('best_val')}")

    def reply(p):
        out, stopped = generate(model, sp, p, args.max_new, args.temperature,
                                args.top_p, args.greedy)
        return out + ("" if stopped else " [NO-EOS: hit token cap]")

    if args.chat:
        print("SEHAJ v7 chat — 'quit' to exit")
        while True:
            p = input("\ntusi> ").strip()
            if p.lower() in ("quit", "exit"):
                break
            print("sehaj>", reply(p))
    else:
        print(reply(args.prompt or "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"))


if __name__ == "__main__":
    main()
