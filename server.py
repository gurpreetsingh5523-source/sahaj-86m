"""SEHAJ v7 Punjabi — FastAPI server, drop-in replacement for the Sevak-24M backend.
Matches the existing iOS app contract: POST /v1/chat {prompt, temperature, max_bytes, top_p}
-> ChatResponse {reply, tool_called, model, latency_ms, audio_text, content_type, backend};
GET /v1/health. Run: python3 server.py [--port 8001] [--checkpoint sft_checkpoint_best.pt]
"""
import argparse
import time
import subprocess
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from inference import load_model, generate, WORK
from rag_qa import build_index, extractive_answer, looks_like_factual_question
from build_sft import PROMPT_TEMPLATE  # same "ਸਵਾਲ: ...\nਜਵਾਬ: " wrapper the SFT data was trained with

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8001)
# 2026-09-15: was defaulting to checkpoint_best.pt (the PRETRAIN checkpoint,
# never SFT'd) — chat quality looked broken in testing purely because of this,
# nothing to do with RAG. sft_checkpoint_best.pt is the round-2 (2026-09-13)
# checkpoint, the one actually decided to ship.
ap.add_argument("--checkpoint", default=f"{WORK}/sft_checkpoint_best.pt")
ap.add_argument("--rag-threshold", type=float, default=0.5,
                help="min extractive-retrieval confidence to trust a Wikipedia-pa answer "
                     "over the model's own generation (see rag_qa.py --calibrate)")
args = ap.parse_args()

model, sp, ck = load_model(args.checkpoint)
CKPT_INFO = f"step={ck.get('step')} tokens={ck.get('tokens_seen', 0):,}"
RAG_INDEX = build_index()

app = FastAPI(title="SEHAJ v7 Punjabi Brain", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_bytes: Optional[int] = 256
    top_p: Optional[float] = 0.88


class ChatResponse(BaseModel):
    reply: str
    tool_called: str
    model: str
    latency_ms: float
    audio_text: str = ""
    content_type: str = "text"
    backend: str = ""


@app.get("/")
def root():
    return {"status": "online", "system": "SEHAJ v7 Punjabi Brain",
            "checkpoint": CKPT_INFO, "version": "1.0"}


@app.post("/v1/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt provided.")
    t0 = time.time()
    # 2026-09-15: the 30M model's own ground-truth QA accuracy is ~0% (capacity
    # limit, not fixable by more SFT — see eval_report.md). For factual-looking
    # questions, prefer a grounded Wikipedia-pa extractive answer (never
    # invented, always cites a real article) over the model hallucinating.
    if looks_like_factual_question(req.prompt):
        hit = extractive_answer(req.prompt, RAG_INDEX)
        if hit and hit["confidence"] >= args.rag_threshold:
            dt = (time.time() - t0) * 1000
            return ChatResponse(reply=hit["answer"], tool_called="sehaj_v7_rag",
                                model=f"SEHAJ-v7-30M+RAG ({CKPT_INFO})", latency_ms=round(dt, 2),
                                audio_text=hit["answer"], content_type="text",
                                backend=f"wikipedia_pa:{hit['citation']}")
    max_new = max(16, (req.max_bytes or 256) // 3)
    reply, stopped = generate(model, sp, PROMPT_TEMPLATE.format(prompt=req.prompt), max_new=max_new,
                              temperature=req.temperature or 0.7,
                              top_p=req.top_p or 0.9)
    dt = (time.time() - t0) * 1000
    return ChatResponse(reply=reply, tool_called="sehaj_v7" if stopped else "sehaj_v7_no_eos",
                        model=f"SEHAJ-v7-30M ({CKPT_INFO})", latency_ms=round(dt, 2),
                        audio_text=reply, content_type="text", backend="sehaj_v7_30m")


@app.get("/v1/health")
def health():
    try:
        ip = subprocess.run(["ipconfig", "getifaddr", "en0"],
                            capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        ip = ""
    return {"status": "online", "brain": f"sehaj_v7_30m ({CKPT_INFO})",
            "lan_url": f"http://{ip}:{args.port}" if ip else None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
