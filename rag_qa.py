"""SEHAJ v7 Punjabi — retrieval-augmented QA over the local Wikipedia-pa cache.

Why: the 30M model cannot memorize broad facts (ground-truth QA accuracy ~0%,
see eval_report.md) — that's a capacity limit, not fixable by more SFT at this
scale. This is the cheap fix that doesn't need renting a bigger training rig:
answer factual questions by RETRIEVING from the same wikipedia_pa corpus that
was already downloaded for pretraining, instead of asking the LLM to recall.

Two-stage retrieval (article-level BM25 is fast at 51k docs; virasat_notebook's
per-chunk trigram index is NOT reused here — a trigram-set per chunk over the
full corpus would blow up memory alongside the loaded model):
  1. coarse: inverted-index BM25 over whole articles (title-boosted)
  2. fine:   sentence-level extractive match within the top articles only

Pure extractive — no LLM call, no network, nothing to rent. Always grounded:
the answer is always a real sentence that exists in a real Wikipedia article,
never a generated/invented one.

Usage:
  python3 rag_qa.py --build                 # build + cache the BM25 index
  python3 rag_qa.py --ask "ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕਿਹੜੀ ਹੈ?"
  python3 rag_qa.py --eval                  # honest accuracy vs sevak_factual_ground_truth.jsonl
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

WORK = os.path.dirname(os.path.abspath(__file__))
HF = os.path.expanduser("~/.cache/huggingface/hub")
INDEX_CACHE = os.path.join(WORK, "rag_index.pkl")

# same footer-stripping as prepare_data.py, copied (not imported — that module
# executes the whole data-prep pipeline at import time)
WIKI_FOOTER_RE = re.compile(
    r"\n==\s*(ਹਵਾਲੇ|ਬਾਹਰੀ ਲਿੰਕ|ਇਹ ਵੀ ਦੇਖੋ|References|External links|See also)\s*==.*",
    re.S,
)

_GURMUKHI = re.compile(r"[਀-੿]")
_TOKEN = re.compile(r"[਀-੿]+|[A-Za-z]+|[0-9੦-੯]+")
_SENT = re.compile(r"(?<=[।॥\.\?!])\s+")

# interrogatives + high-frequency function words — stripped before scoring so
# "ਕੀ...ਹੈ" doesn't farm matches on every sentence in the corpus
STOPWORDS = {
    "ਕੀ", "ਕੌਣ", "ਕਦੋਂ", "ਕਿੱਥੇ", "ਕਿਹੜਾ", "ਕਿਹੜੀ", "ਕਿਹੜੇ", "ਕਿਵੇਂ", "ਕਿਉਂ", "ਕਿੰਨੇ", "ਕਿੰਨਾ", "ਕਿਸ",
    "ਹੈ", "ਹਨ", "ਸੀ", "ਸਨ", "ਹੋ", "ਹੋਏ", "ਹੋਈ",
    "ਦਾ", "ਦੀ", "ਦੇ", "ਦਿਆਂ", "ਨੇ", "ਨੂੰ", "ਨਾਲ", "ਤੋਂ", "ਤੇ", "ਵਿੱਚ", "ਵਿਚ", "ਵਿੱਚੋਂ",
    "ਇਹ", "ਉਹ", "ਇੱਕ", "ਅਤੇ", "ਜੀ", "ਜੋ", "ਜਿਸ", "ਜਿਨ੍ਹਾਂ", "ਸਭ", "ਵੀ", "ਹੀ", "ਪਰ",
}


def gurmukhi_ratio(text: str) -> float:
    letters = re.findall(r"[਀-੿A-Za-z]", text)
    if not letters:
        return 0.0
    return len(_GURMUKHI.findall(text)) / len(letters)


def clean_text(t: str) -> str:
    t = WIKI_FOOTER_RE.split(t)[0]
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    return [t for t in toks if t not in STOPWORDS] if drop_stop else toks


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1: article-level inverted-index BM25 (fast at 51k docs)
# ═══════════════════════════════════════════════════════════════════════════
class ArticleIndex:
    def __init__(self, titles: list[str], bodies: list[str]):
        self.titles = titles
        self.bodies = bodies
        self.N = len(bodies)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doclen = [0] * self.N
        title_toks_all = []
        for i, (title, body) in enumerate(zip(titles, bodies)):
            toks = tokens(body)
            self.doclen[i] = len(toks)
            tf = Counter(toks)
            for term, f in tf.items():
                self.postings[term].append((i, f))
            title_toks_all.append(set(tokens(title)))
        self.title_toks = title_toks_all
        self.avgdl = (sum(self.doclen) / self.N) if self.N else 1.0
        self.df = {term: len(plist) for term, plist in self.postings.items()}

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75,
               title_boost: float = 2.0) -> list[tuple[float, int]]:
        q = tokens(query)
        scores: dict[int, float] = defaultdict(float)
        for term in set(q):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for doc_id, f in plist:
                dl = self.doclen[doc_id]
                denom = f + k1 * (1 - b + b * dl / self.avgdl)
                scores[doc_id] += idf * f * (k1 + 1) / denom
        # title boost requires the QUERY to cover most of the title's own
        # words (not just any single generic word in common) — otherwise a
        # long unrelated title sharing one common token (e.g. both titles
        # containing "ਏ") would out-rank the real answer via boost alone.
        qset = set(q)

        def _title_overlap_ratio(ttoks: set) -> float:
            return len(qset & ttoks) / len(ttoks) if ttoks else 0.0

        # a single generic word overlapping a short 2-word title (e.g. both
        # "ਪੰਜਾਬ" and "ਪੱਛਮੀ ਪੰਜਾਬ" sharing "ਪੰਜਾਬ") hits ratio>=0.5 too easily,
        # so also require an absolute overlap of >=2 title words.
        def _boosted(ttoks: set) -> bool:
            overlap = len(qset & ttoks)
            return overlap >= 2 and _title_overlap_ratio(ttoks) >= 0.5

        for doc_id in list(scores.keys()):
            if _boosted(self.title_toks[doc_id]):
                ratio = _title_overlap_ratio(self.title_toks[doc_id])
                scores[doc_id] += title_boost * ratio * len(self.title_toks[doc_id])
        # also surface pure title matches even if body BM25 missed them
        for doc_id, ttoks in enumerate(self.title_toks):
            if doc_id in scores:
                continue
            if _boosted(ttoks):
                scores[doc_id] = title_boost * _title_overlap_ratio(ttoks) * len(ttoks)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(sc, i) for i, sc in ranked]


def load_corpus():
    wpq = glob.glob(os.path.join(HF, "datasets--wikimedia--wikipedia/snapshots/*/20231101.pa/*.parquet"))
    if not wpq:
        raise RuntimeError("wikipedia_pa parquet not found in HF cache — expected from prepare_data.py's source")
    import pandas as pd
    df = pd.read_parquet(wpq[0])
    titles, bodies = [], []
    for title, text in zip(df["title"], df["text"]):
        t = clean_text(str(text))
        if len(t) < 100 or gurmukhi_ratio(t) < 0.6:
            continue
        titles.append(str(title))
        bodies.append(t)
    return titles, bodies


def build_index(force: bool = False) -> ArticleIndex:
    if not force and os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    titles, bodies = load_corpus()
    idx = ArticleIndex(titles, bodies)
    with open(INDEX_CACHE, "wb") as fh:
        pickle.dump(idx, fh)
    print(f"built index: {idx.N} articles in {time.time()-t0:.1f}s -> {INDEX_CACHE}")
    return idx


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2: sentence-level extractive match, within top articles only
# ═══════════════════════════════════════════════════════════════════════════
def _trigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", " ", text)
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else set()


def extractive_answer(question: str, idx: ArticleIndex, k_articles: int = 5,
                       max_sents: int = 3, min_score: float = 0.22):
    """Never echoes the question — scoring uses stopword-stripped content
    words only, so the answer can't just be a rephrasing of the question.

    A sentence can score high purely on a single common word (e.g. "ਤੁਹਾਡਾ"
    = "your") even when its article has nothing to do with the query — seen
    live: "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?" (what's your name) matched a jaggery-remedies
    article via "ਤੁਹਾਡਾ ਗਲਾ ਦੁਖਦਾ ਹੈ" (your throat hurts). Require the
    article's own TITLE to share a content word with the query as a topical
    anchor; without one, demand much stronger sentence evidence instead of
    trusting a single incidental word match."""
    hits = idx.search(question, k=k_articles)
    if not hits:
        return None
    qt = set(tokens(question))
    qg = _trigrams(question)
    cands = []
    for score, doc_id in hits:
        title, body = idx.titles[doc_id], idx.bodies[doc_id]
        anchored = bool(qt & idx.title_toks[doc_id])
        floor = min_score if anchored else min_score + 0.35
        for sent in _SENT.split(body):
            sent = sent.strip()
            if len(sent) < 12:
                continue
            st = set(tokens(sent))
            if not st:
                continue
            overlap = len(qt & st) / max(1, len(qt))
            tri = len(qg & _trigrams(sent)) / max(1, len(qg))
            s = 0.75 * overlap + 0.25 * tri
            if s > floor:
                cands.append((s, sent, title, doc_id))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    # top sentences, deduped by article, most relevant article first
    seen_sent = set()
    picked = []
    for s, sent, title, doc_id in cands:
        if sent in seen_sent:
            continue
        seen_sent.add(sent)
        picked.append({"text": sent, "title": title, "score": round(s, 3)})
        if len(picked) >= max_sents:
            break
    confidence = picked[0]["score"] if picked else 0.0
    answer_text = " ".join(p["text"] for p in picked)
    return {"answer": answer_text, "confidence": confidence, "sentences": picked,
            "citation": picked[0]["title"] if picked else None}


# ═══════════════════════════════════════════════════════════════════════════
#  Eval — honest accuracy vs the same decontaminated ground-truth set + the
#  same answer_correct() substring/keyword matcher evaluate.py uses, so the
#  number is directly comparable to the model-only 0% baseline.
# ═══════════════════════════════════════════════════════════════════════════
def run_eval(idx: ArticleIndex, threshold: float = 0.12):
    sys.path.insert(0, WORK)
    from evaluate import KIT, answer_correct, load_sft_prompts, _norm_key

    path = os.path.join(KIT, "sevak_factual_ground_truth.jsonl")
    items = [json.loads(l) for l in open(path) if l.strip()]
    sft_prompts = load_sft_prompts()
    results, leaked = [], 0
    for it in items:
        q, expected = it["instruction"], it["response"]
        k = _norm_key(q)
        if any(k and (k in p or p in k) for p in sft_prompts):
            leaked += 1
            continue
        r = extractive_answer(q, idx, min_score=threshold)
        output = r["answer"] if r else ""
        correct = answer_correct(expected, output) if output else False
        routed = bool(r and looks_like_factual_question(q) and r["confidence"] >= 0.5)
        results.append({"question": q, "expected": expected, "output": output,
                        "citation": r["citation"] if r else None,
                        "confidence": r["confidence"] if r else 0.0,
                        "correct": correct, "covered": r is not None, "routed": routed,
                        "routed_correct": bool(routed and correct)})
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0.0
    covered = sum(r["covered"] for r in results)
    routed_n = sum(r["routed"] for r in results)
    routed_acc = sum(r["routed_correct"] for r in results) / n if n else 0.0
    out = {"items_total": len(items), "excluded_contaminated": leaked, "items_scored": n,
           "accuracy_unconditional": acc, "accuracy_as_routed": routed_acc, "routed_count": routed_n,
           "covered": covered, "per_item": results}
    out_path = os.path.join(WORK, f"eval_rag_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"RAG-extractive accuracy (unconditional, every item gets an extractive answer): "
          f"{acc*100:.1f}% ({sum(r['correct'] for r in results)}/{n})")
    print(f"RAG-extractive accuracy AS ROUTED BY server.py (question-gate + conf>=0.5, "
          f"else falls back to the model): {routed_acc*100:.1f}% ({sum(r['routed_correct'] for r in results)}/{n}, "
          f"{routed_n}/{n} items actually got routed to RAG) -> {out_path}")
    for r in results:
        mark = "OK " if r["correct"] else ("MISS" if r["covered"] else "N/A ")
        print(f"[{mark}] conf={r['confidence']:.2f} {r['question'][:50]}")
        if not r["correct"]:
            print(f"       expected: {r['expected'][:90]}")
            print(f"       got:      {r['output'][:90]}")
    return out


_QWORDS = {
    "ਕੀ", "ਕੌਣ", "ਕਦੋਂ", "ਕਿੱਥੇ", "ਕਿਹੜਾ", "ਕਿਹੜੀ", "ਕਿਹੜੇ", "ਕਿਵੇਂ", "ਕਿਉਂ",
    "ਕਿੰਨਾ", "ਕਿੰਨੇ", "ਕਿਸ", "what", "who", "when", "where", "which", "how", "why",
}


def looks_like_factual_question(prompt: str) -> bool:
    """Necessary (not sufficient) gate for routing to retrieval: calibration
    (--calibrate) showed the extractive confidence score alone does NOT
    separate factual questions from greetings/instructions/small-talk (e.g.
    "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ" scored 1.000, "Hello, how are you?" scored 0.710) — short
    queries spuriously match on common words. Require question *syntax* too.
    Uses whole-token membership, not substring search — a substring check
    would fire "ਕੀ" inside ਕੀਤਾ/ਕੀਤੀ or "how" inside "show".
    Known remaining false-positive risk: self-referential questions like
    "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?" (what's your name) are grammatically questions but not
    Wikipedia-answerable — mitigated (not fully solved) by extractive_answer's
    title-anchor requirement, not by this gate."""
    p = prompt.strip()
    if p.endswith("?") or "ਜਵਾਬ:" in p:
        return True
    toks = {t.lower() for t in _TOKEN.findall(p)}
    return bool(toks & _QWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ask", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.22)
    args = ap.parse_args()

    if args.calibrate:
        idx = build_index()
        conversational = [
            "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?",
            "ਨਿਰਦੇਸ਼: 'ਸੱਚ' ਸ਼ਬਦ ਦੀ ਵਰਤੋਂ ਕਰਦਿਆਂ ਇੱਕ ਵਾਕ ਲਿਖੋ।",
            "Hello, how are you?",
            "ਨਿਰਦੇਸ਼: ਸਿਹਤਮੰਦ ਰਹਿਣ ਲਈ ਤਿੰਨ ਸੁਝਾਅ ਦਿਓ।",
            "ਨਿਰਦੇਸ਼: ਪੰਜਾਬ ਦੀਆਂ ਤਿੰਨ ਮੁੱਖ ਫਸਲਾਂ ਦੇ ਨਾਮ ਦੱਸੋ।",
        ]
        for q in conversational:
            r = extractive_answer(q, idx, min_score=0.12)
            print(f"conv  conf={r['confidence'] if r else 0.0:.3f}  {q[:40]}")
        return

    if args.build:
        build_index(force=True)
        return
    idx = build_index()
    if args.ask:
        r = extractive_answer(args.ask, idx, min_score=args.threshold)
        print(json.dumps(r, ensure_ascii=False, indent=1) if r else "no retrieval hit")
    if args.eval:
        run_eval(idx, threshold=args.threshold)
    if not args.ask and not args.eval:
        run_eval(idx, threshold=args.threshold)


if __name__ == "__main__":
    main()
