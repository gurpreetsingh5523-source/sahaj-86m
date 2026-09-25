"""SEHAJ v7.0 FINAL - Hybrid Selective-SSM + Causal-Attention char-level LM.
Winner of the v7 low-data architecture sweep: hybrid block (SelectiveSSM + CausalAttention
+ SwiGLU, pre-norm RMSNorm, residuals, tied embeddings) with proper initialization
(embed N(0,0.02); output projections std=0.02/sqrt(2*n_layers)) and dropout 0.1.
Sweep-winning training recipe: AdamW lr=2e-3, betas=(0.9,0.95), wd=0.1 (2D+ non-embedding
weights), cosine decay to 10% with 30-step warmup, grad clip 1.0.
Fixes vs v6: correct init (initial loss ~ ln(vocab)), RoPE lazily extends past max_seq_len,
optional n_loops weight-sharing (ceil(n_layers/n_loops) unique blocks applied cyclically).
v7.1: SelectiveSSM uses a fused numba scan (custom autograd.Function, parallel CPU kernel)
for T >= 2*chunk_size on CPU/fp32 — same recurrence, ~4x faster end-to-end training step;
the original Python-loop path remains for short sequences, generation (T=1), and non-CPU.
Run `python3 SEHAJ_v7_FINAL.py` to execute the built-in self-test suite.
"""
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


@dataclass
class Config:
    vocab_size: int = 65
    d_model: int = 192
    d_state: int = 64
    n_heads: int = 6
    n_layers: int = 3
    d_ff: int = 768
    max_seq_len: int = 256
    dropout: float = 0.1
    tie_embeddings: bool = True
    n_loops: int = 1


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__(); self.a = nn.Linear(d, h, bias=False); self.b = nn.Linear(d, h, bias=False); self.o = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.o(F.silu(self.a(x)) * self.b(x))


class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048, theta: float = 10000.0):
        super().__init__(); self.dim = dim; self.theta = theta; self._build(max_len, torch.device("cpu"))
    def _build(self, n: int, device):
        inv = 1 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        freqs = torch.outer(torch.arange(n, device=device).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False); self.register_buffer("sin", freqs.sin(), persistent=False)
    def forward(self, x, offset: int = 0):
        t = x.size(-2)
        if offset + t > self.cos.size(0) or self.cos.device != x.device:
            self._build(max(offset + t, 2 * self.cos.size(0)), x.device)
        cos = self.cos[offset:offset + t][None, None]; sin = self.sin[offset:offset + t][None, None]
        xe = x[..., 0::2]; xo = x[..., 1::2]
        return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).flatten(-2)


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _ssm_scan_fwd(x, b, c, delta, A, Dv, h0, y, hT):
        B, T, D = x.shape; N = b.shape[2]
        for bd in prange(B * D):
            bi = bd // D; d = bd % D
            h = h0[bi, d].copy()
            for t in range(T):
                xt = x[bi, t, d]
                for n in range(N):
                    h[n] = np.exp(delta[bi, t, n] * A[d, n]) * h[n] + xt * b[bi, t, n] * delta[bi, t, n]
                acc = Dv[d] * xt
                for n in range(N):
                    acc += h[n] * c[bi, t, n]
                y[bi, t, d] = acc
            for n in range(N):
                hT[bi, d, n] = h[n]

    @njit(parallel=True, cache=True)
    def _ssm_scan_bwd(dy, dhT, x, b, c, delta, A, Dv, h0,
                      dx, db, dc, ddelta, dA_all, dDv_all, dh0):
        B, T, D = x.shape; N = b.shape[2]
        h_all = np.empty((B, T, D, N), dtype=np.float32)
        # pass 1: recompute the h trajectory (parallel over B*D rows)
        for bd in prange(B * D):
            bi = bd // D; d = bd % D
            h = h0[bi, d].copy()
            for t in range(T):
                xt = x[bi, t, d]
                for n in range(N):
                    h[n] = np.exp(delta[bi, t, n] * A[d, n]) * h[n] + xt * b[bi, t, n] * delta[bi, t, n]
                for n in range(N):
                    h_all[bi, t, d, n] = h[n]
        # pass 2: reverse recurrence for grads (parallel over B; reductions stay in-thread,
        # dA/dDv accumulate into per-batch rows and are summed outside — no races)
        for bi in prange(B):
            r = dhT[bi].copy()  # (D,N) grad wrt h_t from the future
            g = np.empty((D, N), dtype=np.float32)
            for t in range(T - 1, -1, -1):
                for d in range(D):
                    dyt = dy[bi, t, d]
                    for n in range(N):
                        g[d, n] = r[d, n] + dyt * c[bi, t, n]
                for n in range(N):
                    sc = 0.0; sb = 0.0; sd = 0.0
                    bn = b[bi, t, n]; dn = delta[bi, t, n]
                    for d in range(D):
                        hp = h_all[bi, t - 1, d, n] if t > 0 else h0[bi, d, n]
                        ab = np.exp(dn * A[d, n])
                        gg = g[d, n]
                        sc += dy[bi, t, d] * h_all[bi, t, d, n]
                        sb += gg * x[bi, t, d] * dn
                        sd += gg * (A[d, n] * ab * hp + x[bi, t, d] * bn)
                        dA_all[bi, d, n] += dn * ab * hp * gg
                    dc[bi, t, n] = sc
                    db[bi, t, n] = sb
                    ddelta[bi, t, n] = sd
                for d in range(D):
                    s = Dv[d] * dy[bi, t, d]
                    for n in range(N):
                        s += g[d, n] * b[bi, t, n] * delta[bi, t, n]
                    dx[bi, t, d] = s
                    dDv_all[bi, d] += dy[bi, t, d] * x[bi, t, d]
                for d in range(D):
                    for n in range(N):
                        r[d, n] = np.exp(delta[bi, t, n] * A[d, n]) * g[d, n]
            for d in range(D):
                for n in range(N):
                    dh0[bi, d, n] = r[d, n]


class _SSMScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b, c, delta, A, Dv, h0):
        y = torch.empty_like(x); hT = torch.empty_like(h0)
        _ssm_scan_fwd(x.numpy(), b.numpy(), c.numpy(), delta.numpy(),
                      A.numpy(), Dv.numpy(), h0.numpy(), y.numpy(), hT.numpy())
        ctx.save_for_backward(x, b, c, delta, A, Dv, h0)
        return y, hT

    @staticmethod
    def backward(ctx, dy, dhT):
        x, b, c, delta, A, Dv, h0 = ctx.saved_tensors
        B = x.shape[0]
        dy = dy.contiguous(); dhT = dhT.contiguous()
        dx = torch.empty_like(x); dh0 = torch.empty_like(h0)
        db = torch.empty_like(b); dc = torch.empty_like(c); ddelta = torch.empty_like(delta)
        dA_all = torch.zeros(B, *A.shape); dDv_all = torch.zeros(B, *Dv.shape)
        _ssm_scan_bwd(dy.numpy(), dhT.numpy(), x.numpy(), b.numpy(), c.numpy(),
                      delta.numpy(), A.numpy(), Dv.numpy(), h0.numpy(),
                      dx.numpy(), db.numpy(), dc.numpy(), ddelta.numpy(),
                      dA_all.numpy(), dDv_all.numpy(), dh0.numpy())
        return dx, db, dc, ddelta, dA_all.sum(0), dDv_all.sum(0), dh0


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0, chunk_size: int = 16):
        super().__init__(); self.d_model = d_model; self.d_state = d_state
        self.param = nn.Linear(d_model, 2 * d_state + 1, bias=False)
        self.dt = nn.Linear(1, d_state, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float())[None].repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model)); self.out = nn.Linear(d_model, d_model, bias=False); self.drop = nn.Dropout(dropout)
        self.chunk_size = chunk_size
    def _loop(self, x, b, c, delta, A, h):
        B, T, D = x.shape; ys = []
        for t in range(T):
            abar = torch.exp(delta[:, t, None, :] * A[None])
            inp = x[:, t, :, None] * b[:, t, None, :] * delta[:, t, None, :]
            h = abar * h + inp
            ys.append((h * c[:, t, None, :]).sum(-1) + self.D * x[:, t])
        return torch.stack(ys, 1), h
    def forward(self, x, state=None):
        B, T, D = x.shape; p = self.param(x); b, c, raw = p.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt(raw)).clamp(max=10.0)
        A = -torch.exp(self.A_log).to(x.dtype)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype) if state is None else state
        if _HAS_NUMBA and T >= 2 * self.chunk_size and x.is_cpu and x.dtype == torch.float32:
            y, h = _SSMScan.apply(x.contiguous(), b.contiguous(), c.contiguous(),
                                  delta.contiguous(), A.contiguous(), self.D, h.contiguous())
        else:  # short seqs / generation / non-CPU: original recurrence
            y, h = self._loop(x, b, c, delta, A, h)
        return self.drop(self.out(y)), h


class CausalAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__(); assert c.d_model % c.n_heads == 0; self.h = c.n_heads; self.d = c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False); self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.rope = RoPE(self.d, c.max_seq_len); self.dropout = c.dropout
    def forward(self, x, cache=None):
        B, T, D = x.shape; q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2); k = k.view(B, T, self.h, self.d).transpose(1, 2); v = v.view(B, T, self.h, self.d).transpose(1, 2)
        offset = 0 if cache is None else cache[0].size(-2); q = self.rope(q, offset); k = self.rope(k, offset)
        if cache is not None: k = torch.cat((cache[0], k), -2); v = torch.cat((cache[1], v), -2)
        K = k.size(-2); past = K - T
        mask = None
        if past: mask = torch.ones(T, K, device=x.device, dtype=torch.bool).tril(diagonal=past)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=(past == 0))
        y = y.transpose(1, 2).reshape(B, T, D); return self.o(y), (k.detach(), v.detach())


class HybridBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__(); self.n1 = RMSNorm(c.d_model); self.ssm = SelectiveSSM(c.d_model, c.d_state, c.dropout)
        self.n2 = RMSNorm(c.d_model); self.attn = CausalAttention(c)
        self.n3 = RMSNorm(c.d_model); self.ff = SwiGLU(c.d_model, c.d_ff); self.drop = nn.Dropout(c.dropout)
    def forward(self, x, ssm_state=None, kv_cache=None):
        y, s = self.ssm(self.n1(x), ssm_state); x = x + self.drop(y)
        y, k = self.attn(self.n2(x), kv_cache); x = x + self.drop(y)
        x = x + self.drop(self.ff(self.n3(x))); return x, s, k


class SEHAJ(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        n_unique = math.ceil(c.n_layers / c.n_loops)
        self.unroll = [i % n_unique for i in range(c.n_layers)]
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([HybridBlock(c) for _ in range(n_unique)])
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings: self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        out_std = 0.02 / math.sqrt(2 * len(self.unroll))
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, (SelectiveSSM,)): nn.init.normal_(m.out.weight, mean=0.0, std=out_std)
            elif isinstance(m, CausalAttention): nn.init.normal_(m.o.weight, mean=0.0, std=out_std)
            elif isinstance(m, SwiGLU): nn.init.normal_(m.o.weight, mean=0.0, std=out_std)

    def forward(self, ids, states=None, caches=None):
        x = self.embed(ids); ns = []; nc = []
        for i, bi in enumerate(self.unroll):
            s = None if states is None else states[i]; k = None if caches is None else caches[i]
            x, s, k = self.blocks[bi](x, s, k); ns.append(s); nc.append(k)
        return self.head(self.norm(x)), ns, nc

    @torch.no_grad()
    def generate(self, ids, max_new: int = 64, temperature: float = 1.0, top_k: int = 40):
        states = caches = None
        for _ in range(max_new):
            logits, states, caches = self(ids[:, -1:] if states is not None else ids, states, caches)
            z = logits[:, -1] / max(temperature, 1e-5)
            if top_k: v, _ = torch.topk(z, min(top_k, z.size(-1))); z[z < v[:, -1, None]] = -float("inf")
            nxt = torch.multinomial(F.softmax(z, -1), 1); ids = torch.cat((ids, nxt), 1)
        return ids


def _test_config(**kw) -> Config:
    base = dict(vocab_size=65, d_model=96, d_state=32, n_heads=4, n_layers=2, d_ff=192, max_seq_len=64, dropout=0.1)
    base.update(kw); return Config(**base)


def self_test() -> bool:
    torch.manual_seed(0); torch.set_num_threads(max(1, torch.get_num_threads()))
    results = []
    def check(name, ok, detail=""):
        results.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    c = _test_config()
    m = SEHAJ(c); m.train()
    x = torch.randint(0, c.vocab_size, (2, 32))
    logits, _, _ = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, c.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()
    loss = loss.detach()
    check("smoke: forward+backward finite", bool(torch.isfinite(loss)), f"loss={float(loss):.4f}")

    m.eval()
    with torch.no_grad():
        l0 = F.cross_entropy(m(x)[0].reshape(-1, c.vocab_size), x.reshape(-1)).item()
    check("init sanity: init loss ~ ln(vocab)", abs(l0 - math.log(c.vocab_size)) < 0.5, f"init={l0:.4f} ln(vocab)={math.log(c.vocab_size):.4f}")

    ids = torch.randint(0, c.vocab_size, (1, 20))
    with torch.no_grad():
        full, _, _ = m(ids)
        lo, st, ca = m(ids[:, :10])  # prefill 10
        outs = [lo]
        for t in range(10, 20):  # decode 10 one-by-one
            lo, st, ca = m(ids[:, t:t + 1], st, ca); outs.append(lo)
        cached = torch.cat(outs, 1)
    diff = (full - cached).abs().max().item()
    check("KV equivalence: prefill+decode == full forward", diff < 1e-4, f"max_diff={diff:.2e}")

    m_loop = SEHAJ(_test_config(n_layers=4, n_loops=2)); m_full = SEHAJ(_test_config(n_layers=4, n_loops=1))
    p_loop = sum(p.numel() for p in m_loop.parameters()); p_full = sum(p.numel() for p in m_full.parameters())
    lo, _, _ = m_loop(ids)
    check("looped mode: fewer params, forward works", p_loop < p_full and lo.shape == (1, 20, c.vocab_size),
          f"n_loops=2 params={p_loop:,} vs n_loops=1 params={p_full:,}")

    long_ids = torch.randint(0, c.vocab_size, (1, 2 * c.max_seq_len))
    try:
        with torch.no_grad(): m(long_ids)
        check("long-seq: 2x max_seq_len forward", True, f"T={2 * c.max_seq_len} > max_seq_len={c.max_seq_len}")
    except Exception as e:
        check("long-seq: 2x max_seq_len forward", False, repr(e))

    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    xb = torch.randint(0, c.vocab_size, (4, 48)); yb = torch.randint(0, c.vocab_size, (4, 48))
    first = None
    for step in range(50):
        lo, _, _ = m(xb)
        l = F.cross_entropy(lo.reshape(-1, c.vocab_size), yb.reshape(-1))
        if first is None: first = l.item()
        opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    drop = (first - l.item()) / first
    check("overfit: 50 steps drop loss >50%", drop > 0.5, f"loss {first:.4f} -> {l.item():.4f} ({100 * drop:.1f}% drop)")

    m.eval()
    out = m.generate(torch.randint(0, c.vocab_size, (1, 5)), max_new=10)
    check("generate: (1, prompt+10) tokens", tuple(out.shape) == (1, 15), f"shape={tuple(out.shape)}")

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} tests passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
