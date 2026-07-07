"""Pure-PyTorch reference kernels for DeepSeek-V4 (single-step debug friendly).

The original CUDA/tilelang kernels live in ``kernel_tilelang.py``. They are fast but
opaque to a debugger: you cannot "step into" a JIT-compiled GPU kernel, so the interesting
structure (sparse attention, KV compression, hyper-connection Sinkhorn split, block
quantization) is invisible.

This file re-implements the same six functions used by ``model.py`` in plain torch:

    act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

plus a ``hadamard_transform`` fallback used by ``rotate_activation``.

Design notes
------------
* Everything is plain torch, so it runs on CPU or CUDA and every line is inspectable.
* The FP8/FP4 block-quantized matmuls are simulated by dequantizing to float32 and doing
  an ordinary matmul. Values are *close to* (not bit-identical to) the optimized kernels;
  for studying shape flow and structure this is exactly what you want.
* With ``ModelArgs.dtype == "bf16"`` and ``expert_dtype is None`` (the debug config in
  ``model.py``), ``linear()`` takes the plain ``F.linear`` branch, so ``fp8_gemm`` /
  ``fp4_gemm`` are not on the hot path. They are still implemented here as readable
  references.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

FP8_MAX = 448.0    # largest finite value of float8_e4m3
FP4_MAX = 6.0      # largest value of float4_e2m1


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """Smallest power of two >= x. Matches the e8m0 (exponent-only) scale rounding
    done by ``fast_round_scale`` in the tilelang kernels."""
    return torch.pow(2.0, torch.ceil(torch.log2(x)))


def _expand_block_scale(scale_2d: torch.Tensor, rows: int, cols: int,
                        block_r: int, block_c: int) -> torch.Tensor:
    """Expand a [ceil(rows/block_r), ceil(cols/block_c)] block scale up to [rows, cols]
    by repeating each entry over its block."""
    s = scale_2d.repeat_interleave(block_r, dim=0).repeat_interleave(block_c, dim=1)
    return s[:rows, :cols].contiguous()


# --------------------------------------------------------------------------- #
#  activation quantization
# --------------------------------------------------------------------------- #
def act_quant(
    x: torch.Tensor, block_size: int = 128,
    scale_fmt: Optional[str] = None, scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization (pure-torch reference of ``act_quant``).

    Splits the last dimension into blocks of ``block_size`` and quantizes each block to
    the FP8 range ``[-448, 448]`` using its own per-block scale ``scale = amax / 448``.

    * ``inplace=True``  : fused quant -> dequant. The rounded values are written back into
      ``x`` (same dtype as input). This *simulates* the FP8 round-off error on a bf16
      tensor, which is how the model mimics FP8 without real FP8 hardware.
    * ``inplace=False`` : returns ``(q, s)`` where ``q`` is FP8 (e4m3) and ``s`` the scale.

    When ``scale_fmt`` is set (or ``scale_dtype`` is e8m0), scales are snapped to powers
    of two (MXFP / ue8m0 format).
    """
    N = x.size(-1)
    assert N % block_size == 0, f"last dim {N} not divisible by block_size {block_size}"
    orig_dtype = x.dtype

    x_blk = x.float().unflatten(-1, (-1, block_size))                 # [..., #blocks, block_size]
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)     # [..., #blocks, 1]
    scale = amax * (1.0 / FP8_MAX)
    if scale_fmt is not None or scale_dtype == torch.float8_e8m0fnu:
        scale = _pow2_ceil(scale)                                     # snap to power of two

    q = (x_blk / scale).clamp(-FP8_MAX, FP8_MAX)                      # quantized values

    if inplace:
        x_sim = (q * scale).flatten(-2, -1).to(orig_dtype)            # dequant back, round-trip
        x.copy_(x_sim)
        return x

    s = scale.squeeze(-1)
    if scale_dtype == torch.float8_e8m0fnu:
        s = s.to(torch.float8_e8m0fnu)
    y = q.flatten(-2, -1).to(torch.float8_e4m3fn)
    return y, s


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = True) -> torch.Tensor:
    """Block-wise FP4 quantization (pure-torch reference of ``fp4_act_quant``).

    Same idea as :func:`act_quant` but targeting the FP4 range ``[-6, 6]`` with
    power-of-two (e8m0) scales, matching the FP4 expert / indexer QAT simulation.
    """
    N = x.size(-1)
    assert N % block_size == 0, f"last dim {N} not divisible by block_size {block_size}"
    orig_dtype = x.dtype

    x_blk = x.float().unflatten(-1, (-1, block_size))
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp_min(6 * (2 ** -126))
    scale = _pow2_ceil(amax * (1.0 / FP4_MAX))
    q = (x_blk / scale).clamp(-FP4_MAX, FP4_MAX)

    if inplace:
        x_sim = (q * scale).flatten(-2, -1).to(orig_dtype)
        x.copy_(x_sim)
        return x

    s = scale.squeeze(-1).to(torch.float8_e8m0fnu)
    y = q.flatten(-2, -1).to(torch.float4_e2m1fn_x2)
    return y, s


# --------------------------------------------------------------------------- #
#  quantized matmuls (reference; off the bf16 debug hot path)
# --------------------------------------------------------------------------- #
def _scale_to_f32(t: torch.Tensor) -> torch.Tensor:
    """Convert a scale tensor to float32. ``float8_e8m0fnu`` is exponent-only
    (value = 2**(e - 127)), so reinterpret the byte as uint8 and rebuild the float."""
    if t.dtype == torch.float8_e8m0fnu:
        u = t.view(torch.uint8).to(torch.int32)
        return torch.pow(2.0, (u - 127).to(torch.float32))
    return t.float()


def fp8_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[...,N] = A[...,K] @ B[N,K]^T with per-128 block FP8 scales (pure-torch reference).

    Dequantize the FP8 operands to float32 and do an ordinary matmul:
      * ``a``   is the FP8 activation with any leading dims (``[..., K]``);
        ``a_s`` is its per-128-along-K scale (``[..., K/128]``).
      * ``b``   is the FP8 weight ``[N, K]``; ``b_s`` is the 2-D block scale
        ``[ceil(N/128), K/128]`` (per 128 along both N and K).
    Returns bf16 (``torch.get_default_dtype()``), matching the optimized kernel.
    """
    lead = a.shape[:-1]
    K = a.size(-1)
    N = b.size(0)
    a2 = a.reshape(-1, K)                                                       # [M, K]
    a_d = a2.float() * _scale_to_f32(a_s).reshape(a2.size(0), -1).repeat_interleave(128, dim=1)
    b_d = b.float() * _expand_block_scale(_scale_to_f32(b_s), N, K, 128, 128)  # [N, K]
    c = a_d @ b_d.T                                                            # [M, N]
    return c.reshape(*lead, N).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[...,N] = A_fp8[...,K] @ B_fp4[N,K]^T (pure-torch reference).

    Act scale ``a_s`` is per 128 on K; weight scale ``b_s`` is per 32 on K (e8m0).
    Only reached when ``expert_dtype == "fp4"``. The activation side is handled fully;
    the FP4 weight is treated as already-unpacked float (the packed
    ``float4_e2m1fn_x2`` nibble layout of the real kernel is not unpacked here).
    """
    lead = a.shape[:-1]
    K = a.size(-1)
    N = b.size(0)
    a2 = a.reshape(-1, K)
    a_d = a2.float() * _scale_to_f32(a_s).reshape(a2.size(0), -1).repeat_interleave(128, dim=1)
    b_d = b.float() * _scale_to_f32(b_s).repeat_interleave(32, dim=1)           # [N, K]
    c = a_d @ b_d.T
    return c.reshape(*lead, N).to(torch.get_default_dtype())


# --------------------------------------------------------------------------- #
#  sparse attention  (the educational core)
# --------------------------------------------------------------------------- #
def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention by index gathering (pure-torch reference).

    For every query position we attend only to a small set of KV positions given by
    ``topk_idxs`` (a ``[b, s, topk]`` int tensor; ``-1`` means "no key / masked").

    Shapes
    ------
    q        : [b, s, h, d]
    kv       : [b, n, d]          (n = window + compressed positions)
    attn_sink: [h]                (a learned bias; behaves like one extra "always-on" key
                                   whose value is zero, so it only appears in the denom)
    topk_idxs: [b, s, topk]
    returns  : [b, s, h, d]

    Step through this function to watch: index gathering, scaled dot products, the
    causal/future mask via ``-1`` indices, and the attention-sink normalization.
    """
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)

    # --- gather the selected KV vectors ------------------------------------- #
    valid = topk_idxs != -1                                   # [b, s, topk]
    safe_idx = topk_idxs.clamp_min(0).long()                  # [b, s, topk]  (avoid OOB)
    kv_sel = kv.unsqueeze(1).expand(b, s, -1, d).gather(      # [b, s, topk, d]
        2, safe_idx.unsqueeze(-1).expand(b, s, topk, d))

    # --- scaled dot-product scores (fp32, like the kernel's accumulators) --- #
    qf, kf = q.float(), kv_sel.float()
    scores = torch.einsum("bshd,bskd->bshk", qf, kf) * softmax_scale   # [b, s, h, topk]
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))    # mask -1 positions

    # --- numerically stable softmax, with the sink only in the denominator - #
    max_k = scores.max(dim=-1).values                        # [b, s, h]
    exp_k = torch.exp(scores - max_k.unsqueeze(-1))          # invalid -> exp(-inf) = 0
    sum_k = exp_k.sum(dim=-1)                                # [b, s, h]
    sink_w = torch.exp(attn_sink.view(1, 1, h).float() - max_k)  # [b, s, h]  (zero value -> no num term)
    denom = sum_k + sink_w
    weights = exp_k / denom.unsqueeze(-1)                    # [b, s, h, topk]

    # --- weighted sum of values --------------------------------------------- #
    o = torch.einsum("bshk,bskd->bshd", weights, kf).to(q.dtype)  # [b, s, h, d]
    return o


# --------------------------------------------------------------------------- #
#  hyper-connection Sinkhorn split
# --------------------------------------------------------------------------- #
def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the HC "mixes" tensor into the pre / post / comb mixing tensors.

    ``mixes`` has shape ``[b, s, mix_hc]`` where ``mix_hc = (2 + hc) * hc``:
      * first  ``hc``        entries -> ``pre``  (residual-reduction weights before the sublayer)
      * next   ``hc``        entries -> ``post`` (expansion weights after the sublayer)
      * last   ``hc * hc``   entries -> ``comb`` (hc x hc mixing matrix), Sinkhorn-normalized
    ``hc_scale`` is [3] (one scale for pre / post / comb), ``hc_base`` is [mix_hc].

    Returns ``pre [b,s,hc]``, ``post [b,s,hc]``, ``comb [b,s,hc,hc]``.
    """
    hc = hc_mult
    b, s, _ = mixes.shape

    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps          # [b,s,hc]
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])  # [b,s,hc]

    comb = (mixes[..., 2 * hc:].unflatten(-1, (hc, hc))                # [b,s,hc,hc]
            * hc_scale[2] + hc_base[2 * hc:].view(1, 1, hc, hc))
    # Sinkhorn: alternate row / column normalization toward a doubly-stochastic matrix.
    comb = F.softmax(comb, dim=-1) + eps                               # row softmax (over last dim)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)               # column normalize
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)           # row normalize
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)           # column normalize
    return pre, post, comb


# --------------------------------------------------------------------------- #
#  Hadamard transform (fallback for rotate_activation when fast_hadamard_transform
#  is not installed)
# --------------------------------------------------------------------------- #
_HADAMARD_CACHE = {}


def _hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        H = torch.tensor([[1.0]], dtype=dtype, device=device)
        while H.size(0) < n:                       # Sylvester construction: [[H,H],[H,-H]]
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        assert H.size(0) == n, f"Hadamard dim {n} must be a power of two"
        _HADAMARD_CACHE[key] = H
    return _HADAMARD_CACHE[key]


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform along the last dimension (must be a power of two)."""
    H = _hadamard_matrix(x.size(-1), x.device, x.dtype)
    return torch.matmul(x, H) * scale
