"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom

    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# FlashAttention Kernel — Online Softmax (Streaming)
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr,           # (batch_heads, seq_q, head_dim)
    k_ptr,           # (batch_heads, seq_k, head_dim)
    v_ptr,           # (batch_heads, seq_k, head_dim)
    output_ptr,      # (batch_heads, seq_q, head_dim)
    scale,
    seq_k,           # actual seq_k length
    head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_D: tl.constexpr,   # head_dim (power of 2)
    TILE_KV: tl.constexpr,   # tile size for K/V streaming
):
    """
    FlashAttention with online softmax — streams through K/V tiles
    maintaining running max, sum-of-exp, and weighted output.
    Never materializes the full N×N attention matrix.

    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # Load query vector for this position
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators
    m_prev = tl.full([1], value=-1e30, dtype=tl.float32)  # running max
    l_prev = tl.full([1], value=0.0, dtype=tl.float32)    # running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)            # weighted output accumulator

    # Stream through K/V in tiles
    for kv_start in range(0, seq_k, TILE_KV):
        offs_kv = kv_start + tl.arange(0, TILE_KV)
        kv_mask = offs_kv < seq_k

        # Load K tile: (TILE_KV, head_dim)
        k_tile = tl.load(
            k_ptr + pid_bh * stride_k0
            + offs_kv[:, None] * stride_k1
            + offs_d[None, :] * stride_k2,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Compute scores: Q @ K^T -> (TILE_KV,)
        scores = tl.sum(k_tile * q[None, :], axis=1) * scale
        # Mask padded positions
        scores = tl.where(kv_mask, scores, -1e30)

        # Online softmax: update running max
        m_cur = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_cur)

        # Correction factor for previous accumulator
        exp_correction = tl.exp(m_prev - m_new)

        # New exp(scores - m_new)
        p_tile = tl.exp(scores - m_new)

        # Update running sum
        l_new = l_prev * exp_correction + tl.sum(p_tile, axis=0)

        # Load V tile: (TILE_KV, head_dim)
        v_tile = tl.load(
            v_ptr + pid_bh * stride_v0
            + offs_kv[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Update accumulator: rescale old + add new
        acc = acc * exp_correction + tl.sum(p_tile[:, None] * v_tile, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Final normalization
    acc = acc / l_prev

    # Store output
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        acc,
        mask=d_mask,
    )


# Flash attention dispatch constants
MAX_FLASH_SEQ_K = 1024  # Use flash for seq_k <= this, cuBLAS/torch for larger
FLASH_TILE_KV = 64      # Tile size for K/V streaming in flash kernel


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    Three dispatch paths:
      1. Flash path: fused online-softmax kernel (no intermediate scores tensor)
      2. 3-kernel path: scores + softmax + output (for small seq_k with masks)
      3. Torch fallback: batched matmul (for large seq_k)
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    # --- Flash attention path ---
    can_use_flash = (
        q.is_cuda
        and head_dim_padded <= 256
        and seq_k <= MAX_FLASH_SEQ_K
        and attention_mask is None
        and (not is_causal or seq_q == 1)  # Flash for decode (seq_q=1), not causal prefill
    )

    if can_use_flash:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()

        output = torch.empty(
            (batch * num_heads, seq_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        flash_attention_kernel[grid](
            q_flat, k_flat, v_flat, output,
            float(scale),
            seq_k,
            head_dim,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_D=head_dim_padded,
            TILE_KV=FLASH_TILE_KV,
        )

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # --- 3-kernel Triton path ---
    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(
                    batch * num_heads, seq_q, seq_k
                )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # Torch/cuBLAS fallback path — use PyTorch SDPA when available (encoder prefill)
    if (
        not is_causal
        and attention_mask is None
        and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    ):
        # PyTorch 2.x SDPA: fused kernel, no materialized attention matrix
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale
        )

    # Manual fallback for causal/masked cases
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.matmul(attn_weights, v_f)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
