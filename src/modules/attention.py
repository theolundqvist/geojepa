from sympy.vector import Cross
from timm.layers import drop_path
from torch import functional as F
from timm.layers import DropPath

from src.modules.mlp import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Never got the CrossPredictor to work properly.
# Meaning that most of these are not used.
# -----------------------


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
    ):
        """
        Args:
            dim:        input dimension per token
            heads:  number of attention heads
            bias:   whether to use bias in linear layers for q, k, v
            attn_drop:  dropout rate for attention weights
            proj_drop:  dropout rate for the linear output projection
        """
        super().__init__()

        self.heads = heads
        self.dim = dim
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {heads}")
        self.head_dim = dim // heads

        # For self-attention: a single linear layer that produces Q, K, V all at once
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)

        self.attn_drop = attn_drop
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Self-attention forward pass.

        Args:
            x:               (B, L, C) input sequence
            key_mask (optional): (B, L) with True in positions to attend

        Returns:
            out: (B, L, C)
        """
        B, L, _ = x.shape

        # Project Q, K, V all in one go
        qkv = self.qkv(x)  # (B, L, 3*C)
        q, k, v = qkv.split(self.dim, dim=-1)
        # q, k, v each: (B, L, C)

        # Reshape for multi-head: (B, num_heads, seq_len, head_dim)
        q = q.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B, self.heads, L, L)

        # Note: scaled_dot_product_attention handles the scaling and softmax
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,  # or pass in a causal mask / 2D mask if needed
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )  # (B, num_heads, L, head_dim)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, L, self.dim)  # (B, L, C)

        # Output projection
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


def create_cross_attn_mask(q_token_pad_mask, kv_token_pad_mask):
    # -- for cross-attention, typically you at least want to mask out invalid context tokens:
    q_pad = q_token_pad_mask
    kv_pad = kv_token_pad_mask

    B, N_kv = kv_pad.shape
    _, N_q = q_pad.shape

    # (B, 1, N_kv) -> expand -> (B, N_q, N_kv)
    kv_mask = ~kv_pad.unsqueeze(1).expand(B, N_q, N_kv)
    # cross_mask = kv_mask.masked_fill(kv_mask, float('-inf'))

    # if you also want to mask target queries:
    # true if pay attention to this token
    q_mask = ~q_pad.unsqueeze(2).expand(B, N_q, N_kv)

    # cross_mask should be true if both q_mask and kv_mask are relevant
    cross_mask = kv_mask & q_mask
    # cross_mask = cross_mask.masked_fill(q_mask, float('-inf'))

    return create_self_attn_mask(q_token_pad_mask), cross_mask


def create_self_attn_mask(token_pad_mask):
    if token_pad_mask is None:
        return None
    pad = token_pad_mask
    B, T = pad.shape
    q_expanded = pad.unsqueeze(1).expand(B, T, T)
    # Create the self_mask: wherever q_expanded is True, put -inf
    # self_mask = q_expanded.masked_fill(q_expanded, float('-inf'))
    self_mask = ~q_expanded
    return self_mask


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        """
        Args:
            dim:        input dimension per token
            heads:  number of attention heads
            bias:   whether to use bias in linear layers for q, k, v
            attn_drop:  dropout rate for attention weights
            proj_drop:  dropout rate for the linear output projection
        """
        super().__init__()

        self.heads = heads
        self.dim = dim
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {heads}")
        self.head_dim = dim // heads

        # For cross-attention: separate projections for Q, and for K+V
        self.q_proj = nn.Linear(dim, dim, bias=bias)  # produces Q
        self.kv_proj = nn.Linear(dim, dim * 2, bias=bias)  # produces K, V

        self.attn_drop = attn_drop
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None):
        """
        Cross-attention forward pass.

        Args:
            q:               (B, Lq, C) queries
            kv:              (B, Lkv, C) key/value tokens
            mask (optional): (B, Lkv) with True in positions to attend

        Returns:
            out: (B, Lq, C)
        """
        B, Lq, _ = q.shape
        Lkv = kv.shape[1]

        # Project Q separately, and K+V together
        q = self.q_proj(q)  # (B, Lq, C)
        kv = self.kv_proj(kv)  # (B, Lkv, 2*C)
        k, v = kv.split(self.dim, dim=-1)
        # k, v each: (B, Lkv, C)

        # Reshape for multi-head
        q = q.reshape(B, Lq, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Lkv, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Lkv, self.heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B, self.heads, Lq, Lkv)

        # scaled_dot_product_attention call
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )  # (B, num_heads, Lq, head_dim)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)  # (B, Lq, C)

        # Output projection
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, bias=False, drop=0., attn_drop=0., path_drop=0., norm=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm(dim)
        self.self_attn = SelfAttention(
            dim, heads=heads, bias=bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm(dim)
        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x, mask: torch.Tensor = None):
        x = x + self.drop_path(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., bias=False, drop=0., attn_drop=0.,
                 path_drop=0., norm=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm(dim)
        self.self_attn = SelfAttention(
            dim, heads=heads, bias=bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2q = norm(dim)
        self.norm2kv = norm(dim)
        self.cross_attn = CrossAttention(
            dim, heads=heads, bias=bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, drop=drop)

    def forward(self, q, kv, self_mask: torch.Tensor = None, cross_mask: torch.Tensor = None):
        q = q + self.drop_path(self.self_attn(self.norm1(q), mask=self_mask))
        q = q + self.drop_path(self.cross_attn(self.norm2q(q), self.norm2kv(kv), mask=cross_mask))
        q = q + self.drop_path(self.mlp(self.norm3(q)))
        return q


class StackedAttention(nn.Module):
    def __init__(self, dim, heads, depth, mlp_ratio=4., bias=False, drop=0.0, attn_drop=0.0,
                 path_drop=0.0, norm=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionLayer(
                dim, heads=heads, mlp_ratio=mlp_ratio, bias=bias,
                drop=drop, attn_drop=attn_drop, path_drop=path_drop, norm=norm)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class StackedCrossAttention(nn.Module):
    def __init__(self, dim, heads, depth, mlp_ratio=4., bias=False, drop=0.0, attn_drop=0.0,
                 path_drop=0.0, norm=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionLayer(
                dim, heads=heads, mlp_ratio=mlp_ratio, bias=bias,
                drop=drop, attn_drop=attn_drop, path_drop=path_drop, norm=norm)
            for i in range(depth)])

    def forward(self, q, kv, self_mask=None, cross_mask=None):
        for block in self.blocks:
            q = block(q, kv, self_mask=self_mask, cross_mask=cross_mask)
        return q
