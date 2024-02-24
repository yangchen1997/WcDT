#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: attention.py
@Author: YangChen
@Date: 2023/12/27
"""
import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, head_dim: int = 64, num_heads: int = 8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = head_dim * num_heads
        self.num_heads = num_heads
        # We assume d_v is the same as d_k here aka head_dim == embed_dim // num_heads
        self.head_dim = head_dim
        # Learnable parameters
        self.query_proj = nn.Linear(input_dim, self.embed_dim)
        self.key_proj = nn.Linear(input_dim, self.embed_dim)
        self.value_proj = nn.Linear(input_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, input_dim)

    def forward(self, input_feature, attn_mask=None):
        batch_size, obs_num = input_feature.shape[0], input_feature.shape[1]
        # Linear transformations
        q = self.query_proj(input_feature)
        k = self.key_proj(input_feature)
        v = self.value_proj(input_feature)

        # Split the embedding dimension into number of heads
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim)

        # Transpose to (batch_size, heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -float('inf'))

        # Apply attention score scaling and softmax
        attn = nn.functional.softmax(scores, dim=-1)

        # Multiply attention scores with value and sum to get the final output
        out = torch.matmul(attn, v)

        # Transpose back to (batch_size, seq_len, heads, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, obs_num, self.embed_dim)

        # Final linear transformation
        return self.out_proj(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_q_dim: int, input_kv_dim: int, head_dim: int = 64, num_heads: int = 8):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = head_dim * num_heads
        self.num_heads = num_heads
        # We assume d_v is the same as d_k here aka head_dim == embed_dim // num_heads
        self.head_dim = head_dim
        # Learnable parameters
        self.query_proj = nn.Linear(input_q_dim, self.embed_dim)
        self.key_proj = nn.Linear(input_kv_dim, self.embed_dim)
        self.value_proj = nn.Linear(input_kv_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, input_q_dim)

    def forward(self, input_q, input_kv, attn_mask=None):
        batch_size, obs_num = input_q.shape[0], input_q.shape[1]
        # Linear transformations
        q = self.query_proj(input_q)
        k = self.key_proj(input_kv)
        v = self.value_proj(input_kv)

        # Split the embedding dimension into number of heads
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim)

        # Transpose to (batch_size, heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -float('inf'))

        # Apply attention score scaling and softmax
        attn = nn.functional.softmax(scores, dim=-1)

        # Multiply attention scores with value and sum to get the final output
        out = torch.matmul(attn, v)

        # Transpose back to (batch_size, seq_len, heads, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, obs_num, self.embed_dim)

        # Final linear transformation
        return self.out_proj(out)