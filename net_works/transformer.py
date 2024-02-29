#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: transformer.py
@Author: YangChen
@Date: 2023/12/27
"""
from torch import nn

from net_works.attention import MultiHeadSelfAttention, MultiHeadCrossAttention


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerCrossAttention(nn.Module):

    def __init__(
            self, input_dim: int, conditional_dim: int,
            head_dim: int = 64, num_heads: int = 8, drop: float = 0.1,
    ):
        super(TransformerCrossAttention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(input_dim, head_dim, num_heads)
        self.cross_attention = MultiHeadCrossAttention(input_dim, conditional_dim, head_dim, num_heads)
        self.feed_forward = FeedForward(input_dim)
        self.drop_self_attention = nn.Dropout(drop)
        self.drop_cross_attention = nn.Dropout(drop)
        self.norm_self_attention = nn.LayerNorm(input_dim)
        self.norm_cross_attention = nn.LayerNorm(input_dim)
        self.norm_feed_forward = nn.LayerNorm(input_dim)

    def forward(self, input_x, input_conditional):
        # self attention
        norm_input_x = self.norm_self_attention(input_x)
        self_attention_output = self.self_attention(norm_input_x)
        res_output = input_x + self_attention_output
        res_output = self.drop_self_attention(res_output)
        # cross attention
        norm_input_x = self.norm_cross_attention(res_output)
        cross_attention_output = self.cross_attention(norm_input_x, input_conditional)
        res_output = res_output + cross_attention_output
        res_output = self.drop_cross_attention(res_output)
        # feed_forward
        norm_input_x = self.norm_feed_forward(res_output)
        feed_forward_output = self.feed_forward(norm_input_x)
        norm_input_x = norm_input_x + feed_forward_output
        return norm_input_x


class TransformerSelfAttention(nn.Module):

    def __init__(self, input_dim: int, head_dim: int = 64, num_heads: int = 8, drop: float = 0.1):
        super(TransformerSelfAttention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(input_dim, head_dim, num_heads)
        self.feed_forward = FeedForward(input_dim)
        self.drop_self_attention = nn.Dropout(drop)
        self.norm_self_attention = nn.LayerNorm(input_dim)
        self.norm_feed_forward = nn.LayerNorm(input_dim)

    def forward(self, input_x):
        # self attention
        norm_input_x = self.norm_self_attention(input_x)
        self_attention_output = self.self_attention(norm_input_x)
        res_output = input_x + self_attention_output
        res_output = self.drop_self_attention(res_output)
        # feed_forward
        norm_input_x = self.norm_feed_forward(res_output)
        feed_forward_output = self.feed_forward(norm_input_x)
        res_output = res_output + feed_forward_output
        return res_output
