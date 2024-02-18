#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: traj_decoder.py
@Author: YangChen
@Date: 2023/12/27
"""
import torch
from torch import nn


class TrajDecoder(nn.Module):
    def __init__(self, multimodal: int = 10, dim: int = 256, future_step: int = 80):
        super(TrajDecoder, self).__init__()
        self.multimodal = multimodal
        self.future_step = future_step
        self.one_modal = (future_step * 3 + 1)
        output_dim = multimodal * self.one_modal
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_dim)
        )

    def forward(self, input_x):
        batch_size, obs_num = input_x.shape[0], input_x.shape[1]
        decoder_output = self.decoder(input_x)
        decoder_output = decoder_output.view(batch_size, obs_num, self.multimodal, self.one_modal)
        traj = decoder_output[:, :, :, :-1].view(batch_size, obs_num, self.multimodal, self.future_step, 3)
        confidence = decoder_output[:, :, :, -1]
        return traj, confidence
