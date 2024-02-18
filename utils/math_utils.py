#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: math_utils.py
@Author: YangChen
@Date: 2023/12/20
"""
import numpy as np
import torch


class MathUtil:

    @staticmethod
    def generate_linear_schedule(time_steps: int, low: float = 1e-5, high: float = 2e-5):
        return np.linspace(low, high, time_steps)

    @staticmethod
    def step_cos(time_step: int, time_steps: int, s: float):
        return (np.cos((time_step / time_steps + s) / (1 + s) * np.pi / 2)) ** 2

    # @classmethod
    # def generate_cosine_schedule(cls, time_steps: int, s: float = 0.008):
    #     alphas = []
    #     f0 = cls.step_cos(0, time_steps, s)
    #     for t in range(time_steps + 1):
    #         alphas.append(cls.step_cos(t, time_steps, s) / f0)
    #     betas = []
    #     for t in range(1, time_steps + 1):
    #         betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    #     return np.array(betas)

    @classmethod
    def generate_cosine_schedule(cls, time_steps: int, s: float = 0.008):
        steps = time_steps + 1
        x = torch.linspace(0, time_steps, steps)
        alphas_cum_prod = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cum_prod = alphas_cum_prod / alphas_cum_prod[0]
        betas = 1 - (alphas_cum_prod[1:] / alphas_cum_prod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).detach().numpy()

    @staticmethod
    def post_process_output(generate_traj: torch.Tensor, predicted_his_traj: torch.Tensor) -> torch.Tensor:
        delt_t = 0.1
        batch_size = generate_traj.shape[0]
        num_obs = generate_traj.shape[1]
        vx = generate_traj[:, :, :, :, 0] / delt_t
        vy = generate_traj[:, :, :, :, 1] / delt_t
        start_x = predicted_his_traj[:, :, -1, 0].view(batch_size, num_obs, 1, 1)
        start_y = predicted_his_traj[:, :, -1, 1].view(batch_size, num_obs, 1, 1)
        start_heading = predicted_his_traj[:, :, -1, 2].view(batch_size, num_obs, 1, 1)
        x = torch.cumsum(generate_traj[:, :, :, :, 0], dim=-1) + start_x
        y = torch.cumsum(generate_traj[:, :, :, :, 1], dim=-1) + start_y
        heading = torch.cumsum(generate_traj[:, :, :, :, 2], dim=-1) + start_heading
        output = torch.stack((x, y, heading, vx, vy), dim=-1)
        return output
