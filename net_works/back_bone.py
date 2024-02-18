#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: back_bone.py
@Author: YangChen
@Date: 2023/12/27
"""
from typing import Dict

import numpy as np
import torch
from torch import nn

from net_works.diffusion import GaussianDiffusion
from net_works.scene_encoder import SceneEncoder
from net_works.traj_decoder import TrajDecoder
from utils import MathUtil


class MultiModalLoss(nn.Module):
    def __init__(self):
        super(MultiModalLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction="none")
        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, traj, confidence, predicted_future_traj, predicted_traj_mask):
        batch = traj.shape[0]
        obs_num = traj.shape[1]
        multimodal = traj.shape[2]
        future_step = traj.shape[3]
        output_dim = traj.shape[4]
        predicted_future_extend = predicted_future_traj.unsqueeze(dim=-3)
        loss = self.huber_loss(traj, predicted_future_extend)
        loss = loss.view(batch, obs_num, multimodal, -1)
        loss = torch.mean(loss, dim=-1)
        min_loss, _ = torch.min(loss, dim=-1)
        min_loss_modal = torch.argmin(loss, dim=-1)
        min_loss_index = min_loss_modal.view(batch, obs_num, 1, 1, 1).repeat(
            (1, 1, 1, future_step, output_dim)
        )
        min_loss_traj = torch.gather(traj, -3, min_loss_index).squeeze()
        confidence = confidence.view(-1, multimodal)
        min_loss_modal = min_loss_modal.view(-1)
        confidence_loss = self.confidence_loss(confidence, min_loss_modal).view(batch, obs_num)
        traj_loss = torch.sum(min_loss * predicted_traj_mask) / (torch.sum(predicted_traj_mask) + 0.00001)
        confidence_loss = torch.sum(confidence_loss * predicted_traj_mask) / (torch.sum(predicted_traj_mask) + 0.00001)
        return traj_loss, confidence_loss, min_loss_traj


class BackBone(nn.Module):
    def __init__(self, betas: np.ndarray):
        super(BackBone, self).__init__()
        self.diffusion = GaussianDiffusion(betas=betas)
        self.scene_encoder = SceneEncoder()
        self.traj_decoder = TrajDecoder()
        self.multi_modal_loss = MultiModalLoss()

    def forward(self, data: Dict):
        # batch, other_obs(10), 40, 7
        predicted_feature = data['predicted_feature']
        # batch, other_obs(10), 40, 5
        other_his_pos = data['other_his_pos']
        other_his_traj_delt = data['other_his_traj_delt']
        other_feature = data['other_feature']
        other_traj_mask = data['other_traj_mask']
        # batch, pred_obs(15), 40, 5
        predicted_his_pos = data['predicted_his_pos']
        predicted_his_traj_delt = data['predicted_his_traj_delt']
        predicted_his_traj = data['predicted_his_traj']
        # batch, pred_obs(15), 50, 5
        predicted_future_traj = data['predicted_future_traj']
        predicted_traj_mask = data['predicted_traj_mask']
        # batch, tl_num(10), 2
        traffic_light = data['traffic_light']
        traffic_light_pos = data['traffic_light_pos']
        # batch, num_lane(32), num_point(128), 2
        lane_list = data['lane_list']
        # diffusion训练
        noise = torch.randn_like(predicted_his_traj_delt)
        diffusion_loss = self.diffusion(data)
        noise = self.diffusion.sample(noise, predicted_his_traj)
        # scene encoder
        scene_feature = self.scene_encoder(
            noise, lane_list,
            other_his_traj_delt, other_his_pos, other_feature,
            predicted_his_traj_delt, predicted_his_pos, predicted_feature,
            traffic_light, traffic_light_pos
        )
        # traj_decoder
        traj, confidence = self.traj_decoder(scene_feature)
        traj = MathUtil.post_process_output(traj, predicted_his_traj)
        traj_loss, confidence_loss, min_loss_traj = self.multi_modal_loss(traj, confidence, predicted_future_traj,
                                                                          predicted_traj_mask)
        return diffusion_loss, traj_loss, confidence_loss, min_loss_traj
