#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: scene_encoder.py
@Author: YangChen
@Date: 2023/12/27
"""
import torch
from torch import nn

from net_works.transformer import TransformerCrossAttention, TransformerSelfAttention


class OtherFeatureFormer(nn.Module):
    def __init__(
            self, block_num: int, input_dim: int, conditional_dim: int,
            head_dim: int = 64, num_heads: int = 8
    ):
        super(OtherFeatureFormer, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(block_num):
            self.blocks.append(TransformerCrossAttention(input_dim, conditional_dim, head_dim, num_heads))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input_x, other_feature):
        for block in self.blocks:
            input_x = block(input_x, other_feature)
        return self.norm(input_x)


class SelfFeatureFormer(nn.Module):
    def __init__(self, block_num: int, input_dim: int, head_dim: int = 64, num_heads: int = 8):
        super(SelfFeatureFormer, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(block_num):
            self.blocks.append(TransformerSelfAttention(input_dim, head_dim, num_heads))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input_x):
        for block in self.blocks:
            input_x = block(input_x)
        return self.norm(input_x)


class SceneEncoder(nn.Module):
    def __init__(
            self, dim: int = 256, embedding_dim: int = 32,
            his_step: int = 11, other_agent_depth: int = 4,
            map_feature_depth: int = 4, traffic_light_depth: int = 2,
            self_attention_depth: int = 4
    ):
        super(SceneEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.pos_embedding = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        self.feature_embedding = nn.Sequential(
            nn.Linear(7, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        linear_input_dim = (his_step - 1) * 5 + embedding_dim + embedding_dim
        self.linear_input = nn.Sequential(
            nn.Linear(linear_input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.other_agent_former = OtherFeatureFormer(block_num=other_agent_depth, input_dim=dim, conditional_dim=114)
        self.map_former = OtherFeatureFormer(block_num=map_feature_depth, input_dim=dim, conditional_dim=embedding_dim)
        self.traffic_light_former = OtherFeatureFormer(block_num=traffic_light_depth, input_dim=dim,
                                                       conditional_dim=43)
        self.fusion_block = SelfFeatureFormer(block_num=self_attention_depth, input_dim=dim, num_heads=16)

    def forward(
            self, noise, lane_list,
            other_his_traj_delt, other_his_pos, other_feature,
            predicted_his_traj_delt, predicted_his_pos, predicted_feature,
            traffic_light, traffic_light_pos
    ):
        batch_size, obs_num = noise.shape[0], noise.shape[1]
        # batch, obs_num(8), his_step, 3
        x = predicted_his_traj_delt + (noise * 0.001)
        x = torch.flatten(x, start_dim=2)
        other_his_traj_delt = torch.flatten(other_his_traj_delt, start_dim=2)
        # 对各个位置进行位置编码
        lane_list = self.pos_embedding(lane_list)
        lane_list = lane_list.view(batch_size, -1, self.embedding_dim)
        traffic_light_pos = self.pos_embedding(traffic_light_pos)
        other_his_pos = self.pos_embedding(other_his_pos)
        predicted_his_pos = self.pos_embedding(predicted_his_pos)
        # 对属性进行编码
        other_feature = self.feature_embedding(other_feature)
        predicted_feature = self.feature_embedding(predicted_feature)
        # 组合输入信息
        x = torch.cat((x, predicted_his_pos, predicted_feature), dim=-1)
        # batch, obs_num(15), 256
        x = self.linear_input(x)
        # other agent former
        other_obs_feature = torch.cat((other_his_traj_delt, other_his_pos, other_feature), dim=-1)
        x = self.other_agent_former(x, other_obs_feature)
        # map_point_transformer
        x = self.map_former(x, lane_list)
        # traffic_light_transformer
        traffic_light = torch.cat((traffic_light, traffic_light_pos), dim=-1)
        x = self.traffic_light_former(x, traffic_light)
        x = self.fusion_block(x)
        return x
