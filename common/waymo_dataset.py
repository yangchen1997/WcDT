#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: waymo_dataset.py
@Author: YangChen
@Date: 2023/12/25
"""

import json
import os
import pickle
from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset


class WaymoDataset(Dataset):

    def __init__(
            self, dataset_dir: str, his_step: int, max_pred_num: int,
            max_other_num: int, max_traffic_light: int, max_lane_num: int,
            max_point_num: int, max_data_size: int = -1
    ):
        super().__init__()
        self.__dataset_dir = dataset_dir
        self.__pkl_list = sorted(os.listdir(dataset_dir),
                                 key=lambda x: int(x[:-4].split('_')[-1]))
        if max_data_size != -1:
            self.__pkl_list = self.__pkl_list[:max_data_size]
        # 加载参数
        self.__max_pred_num = max_pred_num
        self.__max_other_num = max_other_num
        self.__max_traffic_light = max_traffic_light
        self.__max_lane_num = max_lane_num
        self.__max_point_num = max_point_num
        self.__his_step = his_step
        self.__future_step = 91 - his_step

    def __len__(self) -> int:
        return len(self.__pkl_list)

    def get_obs_feature(
            self, other_obs_index: List[int],
            all_obs_his_traj: torch.Tensor,
            all_obs_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # other obs历史轨迹
        if len(other_obs_index) > 0:
            other_obs_traj_index = (torch.Tensor(other_obs_index)
                                    .to(torch.long).view(-1, 1, 1)
                                    .repeat((1, self.__his_step, 5)))
            other_his_traj = torch.gather(all_obs_his_traj, 0, other_obs_traj_index)
            other_obs_feature_index = (torch.Tensor(other_obs_index)
                                       .to(torch.long).view(-1, 1)
                                       .repeat((1, 7)))
            # other feature
            other_feature = torch.gather(all_obs_feature, 0, other_obs_feature_index)
            other_his_traj = other_his_traj[:self.__max_other_num]
            other_feature = other_feature[:self.__max_other_num]
            num_gap = self.__max_other_num - other_his_traj.shape[0]
            if num_gap > 0:
                other_his_traj = torch.cat((other_his_traj,
                                            torch.zeros(size=(num_gap, self.__his_step, 5),
                                                        dtype=torch.float32)), dim=0)
                other_feature = torch.cat((other_feature,
                                           torch.zeros(size=(num_gap, 7),
                                                       dtype=torch.float32)), dim=0)
            other_traj_mask = torch.Tensor([1.0] * (other_his_traj.shape[0] - num_gap) + [0.0] *
                                           num_gap)
        else:
            other_his_traj = torch.zeros(
                size=(self.__max_other_num, self.__his_step, 5),
                dtype=torch.float32
            )
            other_feature = torch.zeros(size=(self.__max_other_num, 7), dtype=torch.float32)
            other_traj_mask = torch.zeros(size=[self.__max_other_num], dtype=torch.float32)
        return other_his_traj, other_feature, other_traj_mask

    def get_pred_feature(
            self, predicted_index: torch.Tensor,
            all_obs_his_traj: torch.Tensor,
            all_obs_future_traj: torch.Tensor,
            all_obs_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        predicted_his_traj = torch.gather(
            all_obs_his_traj, 0,
            predicted_index.repeat((1, self.__his_step, 5))
        )
        predicted_future_traj = torch.gather(
            all_obs_future_traj, 0,
            predicted_index.repeat((1, self.__future_step, 5))
        )
        predicted_his_traj = predicted_his_traj[:self.__max_pred_num]
        predicted_future_traj = predicted_future_traj[:self.__max_pred_num]
        # predicted feature
        predicted_feature = torch.gather(
            all_obs_feature, 0,
            predicted_index.view(-1, 1).repeat((1, 7))
        )
        predicted_feature = predicted_feature[:self.__max_pred_num]
        num_gap = self.__max_pred_num - predicted_his_traj.shape[0]
        if num_gap > 0:
            predicted_his_traj = torch.cat((predicted_his_traj,
                                            torch.zeros(size=(num_gap, self.__his_step, 5),
                                                        dtype=torch.float32)), dim=0)
            predicted_future_traj = torch.cat((predicted_future_traj,
                                               torch.zeros(size=(num_gap, self.__future_step, 5),
                                                           dtype=torch.float32)), dim=0)
            predicted_feature = torch.cat((predicted_feature,
                                           torch.zeros(size=(num_gap, 7),
                                                       dtype=torch.float32)), dim=0)
        predicted_traj_mask = torch.Tensor([1.0] * (predicted_his_traj.shape[0] - num_gap) +
                                           [0.0] * num_gap)
        return predicted_future_traj, predicted_his_traj, predicted_feature, predicted_traj_mask

    def get_traffic_light(
            self, traffic_light: torch.Tensor,
            traffic_light_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traffic_light = traffic_light[:self.__max_traffic_light]
        traffic_light_pos = traffic_light_pos[:self.__max_traffic_light]
        num_gap = self.__max_traffic_light - traffic_light.shape[0]
        if num_gap > 0:
            traffic_light = torch.cat((traffic_light,
                                       torch.zeros(size=(num_gap, self.__his_step),
                                                   dtype=torch.float32)), dim=0)
            traffic_light_pos = torch.cat((traffic_light_pos,
                                           torch.zeros(size=(num_gap, 2),
                                                       dtype=torch.float32)), dim=0)
        traffic_mask = torch.Tensor([1.0] * (traffic_light_pos.shape[0] - num_gap) +
                                    [0.0] * num_gap)
        return traffic_light, traffic_light_pos, traffic_mask

    def get_lane_feature(self, lane_feature: List) -> torch.Tensor:
        lane_list = list()
        for lane_index, lane in enumerate(lane_feature):
            if lane_index >= self.__max_lane_num:
                break
            point_list = lane[:self.__max_point_num]
            point_gap = self.__max_point_num - len(point_list)
            # MAX_POINT_NUM, 2
            point_list = torch.Tensor(point_list)
            if point_gap > 0:
                point_list = torch.cat((point_list,
                                        torch.zeros(size=(point_gap, 2),
                                                    dtype=torch.float32)), dim=0)
            lane_list.append(point_list)
        lane_gap = self.__max_lane_num - len(lane_list)
        lane_list = torch.stack(lane_list, dim=0)
        if lane_gap > 0:
            lane_list = torch.cat(
                (lane_list, torch.zeros(size=(lane_gap, self.__max_point_num, 2),
                                        dtype=torch.float32)), dim=0
            )
        return lane_list

    def __getitem__(self, idx) -> Dict:
        with open(os.path.join(self.__dataset_dir, self.__pkl_list[idx]), "rb") as f:
            result = pickle.load(f)
        all_obs_index = set([i for i in range(result[1].shape[0])])
        predicted_index = torch.Tensor(result[0]).to(torch.long).view(-1, 1, 1)
        all_obs_traj = torch.Tensor(result[1])
        all_obs_feature = torch.Tensor(result[2])
        all_obs_his_traj = all_obs_traj[:, :self.__his_step]
        all_obs_future_traj = all_obs_traj[:, self.__his_step:]
        other_obs_index = list(all_obs_index - set(result[0]))
        # 获取障碍物信息
        other_his_traj, other_feature, other_traj_mask = self.get_obs_feature(
            other_obs_index, all_obs_his_traj, all_obs_feature
        )
        # 需要预测的障碍物历史轨迹
        (predicted_future_traj, predicted_his_traj,
         predicted_feature, predicted_traj_mask) = self.get_pred_feature(
            predicted_index, all_obs_his_traj,
            all_obs_future_traj, all_obs_feature
        )
        # 交通灯信息
        traffic_light = torch.Tensor(result[3])
        traffic_light_pos = torch.Tensor(result[4])
        traffic_light, traffic_light_pos, traffic_mask = self.get_traffic_light(
            traffic_light, traffic_light_pos
        )
        # 车道线信息
        lane_list = self.get_lane_feature(result[-1])
        map_json = json.dumps(result[-1])
        other_his_traj_delt = other_his_traj[:, 1:] - other_his_traj[:, :-1]
        other_his_pos = other_his_traj[:, -1, :2]
        predicted_his_traj_delt = predicted_his_traj[:, 1:] - predicted_his_traj[:, :-1]
        predicted_his_pos = predicted_his_traj[:, -1, :2]
        result = {
            "other_his_traj": other_his_traj,
            "other_feature": other_feature,
            "other_traj_mask": other_traj_mask,
            "other_his_traj_delt": other_his_traj_delt,
            "other_his_pos": other_his_pos,
            "predicted_future_traj": predicted_future_traj,
            "predicted_his_traj": predicted_his_traj,
            "predicted_traj_mask": predicted_traj_mask,
            "predicted_feature": predicted_feature,
            "predicted_his_traj_delt": predicted_his_traj_delt,
            "predicted_his_pos": predicted_his_pos,
            "traffic_light": traffic_light,
            "traffic_light_pos": traffic_light_pos,
            "traffic_mask": traffic_mask,
            "lane_list": lane_list,
            "map_json": map_json
        }
        return result
