#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_utils.py
@Author: YangChen
@Date: 2023/12/24
"""
import json
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

from common import ObjectType, LoadConfigResultDate
from utils.map_utils import MapUtil


class DataUtil:

    @classmethod
    def load_scenario_data(cls, scenario: scenario_pb2.Scenario) -> Dict[str, Any]:
        result_dict = dict()
        # 获取自车当前位置把所有轨迹和车道线转化成自车坐标系
        sdc_track_index = scenario.sdc_track_index
        current_time_index = scenario.current_time_index
        curr_state = scenario.tracks[scenario.sdc_track_index].states[current_time_index]
        ego_curr_x, ego_curr_y = curr_state.center_x, curr_state.center_y
        ego_curr_heading = curr_state.heading
        # 需要预测的障碍物id
        predicted_obs_ids = submission_specs.get_evaluation_sim_agent_ids(scenario)
        obs_tracks = cls.load_obs_tracks(scenario, ego_curr_x, ego_curr_y, ego_curr_heading)
        map_features = cls.load_map_features(scenario, ego_curr_x, ego_curr_y, ego_curr_heading)
        traffic_lights = cls.load_traffic_light(scenario, ego_curr_x, ego_curr_y, ego_curr_heading)
        if len(obs_tracks) <= 1:
            return result_dict
        result_dict['predicted_obs_ids'] = predicted_obs_ids
        result_dict['obs_tracks'] = obs_tracks
        result_dict['map_features'] = map_features
        result_dict['dynamic_states'] = traffic_lights
        result_dict['curr_loc'] = (ego_curr_x, ego_curr_y, ego_curr_heading, sdc_track_index)
        return result_dict

    @classmethod
    def load_obs_tracks(cls, scenario: scenario_pb2.Scenario,
                        ego_curr_x: float, ego_curr_y: float,
                        ego_curr_heading: float) -> List[Dict[str, Any]]:
        obs_tracks = list()
        # 一个障碍物的状态(id, 类型, 轨迹)
        for track in scenario.tracks:
            one_obs_track = dict()
            obs_id = track.id
            if hasattr(track, "object_type"):
                object_type = track.object_type
            else:
                object_type = 4
            obs_traj = list()
            for state in track.states:
                if not state.valid:
                    continue
                if 'height' not in one_obs_track and 'length' not in one_obs_track \
                        and 'width' not in one_obs_track:
                    one_obs_track['height'] = state.height
                    one_obs_track['length'] = state.length
                    one_obs_track['width'] = state.width
                center_x, center_y = MapUtil.global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                                             state.center_x, state.center_y)
                center_heading = MapUtil.theta_global_to_local(ego_curr_heading, state.heading)
                # 速度变化有两种方法:
                # 1. 在新坐标系做投影
                # 2. 计算新坐标系的航向角,用速度乘sin和cos
                curr_v = math.sqrt(math.pow(state.velocity_x, 2) +
                                   math.pow(state.velocity_y, 2))
                curr_v_heading = math.atan2(state.velocity_y, state.velocity_x)
                curr_v_heading = MapUtil.theta_global_to_local(ego_curr_heading, curr_v_heading)
                obs_traj.append(
                    (center_x, center_y, center_heading,
                     curr_v * math.cos(curr_v_heading), curr_v * math.sin(curr_v_heading))
                )
            one_obs_track['obs_id'] = obs_id
            one_obs_track['object_type'] = object_type
            one_obs_track['obs_traj'] = obs_traj
            # 轨迹丢失的障碍物不需要
            if len(obs_traj) == 91:
                obs_tracks.append(one_obs_track)
        return obs_tracks

    @staticmethod
    def load_map_features(scenario: scenario_pb2.Scenario,
                          ego_curr_x: float, ego_curr_y: float,
                          ego_curr_heading: float) -> List[Dict[str, Any]]:
        map_features = list()
        for map_feature in scenario.map_features:
            one_map_dict = dict()
            map_id = map_feature.id
            polygon_points = list()
            if hasattr(map_feature, "road_edge") and map_feature.road_edge.polyline:
                map_type = "road_edge"
                polygon_list = map_feature.road_edge.polyline
            elif hasattr(map_feature, "road_line") and map_feature.road_line.polyline:
                map_type = "road_line"
                polygon_list = map_feature.road_line.polyline
            else:
                continue
            for polygon_point in polygon_list:
                polygon_point_x, polygon_point_y = MapUtil.global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                                                           polygon_point.x, polygon_point.y)
                polygon_points.append((polygon_point_x, polygon_point_y))
            one_map_dict['map_id'] = map_id
            one_map_dict['map_type'] = map_type
            one_map_dict['polygon_points'] = polygon_points
            map_features.append(one_map_dict)
        return map_features

    @staticmethod
    def load_traffic_light(scenario: scenario_pb2.Scenario,
                           ego_curr_x: float, ego_curr_y: float,
                           ego_curr_heading: float) -> Dict[str, Any]:
        dynamic_states = dict()
        for dynamic_state in scenario.dynamic_map_states:
            for lane_state in dynamic_state.lane_states:
                lane_id = lane_state.lane
                lane_x, lane_y = MapUtil.global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                                         lane_state.stop_point.x,
                                                         lane_state.stop_point.y)
                if lane_id not in dynamic_states:
                    dynamic_states[lane_id] = list()
                    dynamic_states[lane_id].append((lane_x, lane_y))
                state = lane_state.state
                dynamic_states[lane_id].append(state)
        return dynamic_states

    @staticmethod
    def split_pkl_data(one_pkl_dict: Dict[str, Any], his_step: int) -> Tuple:
        map_points = [feature['polygon_points'] for feature in one_pkl_dict['map_features']]
        predicted_obs_ids = one_pkl_dict['predicted_obs_ids']
        # 初始化需要保存的信息
        index = 0
        traj_list = list()
        obs_feature_list = list()
        predicted_obs_index = list()
        # 障碍物信息
        for one_obs_info in one_pkl_dict['obs_tracks']:
            if len(one_obs_info['obs_traj']) < 91 or None in one_obs_info['obs_traj']:
                continue
            # 障碍物size和type
            obs_feature = list()
            obs_feature.append(one_obs_info['width'])
            obs_feature.append(one_obs_info['length'])
            type_onehot = [0] * len(ObjectType)
            type_onehot[one_obs_info['object_type']] = 1
            obs_feature += type_onehot
            obs_feature_list.append(obs_feature)
            # 记录predicted_obs的索引
            if one_obs_info['obs_id'] in predicted_obs_ids:
                predicted_obs_index.append(index)
            traj = np.array(one_obs_info['obs_traj'])
            traj_list.append(traj)
            index += 1
        if len(predicted_obs_index) < 1:
            return tuple()
        # 动态地图信息
        dynamic_states = list()
        dynamic_pos = list()
        for key, value in one_pkl_dict['dynamic_states'].items():
            dynamic_pos.append(value[0])
            dynamic_state = value[1:]
            dynamic_state = dynamic_state[:his_step]
            if len(dynamic_state) < his_step:
                dynamic_state = dynamic_state + ([0] * (his_step - len(dynamic_state)))
            dynamic_states.append(dynamic_state)
        traj_arr = np.stack(traj_list, axis=0)
        obs_feature_list = np.stack(obs_feature_list, axis=0)
        dynamic_states = np.array(dynamic_states)
        dynamic_pos = np.array(dynamic_pos)
        one_pkl_data = (predicted_obs_index, traj_arr,
                        obs_feature_list, dynamic_states,
                        dynamic_pos, map_points)
        return one_pkl_data

    @staticmethod
    def get_obs_feature(
            config_data: LoadConfigResultDate,
            other_obs_index: List[int],
            all_obs_his_traj: torch.Tensor,
            all_obs_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        his_step = config_data.train_model_config.his_step
        max_other_num = config_data.train_model_config.max_other_num
        # other obs历史轨迹
        if len(other_obs_index) > 0:
            other_obs_traj_index = (torch.Tensor(other_obs_index)
                                    .to(torch.long).view(-1, 1, 1)
                                    .repeat((1, his_step, 5)))
            other_his_traj = torch.gather(all_obs_his_traj, 0, other_obs_traj_index)
            other_obs_feature_index = (torch.Tensor(other_obs_index)
                                       .to(torch.long).view(-1, 1)
                                       .repeat((1, 7)))
            # other feature
            other_feature = torch.gather(all_obs_feature, 0, other_obs_feature_index)
            other_his_traj = other_his_traj[:max_other_num]
            other_feature = other_feature[:max_other_num]
            num_gap = max_other_num - other_his_traj.shape[0]
            if num_gap > 0:
                other_his_traj = torch.cat((other_his_traj,
                                            torch.zeros(size=(num_gap, his_step, 5),
                                                        dtype=torch.float32)), dim=0)
                other_feature = torch.cat((other_feature,
                                           torch.zeros(size=(num_gap, 7),
                                                       dtype=torch.float32)), dim=0)
            other_traj_mask = torch.Tensor([1.0] * (other_his_traj.shape[0] - num_gap) + [0.0] *
                                           num_gap)
        else:
            other_his_traj = torch.zeros(
                size=(max_other_num, his_step, 5),
                dtype=torch.float32
            )
            other_feature = torch.zeros(size=(max_other_num, 7), dtype=torch.float32)
            other_traj_mask = torch.zeros(size=[max_other_num], dtype=torch.float32)
        return other_his_traj, other_feature, other_traj_mask

    @staticmethod
    def get_pred_feature(
            config_data: LoadConfigResultDate,
            predicted_index: torch.Tensor,
            all_obs_his_traj: torch.Tensor,
            all_obs_future_traj: torch.Tensor,
            all_obs_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        his_step = config_data.train_model_config.his_step
        future_step = 91 - his_step
        max_pred_num = config_data.train_model_config.max_pred_num
        predicted_his_traj = torch.gather(
            all_obs_his_traj, 0,
            predicted_index.repeat((1, his_step, 5))
        )
        predicted_future_traj = torch.gather(
            all_obs_future_traj, 0,
            predicted_index.repeat((1, future_step, 5))
        )
        predicted_his_traj = predicted_his_traj[:max_pred_num]
        predicted_future_traj = predicted_future_traj[:max_pred_num]
        # predicted feature
        predicted_feature = torch.gather(
            all_obs_feature, 0,
            predicted_index.view(-1, 1).repeat((1, 7))
        )
        predicted_feature = predicted_feature[:max_pred_num]
        num_gap = max_pred_num - predicted_his_traj.shape[0]
        if num_gap > 0:
            predicted_his_traj = torch.cat((predicted_his_traj,
                                            torch.zeros(size=(num_gap, his_step, 5),
                                                        dtype=torch.float32)), dim=0)
            predicted_future_traj = torch.cat((predicted_future_traj,
                                               torch.zeros(size=(num_gap, future_step, 5),
                                                           dtype=torch.float32)), dim=0)
            predicted_feature = torch.cat((predicted_feature,
                                           torch.zeros(size=(num_gap, 7),
                                                       dtype=torch.float32)), dim=0)
        predicted_traj_mask = torch.Tensor([1.0] * (predicted_his_traj.shape[0] - num_gap) +
                                           [0.0] * num_gap)
        return predicted_future_traj, predicted_his_traj, predicted_feature, predicted_traj_mask

    @staticmethod
    def get_traffic_light(
            config_data: LoadConfigResultDate,
            traffic_light: torch.Tensor,
            traffic_light_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        his_step = config_data.train_model_config.his_step
        max_traffic_light = config_data.train_model_config.max_traffic_light
        traffic_light = traffic_light[:max_traffic_light]
        traffic_light_pos = traffic_light_pos[:max_traffic_light]
        num_gap = max_traffic_light - traffic_light.shape[0]
        if num_gap > 0:
            traffic_light = torch.cat((traffic_light,
                                       torch.zeros(size=(num_gap, his_step),
                                                   dtype=torch.float32)), dim=0)
            traffic_light_pos = torch.cat((traffic_light_pos,
                                           torch.zeros(size=(num_gap, 2),
                                                       dtype=torch.float32)), dim=0)
        traffic_mask = torch.Tensor([1.0] * (traffic_light_pos.shape[0] - num_gap) +
                                    [0.0] * num_gap)
        return traffic_light, traffic_light_pos, traffic_mask

    @staticmethod
    def get_lane_feature(config_data: LoadConfigResultDate, lane_feature: List) -> torch.Tensor:
        max_lane_num = config_data.train_model_config.max_lane_num
        max_point_num = config_data.train_model_config.max_point_num
        lane_list = list()
        for lane_index, lane in enumerate(lane_feature):
            if lane_index >= max_lane_num:
                break
            point_list = lane[:max_point_num]
            point_gap = max_point_num - len(point_list)
            # MAX_POINT_NUM, 2
            point_list = torch.Tensor(point_list)
            if point_gap > 0:
                point_list = torch.cat((point_list,
                                        torch.zeros(size=(point_gap, 2),
                                                    dtype=torch.float32)), dim=0)
            lane_list.append(point_list)
        lane_gap = max_lane_num - len(lane_list)
        lane_list = torch.stack(lane_list, dim=0)
        if lane_gap > 0:
            lane_list = torch.cat(
                (lane_list, torch.zeros(size=(lane_gap, max_point_num, 2),
                                        dtype=torch.float32)), dim=0
            )
        return lane_list

    @classmethod
    def transform_data_to_input(cls, scenario: scenario_pb2.Scenario,
                                config_data: LoadConfigResultDate) -> Dict[str, Any]:
        data_dict = cls.load_scenario_data(scenario)
        if len(data_dict) == 0:
            return dict()
        his_step = config_data.train_model_config.his_step
        pkl_data = DataUtil.split_pkl_data(data_dict, his_step)
        all_obs_index = set([i for i in range(pkl_data[1].shape[0])])
        predicted_index = torch.Tensor(pkl_data[0]).to(torch.long).view(-1, 1, 1)
        all_obs_traj = torch.Tensor(pkl_data[1])
        all_obs_feature = torch.Tensor(pkl_data[2])
        all_obs_his_traj = all_obs_traj[:, :his_step]
        all_obs_future_traj = all_obs_traj[:, his_step:]
        other_obs_index = list(all_obs_index - set(pkl_data[0]))
        # 获取障碍物信息
        other_his_traj, other_feature, other_traj_mask = cls.get_obs_feature(
            config_data, other_obs_index, all_obs_his_traj, all_obs_feature
        )
        # 需要预测的障碍物历史轨迹
        (predicted_future_traj, predicted_his_traj,
         predicted_feature, predicted_traj_mask) = cls.get_pred_feature(
            config_data, predicted_index, all_obs_his_traj,
            all_obs_future_traj, all_obs_feature
        )
        # 交通灯信息
        traffic_light = torch.Tensor(pkl_data[3])
        traffic_light_pos = torch.Tensor(pkl_data[4])
        traffic_light, traffic_light_pos, traffic_mask = cls.get_traffic_light(
            config_data, traffic_light, traffic_light_pos
        )
        # 车道线信息
        lane_list = cls.get_lane_feature(config_data, pkl_data[-1])
        map_json = json.dumps(pkl_data[-1])
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
            "map_json": map_json,
            "curr_loc": data_dict['curr_loc']
        }
        return result
