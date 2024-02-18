import json
import math
import os
import tarfile
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import tensorflow as tf
import torch
import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from nets import (GaussianDiffusion, generate_linear_schedule, SimpleViT, TrajDecorder, MapEncoder)

SHIFT_H = 125
SHIFT_W = 125
PIX_SIZE = 0.8
MAX_LANE_NUM = 50
MAX_POINT_NUM = 250
HIS_STEP = 11
FUTRUE_STEP = 80
MAX_PRED_NUM = 8
MAX_OTHER_NUM = 6
MAX_TRAFFIC_LIGHT_NUM = 8
MAX_LANE_NUM = 32
MAX_POINT_NUM = 128

transform_arr = np.array([SHIFT_W, SHIFT_H])
pix_size = PIX_SIZE
object_type = {
    'TYPE_UNSET': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_CYCLIST': 3,
    'TYPE_OTHER': 4
}

ObjectType = {0: "TYPE_UNSET", 1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST", 4: "TYPE_OTHER"}
MapState = {0: "LANE_STATE_UNKNOWN", 1: "LANE_STATE_ARROW_STOP", 2: "LANE_STATE_ARROW_CAUTION",
            3: "LANE_STATE_ARROW_GO", 4: "LANE_STATE_STOP", 5: "LANE_STATE_CAUTION",
            6: "LANE_STATE_GO", 7: "LANE_STATE_FLASHING_STOP", 8: "LANE_STATE_FLASHING_CAUTION"}

VALID_PATH = r"/mnt/share_disk/yangchen/Scene-Diffusion/valid_set"
TEST_PATH = r"/mnt/share_disk/waymo_dataset/waymodata_1_1/testing"
MODEL_PATH = r"/mnt/share_disk/yangchen/Scene-Diffusion/diffusion_model_last_epoch_weights.pth"


def global_to_local(curr_x: float, curr_y: float, curr_heading: float,
                    point_x: float, point_y: float) -> Tuple[float, float]:
    delta_x = point_x - curr_x
    delta_y = point_y - curr_y
    return delta_x * math.cos(curr_heading) + delta_y * math.sin(curr_heading), \
           delta_y * math.cos(curr_heading) - delta_x * math.sin(curr_heading)


def theta_global_to_local(curr_heading: float, heading: float) -> float:
    """
    将世界坐标系下的角度转成自车坐标系下的角度
    Args:
        curr_heading: 自车在世界系的转角
        heading: 要转换的heading
    Returns: 自车坐标系下的heading
    """
    return normalize_angle(-curr_heading + heading)


def normalize_angle(angle: float) -> float:
    """
    归一化弧度值, 使其范围在[-pi, pi]之间
    Args:
        angle: 输入的弧度
    Returns: 归一化之后的弧度
    """
    angle = (angle + math.pi) % (2 * math.pi)
    if angle < .0:
        angle += 2 * math.pi
    return angle - math.pi


def draw_one_rect(traj_list: List, image: np.ndarray,
                  box_points: np.ndarray, color: int) -> Tuple[np.ndarray, np.ndarray]:
    one_step_color = color / 11
    image_list = list()
    for i in range(0, HIS_STEP):
        yaw = traj_list[i][2]
        rotate_matrix = np.array([np.cos(yaw), -np.sin(yaw),
                                  np.sin(yaw), np.cos(yaw)]).reshape(2, 2)

        # 坐标变换
        box_points_temp = box_points @ -rotate_matrix
        coord = np.array([traj_list[i][0], traj_list[i][1]])
        box_points_temp = box_points_temp + coord
        box_points_temp = box_points_temp / pix_size + transform_arr
        box_points_temp = np.round(box_points_temp).astype(int).reshape(1, -1, 2)
        color_temp = int(one_step_color * (i + 1))
        temp_image = cv2.fillPoly(
            np.zeros_like(image),
            box_points_temp,
            color=[color_temp],
            shift=0,
        )
        image_list.append(temp_image)
    image_list.append(image)
    image = np.concatenate(image_list, axis=2)
    image = np.expand_dims(np.max(image, axis=2), axis=2)
    return image, np.array(traj_list[:HIS_STEP])


def gene_model_input_step_three(one_pkl_data: Tuple, predicted_obs_id: List) -> List:
    result = list()
    if len(one_pkl_data[1]) != len(predicted_obs_id):
        raise ValueError("data preprocess predicted_obs_id size != predicted_obs_id size")
    map_image = torch.Tensor(one_pkl_data[0])
    all_obs_index = set([i for i in range(one_pkl_data[2].shape[0])])
    predicted_index = torch.Tensor(one_pkl_data[1]).to(torch.long).view(-1, 1, 1)
    all_obs_traj = torch.Tensor(one_pkl_data[2])
    all_obs_feature = torch.Tensor(one_pkl_data[3])
    all_obs_his_traj = all_obs_traj[:, :HIS_STEP]
    other_obs_index = list(all_obs_index - set(one_pkl_data[1]))
    if len(other_obs_index) > 0:
        # other obs历史轨迹
        other_obs_traj_index = torch.Tensor(other_obs_index).to(torch.long).view(-1, 1, 1).repeat((1, HIS_STEP, 5))
        other_his_traj = torch.gather(all_obs_his_traj, 0, other_obs_traj_index)
        other_obs_feature_index = torch.Tensor(other_obs_index).to(torch.long).view(-1, 1).repeat((1, 7))
        # other feature
        other_feature = torch.gather(all_obs_feature, 0, other_obs_feature_index)
        other_his_traj = other_his_traj[:MAX_OTHER_NUM]
        other_feature = other_feature[:MAX_OTHER_NUM]
        num_gap = MAX_OTHER_NUM - other_his_traj.shape[0]
        if num_gap > 0:
            other_his_traj = torch.cat((other_his_traj,
                                        torch.zeros(size=(num_gap, HIS_STEP, 5),
                                                    dtype=torch.float32)), dim=0)
            other_feature = torch.cat((other_feature,
                                       torch.zeros(size=(num_gap, 7),
                                                   dtype=torch.float32)), dim=0)
        other_traj_mask = torch.Tensor([1.0] * (other_his_traj.shape[0] - num_gap) + [0.0] * num_gap)
    else:
        other_his_traj = torch.zeros(size=(MAX_OTHER_NUM, HIS_STEP, 5), dtype=torch.float32)
        other_feature = torch.zeros(size=(MAX_OTHER_NUM, 7), dtype=torch.float32)
        other_traj_mask = torch.zeros(size=[MAX_OTHER_NUM], dtype=torch.float32)
    # 交通灯信息
    traffic_light = torch.Tensor(one_pkl_data[4])
    traffic_light_pos = torch.Tensor(one_pkl_data[5])
    traffic_light = traffic_light[:MAX_TRAFFIC_LIGHT_NUM]
    traffic_light_pos = traffic_light_pos[:MAX_TRAFFIC_LIGHT_NUM]
    num_gap = MAX_TRAFFIC_LIGHT_NUM - traffic_light.shape[0]
    if num_gap > 0:
        traffic_light = torch.cat((traffic_light,
                                   torch.zeros(size=(num_gap, 91),
                                               dtype=torch.float32)), dim=0)
        traffic_light_pos = torch.cat((traffic_light_pos,
                                       torch.zeros(size=(num_gap, 2),
                                                   dtype=torch.float32)), dim=0)
    traffic_mask = torch.Tensor([1.0] * (traffic_light_pos.shape[0] - num_gap) + [0.0] * num_gap)
    # 车道线信息
    lane_list = list()
    for lane_index, lane in enumerate(one_pkl_data[6]):
        if lane_index >= MAX_LANE_NUM:
            break
        point_list = lane[:MAX_POINT_NUM]
        point_gap = MAX_POINT_NUM - len(point_list)
        # MAX_POINT_NUM, 2
        point_list = torch.Tensor(point_list)
        if point_gap > 0:
            point_list = torch.cat((point_list,
                                    torch.zeros(size=(point_gap, 2),
                                                dtype=torch.float32)), dim=0)
        lane_list.append(point_list)
    lane_gap = MAX_LANE_NUM - len(lane_list)
    if len(lane_list) > 0:
        lane_list = torch.stack(lane_list, dim=0)
        if lane_gap > 0:
            lane_list = torch.cat((lane_list,
                                torch.zeros(size=(lane_gap, MAX_POINT_NUM, 2),
                                            dtype=torch.float32)), dim=0)
    else:
        lane_list = torch.zeros(size=(MAX_LANE_NUM, MAX_POINT_NUM, 2), dtype=torch.float32)
    map_json = json.dumps(one_pkl_data[6])
    other_his_traj_delt = other_his_traj[:, 1:, :2] - other_his_traj[:, :-1, :2]
    other_his_traj_delt = torch.cat((other_his_traj_delt, other_his_traj[:, 1:, 2:]), dim=-1)
    other_his_pos = other_his_traj[:, -1, :2]
    # 如果需要预测的障碍物大于8个，需要分批次推理
    predicted_obs_batch_num = (len(one_pkl_data[1]) // MAX_PRED_NUM) + 1
    predicted_his_traj = torch.gather(all_obs_his_traj, 0, predicted_index.repeat((1, HIS_STEP, 5)))
    predicted_feature = torch.gather(all_obs_feature, 0, predicted_index.view(-1, 1).repeat((1, 7)))
    for i in range(predicted_obs_batch_num):
        start_index = i * MAX_PRED_NUM
        end_index = start_index + MAX_PRED_NUM - 1
        # 需要预测的障碍物历史轨迹
        predicted_his_traj_batch = predicted_his_traj[start_index:end_index]
        # predicted feature
        predicted_feature_batch = predicted_feature[start_index:end_index]
        # mask
        num_gap = MAX_PRED_NUM - predicted_his_traj_batch.shape[0]
        predicted_traj_mask = torch.Tensor([1.0] * (predicted_his_traj_batch.shape[0]) + [0.0] * num_gap)
        if num_gap > 0:
            predicted_his_traj_batch = torch.cat((predicted_his_traj_batch,
                                                  torch.zeros(size=(num_gap, HIS_STEP, 5),
                                                              dtype=torch.float32)), dim=0)
            predicted_feature_batch = torch.cat((predicted_feature_batch,
                                                 torch.zeros(size=(num_gap, 7),
                                                             dtype=torch.float32)), dim=0)
        predicted_his_traj_delt = predicted_his_traj_batch[:, 1:, :2] - predicted_his_traj_batch[:, :-1, :2]
        predicted_his_traj_delt = torch.cat((predicted_his_traj_delt, predicted_his_traj_batch[:, 1:, 2:]), dim=-1)
        predicted_his_pos = predicted_his_traj_batch[:, -1, :2]
        result_dict = {
            "map_image": map_image,
            "other_his_traj": other_his_traj,
            "other_feature": other_feature,
            "other_traj_mask": other_traj_mask,
            "other_his_traj_delt": other_his_traj_delt,
            "other_his_pos": other_his_pos,
            "predicted_traj_mask": predicted_traj_mask,
            "predicted_his_traj": predicted_his_traj_batch,
            "predicted_feature": predicted_feature_batch,
            "predicted_his_traj_delt": predicted_his_traj_delt,
            "predicted_his_pos": predicted_his_pos,
            "traffic_light": traffic_light,
            "traffic_light_pos": traffic_light_pos,
            "traffic_mask": traffic_mask,
            "lane_list": lane_list,
            "map_json": map_json
        }
        for key, value in result_dict.items():
            if not isinstance(value, str):
                result_dict[key] = value.unsqueeze(dim=0)

        result.append(result_dict)
    return result


def gene_model_input_step_two(data_dict: Dict[str, Any]) -> Tuple:
    # 把车道线光栅化为图片
    image_roadmap = np.zeros((256, 256, 1), dtype=np.uint8)
    image_vru_traj = np.zeros((256, 256, 1), dtype=np.uint8)
    image_car_traj = np.zeros((256, 256, 1), dtype=np.uint8)
    for road_info in data_dict['map_features']:
        if len(road_info['polygon_points']) <= 0:
            continue
        lane_points = np.array(road_info['polygon_points'])
        lane_points = lane_points / pix_size + transform_arr
        lane_points = np.round(lane_points).astype(int)
        image_roadmap = cv2.polylines(
            image_roadmap,
            [lane_points],
            False,
            [255],
            shift=0,
        )
    # 障碍物轨迹,光栅化图片或做成vector
    traj_list = list()
    obs_feature_list = list()
    valid_index = list()
    index = 0
    predicted_obs_ids = data_dict['predicted_obs_ids']
    predicted_obs_index = list()
    predicted_obs_real_index = list()
    for obs_index, one_obs_info in enumerate(data_dict['obs_tracks']):
        if len(one_obs_info['obs_traj']) < 11 or None in one_obs_info['obs_traj']:
            continue
        obs_feature = list()
        obs_feature.append(one_obs_info['width'])
        obs_feature.append(one_obs_info['length'])
        type_onehot = [0] * len(object_type)
        type_onehot[object_type[one_obs_info['object_type']]] = 1
        obs_feature += type_onehot
        obs_feature_list.append(obs_feature)
        if one_obs_info['obs_id'] in predicted_obs_ids:
            predicted_obs_index.append(index)
            predicted_obs_real_index.append(one_obs_info['obs_id'])
        obs_half_width = 0.5 * one_obs_info['width']
        obs_half_length = 0.5 * one_obs_info['length']
        box_points = np.array([-obs_half_length, -obs_half_width,
                               obs_half_length, -obs_half_width,
                               obs_half_length, obs_half_width,
                               -obs_half_length, obs_half_width]).reshape(4, 2).astype(np.float32)

        if one_obs_info['object_type'] in ('TYPE_PEDESTRIAN', 'TYPE_CYCLIST'):
            image_vru_traj, traj = draw_one_rect(one_obs_info['obs_traj'],
                                                 image_vru_traj,
                                                 box_points, 255)

        elif one_obs_info['object_type'] == 'TYPE_VEHICLE':
            valid_index.append(index)
            image_car_traj, traj = draw_one_rect(one_obs_info['obs_traj'],
                                                 image_car_traj,
                                                 box_points, 255)

        else:
            image_car_traj, traj = draw_one_rect(one_obs_info['obs_traj'],
                                                 image_car_traj,
                                                 box_points, 125)
        traj_list.append(traj)
        index += 1
    # 动态地图信息
    dynamic_states = list()
    dynamic_pos = list()
    for key, value in data_dict['dynamic_states'].items():
        dynamic_pos.append(value[0])
        dynamic_state = value[1:]
        if len(dynamic_state) < 91:
            dynamic_state = dynamic_state + [0] * (91 - len(dynamic_state))
        dynamic_states.append(dynamic_state)
    if len(predicted_obs_index) < 1:
        raise ValueError("predicted_obs_index < 1")
    image = image_roadmap
    traj_arr = np.stack(traj_list, axis=0)
    obs_feature_list = np.stack(obs_feature_list, axis=0)

    dynamic_states = np.array(dynamic_states)
    dynamic_pos = np.array(dynamic_pos)
    map_feature_list = [feature['polygon_points'] for feature in data_dict['map_features']]
    one_pkl_data = (image.transpose(2, 1, 0), predicted_obs_index, traj_arr,
                    obs_feature_list, dynamic_states,
                    dynamic_pos, map_feature_list, predicted_obs_real_index)
    return one_pkl_data


def gene_model_input_step_one(scenario: scenario_pb2.Scenario) -> Dict[str, Any]:
    data_dict = dict()
    sdc_track_index = scenario.sdc_track_index
    current_time_index = scenario.current_time_index
    obs_tracks = list()
    # 获取自车当前位置把所有轨迹和车道线转化成自车坐标系
    curr_state = scenario.tracks[scenario.sdc_track_index].states[current_time_index]
    ego_curr_x, ego_curr_y = curr_state.center_x, curr_state.center_y
    ego_curr_heading = curr_state.heading
    predicted_obs_ids = list()
    for predicted_obs in scenario.tracks_to_predict:
        predicted_obs_ids.append(scenario.tracks[predicted_obs.track_index].id)
    predicted_obs_ids.append(scenario.tracks[scenario.sdc_track_index].id)
    # 一个障碍物的状态(id, 类型, 轨迹)
    for track in scenario.tracks:
        one_obs_track = dict()
        obs_id = track.id
        if hasattr(track, "object_type"):
            object_type = ObjectType[track.object_type]
        else:
            object_type = ObjectType[4]
        obs_traj = list()
        # 遍历障碍物轨迹
        for state in track.states:
            if not state.valid and obs_id not in predicted_obs_ids:
                continue
            if not state.valid and obs_id in predicted_obs_ids and len(obs_traj) > 0:
                obs_traj.append(obs_traj[-1])
                continue
            if 'height' not in one_obs_track and 'length' not in one_obs_track \
                    and 'width' not in one_obs_track:
                one_obs_track['height'] = state.height
                one_obs_track['length'] = state.length
                one_obs_track['width'] = state.width
            center_x, center_y = global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                                 state.center_x, state.center_y)
            center_heading = theta_global_to_local(ego_curr_heading, state.heading)
            # 速度变化有两种方法:
            # 1. 在新坐标系做投影
            # 2. 计算新坐标系的航向角,用速度乘sin和cos
            curr_v = math.sqrt(math.pow(state.velocity_x, 2) +
                               math.pow(state.velocity_y, 2))
            curr_v_heading = math.atan2(state.velocity_y, state.velocity_x)
            curr_v_heading = theta_global_to_local(ego_curr_heading, curr_v_heading)
            obs_traj.append(
                (center_x, center_y, center_heading,
                 curr_v * math.cos(curr_v_heading), curr_v * math.sin(curr_v_heading))
            )
        one_obs_track['obs_id'] = obs_id
        one_obs_track['object_type'] = object_type
        one_obs_track['obs_traj'] = obs_traj
        # 轨迹丢失的障碍物不需要
        if len(obs_traj) >= 11:
            obs_tracks.append(one_obs_track)
    # 全局地图信息
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
            polygon_point_x, polygon_point_y = global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                                               polygon_point.x, polygon_point.y)
            polygon_points.append((polygon_point_x, polygon_point_y))
        one_map_dict['map_id'] = map_id
        one_map_dict['map_type'] = map_type
        one_map_dict['polygon_points'] = polygon_points
        map_features.append(one_map_dict)
    # 动态地图信息
    dynamic_states = dict()
    for dynamic_state in scenario.dynamic_map_states:
        for lane_state in dynamic_state.lane_states:
            lane_id = lane_state.lane
            lane_x, lane_y = global_to_local(ego_curr_x, ego_curr_y, ego_curr_heading,
                                             lane_state.stop_point.x,
                                             lane_state.stop_point.y)
            if lane_id not in dynamic_states:
                dynamic_states[lane_id] = list()
                dynamic_states[lane_id].append((lane_x, lane_y))
            state = lane_state.state
            dynamic_states[lane_id].append(state)
    data_dict['predicted_obs_ids'] = predicted_obs_ids
    data_dict['obs_tracks'] = obs_tracks
    data_dict['map_features'] = map_features
    data_dict['dynamic_states'] = dynamic_states
    data_dict['curr_loc'] = (ego_curr_x, ego_curr_y, ego_curr_heading, sdc_track_index)
    return data_dict


def inference(input_batch: List) -> np.ndarray:
    input_shape = (256, 256)
    num_timesteps = 100
    schedule_low = 1e-4
    schedule_high = 0.008
    betas = generate_linear_schedule(
        num_timesteps,
        schedule_low * 1000 / num_timesteps,
        schedule_high * 1000 / num_timesteps,
    )
    diffusion_model = GaussianDiffusion(SimpleViT(), MapEncoder(), TrajDecorder(),
                                        input_shape, 3, betas=betas)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and False:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    diffusion_model = diffusion_model.to(device)
    model_dict = diffusion_model.state_dict()
    pretrained_dict = torch.load(MODEL_PATH, map_location=device)
    # 模型参数赋值
    new_model_dict = dict()
    for key in model_dict.keys():
        if ("module." + key) in pretrained_dict:
            new_model_dict[key] = pretrained_dict["module." + key]
        elif key in pretrained_dict:
            new_model_dict[key] = pretrained_dict[key]
        else:
            print("key: ", key, ", not in pretrained")
    diffusion_model.load_state_dict(new_model_dict)
    print("load param success")
    results = list()
    for one_batch_input in input_batch:
        result = diffusion_model.sample(one_batch_input)[0]
        results.append(result)
    results = torch.cat(results, dim=0)
    return results.numpy()


def local_to_global(ego_heading: float, position_x: np.ndarray, position_y: np.ndarray, ego_local_x: float,
                    ego_local_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将世界坐标系下的x,y坐标转成自车坐标系下的坐标
    Args:
        ego_heading: 自车在世界系的转角
        position_x: 要转换的x坐标
        position_y: 要转换的y坐标
        ego_local_x: 自车在世界系的位置
        ego_local_y: 自车在世界系的位置
    Returns: 世界坐标系的x坐标, 世界坐标系的y坐标
    """
    yaw = ego_heading
    # global_x = [(ego_local_x + x * math.cos(yaw) - y * math.sin(yaw))  for x, y in zip(position_x.tolist(), position_y.tolist())]
    # global_y = [(ego_local_y + x * math.sin(yaw) + y * math.cos(yaw))  for x, y in zip(position_x.tolist(), position_y.tolist())]
    # return np.array(global_x), np.array(global_y)
    global_x = ego_local_x + position_x * math.cos(yaw) - position_y * math.sin(yaw)
    global_y = ego_local_y + position_x * math.sin(yaw) + position_y * math.cos(yaw)
    return global_x, global_y


def theta_local_to_global(ego_heading: float, heading: np.ndarray) -> np.ndarray:
    """
    将自车坐标系下的角度转成世界坐标系下的角度
    Args:
        ego_heading: 自车在世界系的转角
        heading: 要转换的heading
    Returns: 世界标系下的heading
    """
    heading = heading.tolist()
    heading_list = []
    for one_heading in heading:
        heading_list.append(normalize_angle(ego_heading + one_heading))
    return np.array(heading_list)


def simulate_with_extrapolation_new(
        scenario: scenario_pb2.Scenario,
        print_verbose_comments: bool = True) -> tf.Tensor:
    vprint = print if print_verbose_comments else lambda arg: None

    # To load the data, we create a simple tensorized version of the object tracks.
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    # Using `ObjectTrajectories` we can select just the objects that we need to
    # simulate and remove the "future" part of the Scenario.
    vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
    logged_trajectories = logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    logged_trajectories = logged_trajectories.slice_time(
        start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
    vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
    # We can verify that all of these objects are valid at the last step.
    vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')
    # 数据预处理和模型推理
    all_logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    all_logged_trajectories = all_logged_trajectories.slice_time(
        start_index=0, end_index=submission_specs.N_FULL_SCENARIO_STEPS + 1)
    logged_pred = all_logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_evaluation_sim_agent_ids(scenario)))
    # if not tf.reduce_all(logged_pred.valid):
    #     print("logged_pred include invalid state")
    predicted_obs_id = submission_specs.get_evaluation_sim_agent_ids(scenario)
    data_dict = gene_model_input_step_one(scenario)
    one_pkl_data = gene_model_input_step_two(data_dict)
    input_batch = gene_model_input_step_three(one_pkl_data, predicted_obs_id)
    predicted_obs_traj = inference(input_batch)
    predicted_obs_id_in_pkl = one_pkl_data[7]
    predicted_obs_traj = predicted_obs_traj[:len(predicted_obs_id_in_pkl)]
    # 推理出来的轨迹与id一一对应
    predicted_obs_id_traj = {obs_id: predicted_obs_traj[index] for index, obs_id in enumerate(predicted_obs_id_in_pkl)}
    # 自车在当前时刻的位置
    curr_loc = data_dict['curr_loc']

    simulated_states = list()
    for index, obs_id in enumerate(submission_specs.get_sim_agent_ids(scenario)):
        if obs_id not in predicted_obs_id_traj.keys():
            simulated_states.append(np.zeros(shape=(80, 4)))
        else:
            one_predicted_obs_traj = predicted_obs_id_traj[obs_id]
            one_predicted_obs_x = one_predicted_obs_traj[:, 0]
            one_predicted_obs_y = one_predicted_obs_traj[:, 1]
            one_predicted_obs_z = np.array([float(logged_trajectories.z[:, -1][index])] * 80)
            one_predicted_obs_x, one_predicted_obs_y = local_to_global(curr_loc[2], one_predicted_obs_x,
                                                                       one_predicted_obs_y, curr_loc[0], curr_loc[1])
            one_predicted_obs_heading = theta_local_to_global(curr_loc[2], one_predicted_obs_traj[:, 2])
            one_simulated_state = np.stack((one_predicted_obs_x, one_predicted_obs_y,
                                            one_predicted_obs_z, one_predicted_obs_heading), axis=-1)
            simulated_states.append(one_simulated_state)
    simulated_states = np.stack(simulated_states, axis=0)
    simulated_states = np.stack([simulated_states] * submission_specs.N_ROLLOUTS, axis=0)
    simulated_states = tf.convert_to_tensor(simulated_states)
    return logged_trajectories, simulated_states


def simulate_with_extrapolation(
        scenario: scenario_pb2.Scenario,
        print_verbose_comments: bool = True) -> tf.Tensor:
    vprint = print if print_verbose_comments else lambda arg: None

    # To load the data, we create a simple tensorized version of the object tracks.
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario)
    # Using `ObjectTrajectories` we can select just the objects that we need to
    # simulate and remove the "future" part of the Scenario.
    vprint(f'Original shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')
    logged_trajectories = logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario)))
    logged_trajectories = logged_trajectories.slice_time(
        start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
    vprint(f'Modified shape of tensors inside trajectories: {logged_trajectories.valid.shape} (n_objects, n_steps)')

    # We can verify that all of these objects are valid at the last step.
    vprint(f'Are all agents valid: {tf.reduce_all(logged_trajectories.valid[:, -1]).numpy()}')

    # We extract the speed of the sim agents (in the x/y/z components) ready for
    # extrapolation (this will be our policy).
    states = tf.stack([logged_trajectories.x, logged_trajectories.y,
                       logged_trajectories.z, logged_trajectories.heading],
                      axis=-1)
    n_objects, n_steps, _ = states.shape
    last_velocities = states[:, -1, :3] - states[:, -2, :3]
    # We also make the heading constant, so concatenate 0. as angular speed.
    last_velocities = tf.concat(
        [last_velocities, tf.zeros((n_objects, 1))], axis=-1)
    # It can happen that the second to last state of these sim agents might be
    # invalid, so we will set a zero speed for them.
    vprint(f'Is any 2nd to last state invalid: {tf.reduce_any(tf.logical_not(logged_trajectories.valid[:, -2]))}')
    vprint(f'This will result in either min or max speed to be really large: {tf.reduce_max(tf.abs(last_velocities))}')
    valid_diff = tf.logical_and(logged_trajectories.valid[:, -1],
                                logged_trajectories.valid[:, -2])
    # `last_velocities` shape: (n_objects, 4).
    last_velocities = tf.where(valid_diff[:, tf.newaxis],
                               last_velocities,
                               tf.zeros_like(last_velocities))
    vprint(f'Now this should be back to a normal value: {tf.reduce_max(tf.abs(last_velocities))}')

    # Now we carry over a simulation. As we discussed, we actually want 32 parallel
    # simulations, so we make this batched from the very beginning. We add some
    # random noise on top of our actions to make sure the behaviours are different.
    # To properly scale the noise, we get the max velocities (average over all
    # objects, corresponding to axis 0) in each of the dimensions (x/y/z/heading).
    NOISE_SCALE = 0.01
    # `max_action` shape: (4,).
    max_action = tf.reduce_max(last_velocities, axis=0)
    # We create `simulated_states` with shape (n_rollouts, n_objects, n_steps, 4).
    simulated_states = tf.tile(states[tf.newaxis, :, -1:, :], [submission_specs.N_ROLLOUTS, 1, 1, 1])
    vprint(f'Shape: {simulated_states.shape}')

    for step in range(submission_specs.N_SIMULATION_STEPS):
        current_state = simulated_states[:, :, -1, :]
        # Random actions, take a normal and normalize by min/max actions
        action_noise = tf.random.normal(
            current_state.shape, mean=0.0, stddev=NOISE_SCALE)
        actions_with_noise = last_velocities[None, :, :] + (action_noise * max_action)
        next_state = current_state + actions_with_noise
        simulated_states = tf.concat(
            [simulated_states, next_state[:, :, None, :]], axis=2)

    # We also need to remove the first time step from `simulated_states` (it was
    # still history).
    # `simulated_states` shape before: (n_rollouts, n_objects, 81, 4).
    # `simulated_states` shape after: (n_rollouts, n_objects, 80, 4).
    simulated_states = simulated_states[:, :, 1:, :]
    vprint(f'Final simulated states shape: {simulated_states.shape}')

    return logged_trajectories, simulated_states


def joint_scene_from_states(
        states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.JointScene:
    # States shape: (num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    states = states.numpy()
    simulated_trajectories = []
    for i_object in range(len(object_ids)):
        simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
            object_id=object_ids[i_object]
        ))
    return sim_agents_submission_pb2.JointScene(
        simulated_trajectories=simulated_trajectories)


# Now we can replicate this strategy to export all the parallel simulations.
def scenario_rollouts_from_states(
        scenario: scenario_pb2.Scenario,
        states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.ScenarioRollouts:
    # States shape: (num_rollouts, num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    joint_scenes = []
    for i_rollout in range(states.shape[0]):
        joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
    return sim_agents_submission_pb2.ScenarioRollouts(
        # Note: remember to include the Scenario ID in the proto message.
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


def inference_valid_set():
    file_names = os.path.join(VALID_PATH, "validation*")
    match_filenames = tf.io.matching_files(file_names)
    dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
    dataset_iterator = dataset.as_numpy_iterator()
    for data in dataset_iterator:
        scenario = scenario_pb2.Scenario.FromString(data)
        valid_list = [index for index, track in enumerate(scenario.tracks) if track.states[10].valid]
        predicted_list = [obs.track_index for obs in scenario.tracks_to_predict]
        print(valid_list)
        print(predicted_list)
        # logged_trajectories, simulated_states = simulate_with_extrapolation(
        #     scenario, print_verbose_comments=True)

        logged_trajectories, simulated_states = simulate_with_extrapolation_new(
            scenario, print_verbose_comments=True)
        # # Package the first simulation into a `JointScene`
        joint_scene = joint_scene_from_states(simulated_states[0, :, :, :],
                                              logged_trajectories.object_id)
        # Validate the joint scene. Should raise an exception if it's invalid.
        submission_specs.validate_joint_scene(joint_scene, scenario)
        scenario_rollouts = scenario_rollouts_from_states(
            scenario, simulated_states, logged_trajectories.object_id)
        # As before, we can validate the message we just generate.
        submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)
        # Compute the features for a single JointScene.
        # single_scene_features = metric_features.compute_metric_features(
        #     scenario, joint_scene)
        config = metrics.load_metrics_config()
        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollouts)
        print(scenario_metrics)


def inference_test_set():
    OUTPUT_ROOT_DIRECTORY = r'waymo_output'
    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    output_filenames = []
    file_names = os.path.join(TEST_PATH, "testing*")
    match_filenames = tf.io.matching_files(file_names)
    for shard_filename in match_filenames:
        print(f"{shard_filename} start inference")
        # extract the suffix.
        shard_suffix = shard_filename.numpy().decode('utf8')[-len('-00000-of-00150'):]
        shard_dataset = tf.data.TFRecordDataset([shard_filename])
        shard_iterator = shard_dataset.as_numpy_iterator()
        scenario_rollouts = []
        for scenario_bytes in tqdm.tqdm(shard_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            logged_trajectories, simulated_states = simulate_with_extrapolation_new(
                scenario, print_verbose_comments=False)
            sr = scenario_rollouts_from_states(
                scenario, simulated_states, logged_trajectories.object_id)
            submission_specs.validate_scenario_rollouts(sr, scenario)
            scenario_rollouts.append(sr)
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name='your_account@test.com',
            unique_method_name='sim_agents_tutorial',
            authors=['test'],
            affiliation='waymo',
            description='Submission from the Sim Agents tutorial',
            method_link='https://waymo.com/open/'
        )
        # Now we can export this message to a binproto, saved to local storage.
        output_filename = f'submission.binproto{shard_suffix}'
        with open(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename), 'wb') as f:
            f.write(shard_submission.SerializeToString())
        output_filenames.append(output_filename)

    # Once we have created all the shards, we can package them directly into a
    # tar.gz archive, ready for submission.
    with tarfile.open(
            os.path.join(OUTPUT_ROOT_DIRECTORY, 'submission.tar.gz'), 'w:gz') as tar:
        for output_filename in output_filenames:
            tar.add(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename),
                    arcname=output_filename)


def cal_dynamic_map_states(is_test: bool = True):
    if is_test:
        file_names = os.path.join(TEST_PATH, "testing*")
        match_filenames = tf.io.matching_files(file_names)
    else:
        file_names = os.path.join(VALID_PATH, "validation*")
        match_filenames = tf.io.matching_files(file_names)
    dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
    dataset_iterator = dataset.as_numpy_iterator()
    for scenario_bytes in tqdm.tqdm(dataset_iterator):
        scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
        print("dynamic_map_states list")
        print([len(state.lane_states) for state in scenario.dynamic_map_states])


if __name__ == "__main__":
    # inference_valid_set()
    # inference_test_set()
    cal_dynamic_map_states(is_test=False)