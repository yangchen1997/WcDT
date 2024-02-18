#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: show_result_task.py
@Author: YangChen
@Date: 2024/1/6
"""
import os.path
import shutil
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs

from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil

RESULT_DIR = r"/home/haomo/yangchen/Scene-Diffusion/output/image"
DATA_SET_PATH = r"/home/haomo/yangchen/Scene-Diffusion/data_set/train_set/training.tfrecord-00000-of-01000"
MODEL_PATH = r"/home/haomo/yangchen/Scene-Diffusion/model_epoch152.pth"


class ShowResultsTask(BaseTask):

    def __init__(self):
        super(ShowResultsTask, self).__init__()
        self.task_type = TaskType.SHOW_RESULTS
        self.cmap = LinearSegmentedColormap.from_list(
            'my_cmap',
            [np.array([0., 232., 157.]) / 255, np.array([0., 120., 255.]) / 255],
            100
        )
        self.color_dict = {
            0: np.array([0., 120., 255.]) / 255,
            1: np.array([0., 232., 157.]) / 255,
            2: np.array([255., 205., 85.]) / 255,
            3: np.array([244., 175., 145.]) / 255,
            4: np.array([145., 80., 200.]) / 255,
            5: np.array([0., 51., 102.]) / 255,
            6: np.array([1, 0, 0]),
            7: np.array([0, 1, 0]),
        }

    def execute(self, result_info: LoadConfigResultDate):
        if os.path.exists(RESULT_DIR):
            shutil.rmtree(RESULT_DIR)
        os.makedirs(RESULT_DIR, exist_ok=True)
        self.show_result(result_info)

    @staticmethod
    def load_pretrain_model(result_info: LoadConfigResultDate) -> BackBone:
        betas = MathUtil.generate_linear_schedule(result_info.train_model_config.time_steps)
        model = BackBone(betas).eval()
        device = torch.device("cpu")
        pretrained_dict = torch.load(MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
        # 模型参数赋值
        new_model_dict = dict()
        for key in model_dict.keys():
            if ("module." + key) in pretrained_dict:
                new_model_dict[key] = pretrained_dict["module." + key]
            elif key in pretrained_dict:
                new_model_dict[key] = pretrained_dict[key]
            else:
                print("key: ", key, ", not in pretrained")
        model.load_state_dict(new_model_dict)
        print("load_pretrain_model success")
        return model

    def show_result(self, result_info: LoadConfigResultDate):
        model = self.load_pretrain_model(result_info)
        match_filenames = tf.io.matching_files([DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data").take(100)
        dataset_iterator = dataset.as_numpy_iterator()
        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)

            predict_traj = model(data_dict)[-1]
            predicted_traj_mask = data_dict['predicted_traj_mask'][0]
            predicted_future_traj = data_dict['predicted_future_traj'][0]
            predicted_his_traj = data_dict['predicted_his_traj'][0]
            predicted_num = 0
            for i in range(predicted_traj_mask.shape[0]):
                if int(predicted_traj_mask[i]) == 1:
                    predicted_num += 1
            generate_traj = predict_traj[:predicted_num]
            predicted_future_traj = predicted_future_traj[:predicted_num]
            predicted_his_traj = predicted_his_traj[:predicted_num]
            real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
            real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].detach().numpy()
            model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
            model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].detach().numpy()
            # 可视化输入
            image_path = os.path.join(RESULT_DIR, f"{index}_input.png")
            self.draw_input(scenario, image_path)
            # 可视化ground truth
            image_path = os.path.join(RESULT_DIR, f"{index}_ground_truth.png")
            self.draw_scene(predicted_num, real_traj, data_dict, scenario, image_path)
            # 可视化model output
            image_path = os.path.join(RESULT_DIR, f"{index}_model_output.png")
            self.draw_scene(predicted_num, model_output, data_dict, scenario, image_path)
            # 可视化动态图
            image_path = os.path.join(RESULT_DIR, f"{index}_ground_truth.gif")
            self.draw_gif(predicted_num, real_traj, real_yaw, data_dict, scenario, image_path)
            image_path = os.path.join(RESULT_DIR, f"{index}_model_output.gif")
            self.draw_gif(predicted_num, model_output, model_yaw, data_dict, scenario, image_path)

        fig, axis = plt.subplots(1, 1, figsize=(10, 11))
        num = np.array([i for i in range(91)])
        segments = np.stack((num, num), axis=-1)[np.newaxis, :]
        image_path = os.path.join(RESULT_DIR, f"color_bar.png")
        line_segments = LineCollection(segments=segments, linewidths=1,
                                       linestyles='solid', cmap=self.cmap)
        cbar = fig.colorbar(line_segments, cmap=self.cmap, orientation='horizontal')
        cbar.set_ticks(np.linspace(0, 1, 10))
        cbar.set_ticklabels([str(i) for i in range(0, 91, 10)])
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏

    @staticmethod
    def draw_input(scenario: Scenario, image_path: str):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        predicted_obs_ids = submission_specs.get_evaluation_sim_agent_ids(scenario)
        current_time_index = scenario.current_time_index
        for track in scenario.tracks:
            if track.id not in predicted_obs_ids:
                continue
            param_dict = {
                "x": track.states[current_time_index].center_x,
                "y": track.states[current_time_index].center_y,
                "bbox_yaw": track.states[current_time_index].heading,
                "length": track.states[current_time_index].length,
                "width": track.states[current_time_index].width,
            }
            rect = visualizations.get_bbox_patch(**param_dict)
            axis.add_patch(rect)
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏

    def draw_scene(
            self, predicted_num: int, traj: np.ndarray,
            data_dict: Dict[str, Any], scenario: Scenario, image_path: str
    ):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        # visualizations.get_bbox_patch()
        # axis.axis('equal')  # 横纵坐标比例相等
        curr_x, curr_y, curr_heading, _ = data_dict['curr_loc']
        for i in range(predicted_num):
            real_traj_x, real_traj_y = MapUtil.local_to_global(curr_heading, traj[i, :, 0],
                                                               traj[i, :, 1], curr_x, curr_y)
            num = np.linspace(0, 1, len(real_traj_x))
            for j in range(2, len(real_traj_x)):
                axis.plot(
                    real_traj_x[j - 2:j],
                    real_traj_y[j - 2:j],
                    linewidth=5,
                    color=self.cmap(num[j]),
                )
        axis.set_xticks([])
        axis.set_yticks([])
        # plt.show()
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏
        print(f"{image_path} save success")

    def draw_gif(
            self, predicted_num: int, traj: np.ndarray, real_yaw: np.ndarray,
            data_dict: Dict[str, Any], scenario: Scenario, image_path: str
    ):
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        visualizations.add_map(axis, scenario)
        # visualizations.get_bbox_patch()
        # axis.axis('equal')  # 横纵坐标比例相等
        curr_x, curr_y, curr_heading, _ = data_dict['curr_loc']
        x_list = list()
        y_list = list()
        yaw_list = list()
        for i in range(predicted_num):
            real_traj_x, real_traj_y = MapUtil.local_to_global(curr_heading, traj[i, :, 0],
                                                               traj[i, :, 1], curr_x, curr_y)
            real_traj_yaw = MapUtil.theta_local_to_global(curr_heading, real_yaw[i])
            x_list.append(real_traj_x)
            y_list.append(real_traj_y)
            yaw_list.append(real_traj_yaw)
        # [num, step]
        x_list = np.stack(x_list, axis=0)
        y_list = np.stack(y_list, axis=0)
        yaw_list = np.stack(yaw_list, axis=0)
        predicted_feature = data_dict['predicted_feature'].squeeze()[:, :2]

        def animate(t: int) -> list[patches.Rectangle]:
            # At each animation step, we need to remove the existing patches. This can
            # only be done using the `pop()` operation.
            for _ in range(len(axis.patches)):
                axis.patches.pop()
            bboxes = list()
            for j in range(x_list.shape[0]):
                bboxes.append(axis.add_patch(
                    self.get_bbox_patch(
                        x_list[:, t][j], y_list[:, t][j], yaw_list[:, t][j],
                        predicted_feature[j, 1], predicted_feature[j, 0], self.color_dict[j]
                    )
                ))
            return bboxes

        animations = animation.FuncAnimation(
            fig, animate, frames=x_list.shape[1], interval=100,
            blit=True)
        axis.set_xticks([])
        axis.set_yticks([])
        # plt.show()
        animations.save(image_path, writer='ffmpeg', fps=30)
        plt.close('all')  # 避免内存泄漏
        print(f"{image_path} save success")

    @staticmethod
    def get_bbox_patch(
            x: float, y: float, bbox_yaw: float, length: float, width: float,
            color: np.ndarray
    ) -> patches.Rectangle:
        left_rear_object = np.array([-length / 2, -width / 2])

        rotation_matrix = np.array([[np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                                    [np.sin(bbox_yaw), np.cos(bbox_yaw)]])
        left_rear_rotated = rotation_matrix.dot(left_rear_object)
        left_rear_global = np.array([x, y]) + left_rear_rotated
        color = list(color) + [0.5]
        rect = patches.Rectangle(
            left_rear_global, length, width, angle=np.rad2deg(bbox_yaw), color=color)
        return rect


if __name__ == "__main__":
    # show_result()
    pass
