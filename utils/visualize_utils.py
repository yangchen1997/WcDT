#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: visualize_utils.py
@Author: YangChen
@Date: 2023/12/20
"""
from typing import Any, Dict

import torch
from matplotlib import pyplot as plt


class VisualizeUtil:
    @staticmethod
    def show_result(image_path: str, min_loss_traj: torch.Tensor, data: Dict):
        min_loss_traj = min_loss_traj[0]
        predicted_traj_mask = data['predicted_traj_mask'][0]
        predicted_future_traj = data['predicted_future_traj'][0]
        predicted_his_traj = data['predicted_his_traj'][0]
        predicted_num = 0
        for i in range(predicted_traj_mask.shape[0]):
            if int(predicted_traj_mask[i]) == 1:
                predicted_num += 1
        generate_traj = min_loss_traj[:predicted_num]
        predicted_future_traj = predicted_future_traj[:predicted_num]
        predicted_his_traj = predicted_his_traj[:predicted_num]
        map_feature_list = eval(data['map_json'][0])
        real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
        model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # 画地图
        for map_feature in map_feature_list:
            x_list = [float(point[0]) for point in map_feature]
            y_list = [float(point[1]) for point in map_feature]
            ax[0].plot(x_list, y_list, color="grey")
            ax[1].plot(x_list, y_list, color="grey")
        # 画原图,画模型输出
        for i in range(predicted_num):
            ax[0].plot(real_traj[i, :, 0], real_traj[i, :, 1])
            ax[1].plot(model_output[i, :, 0], model_output[i, :, 1])

        # label = 'Epoch {0}'.format(num_epoch)
        # plt.show()
        # fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(image_path)
        plt.close('all')  # 避免内存泄漏
        print("save_image_success")
