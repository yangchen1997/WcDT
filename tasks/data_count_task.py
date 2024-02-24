#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_count_task.py
@Author: YangChen
@Date: 2023/12/26
"""
import glob
import os
import pickle
import shutil

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from common import TaskType, LoadConfigResultDate
from tasks import BaseTask


class DataCountTask(BaseTask):

    def __init__(self):
        super(DataCountTask, self).__init__()
        self.task_type = TaskType.DATA_COUNT

    def execute(self, result_info: LoadConfigResultDate):
        logger = result_info.task_logger.get_logger()
        result_dir = result_info.task_config.result_dir
        self.rebuild_dir(result_dir)
        pkl_dir = result_info.task_config.train_dir
        self.check_dir_exist(pkl_dir)
        pkl_path_list = glob.glob(os.path.join(pkl_dir, "*.pkl"))
        sum_obs = list()
        num_predicted_obs = list()
        num_other_obs = list()
        num_lanes = list()
        num_traffic_light = list()
        for pkl_path in pkl_path_list:
            with open(pkl_path, "rb") as f:
                pkl_obj = pickle.load(f)
            sum_obs.append(len(pkl_obj[1]))
            num_predicted_obs.append(len(pkl_obj[0]))
            num_other_obs.append(len(pkl_obj[1]) - len(pkl_obj[0]))
            num_lanes.append(len(pkl_obj[4]))
            num_traffic_light.append(len(pkl_obj[5]))
        logger.info("data count success")
        result = {
            "sum_obs": sum_obs,
            "num_predicted_obs": num_predicted_obs,
            "num_other_obs": num_other_obs,
            "num_lanes": num_lanes,
            "num_traffic_light": num_traffic_light
        }
        with open(os.path.join(result_dir, f"{result_info.task_id}.pkl"), "wb") as f:
            pickle.dump(result, f)
        sum_obs = np.array(sum_obs)
        num_predicted_obs = np.array(num_predicted_obs)
        num_other_obs = np.array(num_other_obs)
        num_lanes = np.array(num_lanes)
        num_traffic_light = np.array(num_traffic_light)
        #
        feature_array = np.stack(
            (sum_obs, num_predicted_obs, num_other_obs, num_lanes, num_traffic_light),
            axis=-1
        )
        scaler = MinMaxScaler()
        # 拟合并转换数据
        feature_array = scaler.fit_transform(feature_array)
        # 创建PCA模型
        pca = PCA(n_components=2)
        feature_array = pca.fit_transform(feature_array)
        # 聚类
        kmeans = KMeans(n_clusters=4, random_state=5)
        kmeans.fit(feature_array)
        y_pred = kmeans.predict(feature_array)
        plt.scatter(feature_array[:, 0], feature_array[:, 1], c=y_pred, s=5, cmap='viridis')
        plt.show()

