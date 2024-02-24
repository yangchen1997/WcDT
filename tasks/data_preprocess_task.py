#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_preprocess_task.py
@Author: YangChen
@Date: 2023/12/23
"""
import os
import pickle
import shutil

import tensorflow as tf
import tqdm
from waymo_open_dataset.protos import scenario_pb2

from common import LoadConfigResultDate, TaskType
from tasks.base_task import BaseTask
from utils import DataUtil


class DataPreprocessTask(BaseTask):
    def __init__(self):
        super(DataPreprocessTask, self).__init__()
        self.task_type = TaskType.DATA_PREPROCESS

    def execute(self, result_info: LoadConfigResultDate):
        # 清空原始路径
        data_preprocess_dir = result_info.task_config.data_preprocess_dir
        if os.path.exists(data_preprocess_dir):
            shutil.rmtree(data_preprocess_dir)
        os.makedirs(data_preprocess_dir, exist_ok=True)
        self.load_waymo_train_data(result_info)

    @staticmethod
    def check_waymo_dir(result_info: LoadConfigResultDate):
        # 检查waymo数据路径
        task_config = result_info.task_config
        dataset_names = ["train_set", "val_set", "test_set"]
        dataset_dirs = [task_config.waymo_train_dir,
                        task_config.waymo_val_dir,
                        task_config.waymo_test_dir]
        for dataset_name, dataset_dir in zip(dataset_names, dataset_dirs):
            if not os.path.isdir(dataset_dir) or not os.path.isdir(dataset_dir):
                error_info = f"waymo {dataset_name} error, {dataset_dir} is not a dir"
                result_info.task_logger.logger.error(error_info)
                raise ValueError(error_info)
            if len(os.listdir(dataset_dir)) == 0 and dataset_names == "train_set":
                warn_info = f"waymo {dataset_name} warn, {dataset_dir} size = 0"
                result_info.task_logger.logger.warn(warn_info)

    @classmethod
    def load_waymo_train_data(cls, result_info: LoadConfigResultDate):
        # 读取参数
        data_size = result_info.data_preprocess_config.data_size
        max_data_size = result_info.data_preprocess_config.max_data_size
        waymo_train_dir = result_info.task_config.waymo_train_dir
        preprocess_dir = result_info.task_config.data_preprocess_dir
        # 加载数据
        file_names = os.path.join(waymo_train_dir, "*training.*")
        match_filenames = tf.io.matching_files(file_names)
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
        dataset_iterator = dataset.as_numpy_iterator()
        bar = tqdm.tqdm(dataset_iterator, desc="load waymo train data: ")
        all_data = list()
        result_info.task_logger.logger.info("load_waymo_train_data start")
        number = 0
        for index, data in enumerate(bar):
            scenario = scenario_pb2.Scenario.FromString(data)
            data_dict = DataUtil.load_scenario_data(scenario)
            if len(data_dict) == 0:
                result_info.task_logger.logger.warn(f"scenario: {index} obs track is none")
                continue
            all_data.append(data_dict)
            number += 1
            if number % data_size == 0:
                file_name = f"result_{number}.pkl"
                with open(os.path.join(preprocess_dir, file_name), 'wb') as file:
                    pickle.dump(all_data, file)
                all_data.clear()
                result_info.task_logger.logger.warn(f"file: {file_name} save success")
            if number > max_data_size:
                break
        result_info.task_logger.logger.info("load_waymo_train_data success")
