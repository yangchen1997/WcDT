#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: load_config_task.py
@Author: YangChen
@Date: 2023/12/20
"""
import os
import random
from datetime import datetime

import torch
import yaml

from common import TaskType, LoadConfigResultDate, TaskLogger
from common import TaskConfig, DataPreprocessConfig, TrainModelConfig
from tasks.base_task import BaseTask

# 可能在不同的地方调用ConfigFactory,因此优先初始化config_file
path_list = os.path.realpath(__file__).split(os.path.sep)
path_list = path_list[:-2]
path_list.append("config.yaml")
CONFIG_PATH = os.path.sep.join(path_list)


class LoadConfigTask(BaseTask):
    """
    主要功能：
        加载配置文件
        检查config是否合理
        根据config初始化output
        根据config初始化log
    """

    def __init__(self):
        super(LoadConfigTask, self).__init__()
        self.task_type = TaskType.LOAD_CONFIG
        with open(CONFIG_PATH, "rb") as file:
            self.__yaml_loader = yaml.safe_load(file)

    def execute(self, result_info: LoadConfigResultDate):
        # 初始化结果信息
        task_config = self.get_task_config()
        data_preprocess_config = self.get_preprocess_config()
        train_model_config = self.get_train_model_config()
        self.init_dirs_and_log(task_config, result_info)
        self.check_preprocess_config(data_preprocess_config)
        self.check_train_model_config(train_model_config)
        # 结果信息赋值
        result_info.task_config = task_config
        result_info.data_preprocess_config = data_preprocess_config
        result_info.train_model_config = train_model_config

    def get_task_config(self) -> TaskConfig:
        task_config = TaskConfig()
        self.__init_config_object_attr(task_config, self.__yaml_loader['task_config'])
        return task_config

    def get_preprocess_config(self) -> DataPreprocessConfig:
        data_preprocess_config = DataPreprocessConfig()
        self.__init_config_object_attr(data_preprocess_config, self.__yaml_loader['data_preprocess_config'])
        return data_preprocess_config

    def get_train_model_config(self) -> TrainModelConfig:
        train_model_config = TrainModelConfig()
        self.__init_config_object_attr(train_model_config, self.__yaml_loader['train_model_config'])
        return train_model_config

    @staticmethod
    def __init_config_object_attr(instance: object, attrs: dict):
        """
        根据字典给配置类对象赋值
        @param instance: 配置类对象
        @param attrs: 从配置文件中读取的属性
        @return: None
        """
        if not instance or not attrs:
            return
        for name, value in attrs.items():
            if hasattr(instance, name):
                setattr(instance, name, value)
            else:
                raise ValueError(f"unknown config, config name is {name}")

    @staticmethod
    def init_dirs_and_log(task_config: TaskConfig, result_data: LoadConfigResultDate):
        """
        检查task_config的参数并初始化路径
        @param task_config:
        @param result_data
        @return:
        """
        # output路径检查
        os.makedirs(task_config.output_dir, exist_ok=True)
        task_config.log_dir = os.path.join(task_config.output_dir, task_config.log_dir)
        os.makedirs(task_config.log_dir, exist_ok=True)
        # 创建log
        now = datetime.now()
        formatted_now = now.strftime('%Y%m%d-%H-%M-%S')
        random_num = random.randint(10000, 99999)
        result_data.task_id = f"{formatted_now}_{random_num}"
        log_file_name = os.path.join(task_config.log_dir, f"{result_data.task_id}.log")
        result_data.task_logger = TaskLogger(log_file_name)
        result_data.task_logger.logger.info(f"task id {result_data.task_id} start")
        # task list里的任务必须唯一
        if len(task_config.task_list) != len(set(task_config.task_list)):
            raise ValueError("task_config must be unique")
        # 创建model save dir
        task_config.model_dir = os.path.join(task_config.output_dir, task_config.model_dir)
        # os.makedirs(task_config.model_dir, exist_ok=True)
        # 创建result dir
        task_config.result_dir = os.path.join(task_config.output_dir, task_config.result_dir)
        # 检查模型路径
        if task_config.pre_train_model:
            path_type = ".pth"
            if not os.path.isfile(task_config.pre_train_model) or \
                    not os.path.exists(task_config.pre_train_model) or \
                    task_config.pre_train_model[-len(path_type):] != path_type:
                raise ValueError("task_config.pre_train_model error")
            result_data.task_logger.logger.info(f"{task_config.pre_train_model} check success")
        else:
            result_data.task_logger.logger.warn("pre_train_model path is None")

        # 初始化图片保存路径
        task_config.image_dir = os.path.join(
            task_config.output_dir,
            task_config.image_dir,
            result_data.task_id
        )
        # os.makedirs(task_config.image_dir, exist_ok=True)
        # 数据预处理输出
        os.makedirs(task_config.data_output, exist_ok=True)
        task_config.data_preprocess_dir = os.path.join(task_config.data_output,
                                                       task_config.data_preprocess_dir)
        # os.makedirs(task_config.data_preprocess_dir, exist_ok=True)
        # 初始化训练集验证集测试集
        task_config.train_dir = os.path.join(task_config.data_output,
                                             task_config.train_dir)
        # os.makedirs(task_config.train_dir, exist_ok=True)
        task_config.val_dir = os.path.join(task_config.data_output,
                                           task_config.val_dir)
        # os.makedirs(task_config.val_dir, exist_ok=True)
        task_config.test_dir = os.path.join(task_config.data_output,
                                            task_config.test_dir)
        # os.makedirs(task_config.test_dir, exist_ok=True)
        result_data.task_logger.logger.info(str(task_config))
        result_data.task_logger.logger.info("task config init success")

    @staticmethod
    def check_preprocess_config(data_preprocess_config: DataPreprocessConfig):
        if data_preprocess_config.num_works <= 0:
            raise ValueError(f"num_works = {data_preprocess_config.num_works}, cannot <= 0")

    @staticmethod
    def check_train_model_config(train_model_config: TrainModelConfig):
        if train_model_config.his_step <= 0 or train_model_config.his_step >= 91:
            raise ValueError(f"his_step {train_model_config.his_step} is out of range")
        if train_model_config.use_gpu and not torch.cuda.is_available():
            raise ValueError(f"cuda is unavailable")
        if train_model_config.use_gpu and len(train_model_config.gpu_ids) > 0 and torch.cuda.device_count() <= 1:
            raise ValueError("only one gpu can used")
        if train_model_config.schedule not in ("cosine", "linear"):
            raise ValueError(f"schedule: {train_model_config.schedule}, is not in (cosine, linear)")
