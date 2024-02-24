#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_config.py
@Author: YangChen
@Date: 2023/12/20
"""
from typing import List

from common.data import TaskType, BaseConfig


class TaskConfig(BaseConfig):
    __task_list: List[TaskType] = None
    # 所有输出的目录
    output_dir: str = ""
    # log保存路径
    log_dir: str = ""
    model_dir: str = ""
    result_dir: str = ""
    pre_train_model: str = ""
    # waymo数据的目录
    waymo_train_dir: str = ""
    waymo_val_dir: str = ""
    waymo_test_dir: str = ""
    # 训练产生的图片保存路径
    image_dir: str = ""
    # 数据输出
    data_output: str = ""
    # 数据预处理保存路径
    data_preprocess_dir: str = ""
    # 数据集构建
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""

    @property
    def task_list(self) -> List[TaskType]:
        return self.__task_list

    @task_list.setter
    def task_list(self, task_list: List[str]):
        self.__task_list = [TaskType(task_name) for task_name in task_list]

    def check_config(self):
        """
        检查配置文件的输入
        @return:
        """
        if len(self.task_list) < 0:
            raise Warning("task_list is None")
