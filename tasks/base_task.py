#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: base_task.py
@Author: YangChen
@Date: 2023/12/20
"""
import os
import shutil
from abc import abstractmethod, ABC

from common import TaskType, LoadConfigResultDate


class BaseTask(ABC):
    def __init__(self):
        super(BaseTask, self).__init__()
        self.task_type = TaskType.UNKNOWN

    @abstractmethod
    def execute(self, result_info: LoadConfigResultDate):
        pass

    @staticmethod
    def check_dir_exist(input_dir: str):
        if not os.path.exists(input_dir) or \
                not os.path.isdir(input_dir) or \
                len(os.listdir(input_dir)) <= 0:
            raise FileNotFoundError(f"data_preprocess_dir error: {input_dir}, dir is None")

    @staticmethod
    def rebuild_dir(input_dir: str):
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir, exist_ok=True)
