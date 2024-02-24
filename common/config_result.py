#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_load_config_result.py
@Author: YangChen
@Date: 2023/12/21
"""
from dataclasses import dataclass

from common.data import TaskLogger
from common.data_config import TaskConfig
from common.data_preprocess_config import DataPreprocessConfig
from common.data_train_model_config import TrainModelConfig


@dataclass
class LoadConfigResultDate:
    task_config: TaskConfig = None
    data_preprocess_config: DataPreprocessConfig = None
    train_model_config: TrainModelConfig = None
    task_id: str = ""
    task_logger: TaskLogger = None
