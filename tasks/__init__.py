#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: __init__.py.py
@Author: YangChen
@Date: 2023/12/20a
"""

from tasks.base_task import BaseTask
from tasks.load_config_task import LoadConfigTask
from tasks.data_preprocess_task import DataPreprocessTask
from tasks.data_split_task import DataSplitTask
from tasks.data_count_task import DataCountTask
from tasks.train_model_task import TrainModelTask
from tasks.show_result_task import ShowResultsTask

BaseTask = BaseTask
LoadConfigTask = LoadConfigTask
DataPreprocessTask = DataPreprocessTask
DataSplitTask = DataSplitTask
TrainModelTask = TrainModelTask
ShowResultsTask = ShowResultsTask
