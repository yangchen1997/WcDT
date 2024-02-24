#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_preprocess_config.py
@Author: YangChen
@Date: 2023/12/24
"""
from common.data import BaseConfig


class DataPreprocessConfig(BaseConfig):
    data_size: int = 0
    max_data_size: int = 0
    num_works: int = 1

