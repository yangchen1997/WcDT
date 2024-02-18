#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Scene-Diffusion
@Name: data.py
@Author: YangChen
@Date: 2023/12/20
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any


def object_dict_print(obj: Any) -> str:
    """
    打印基础类里面所有的属性
    @param obj:
    @return:
    """
    result_str = ""
    for key, value in obj.__dict__.items():
        if isinstance(value, list) and len(value) > 0:
            list_str = f"{key}:\n\t["
            for row in value:
                list_str += f"{str(row)}, "
            list_str = list_str[:-2]
            list_str += "]\n"
            result_str += list_str
        else:
            result_str += f"{key}: {value} \n"
    return result_str


class TaskType(Enum):
    LOAD_CONFIG = "LOAD_CONFIG"
    DATA_PREPROCESS = "DATA_PREPROCESS"
    DATA_SPLIT = "DATA_SPLIT"
    DATA_COUNT = "DATA_COUNT"
    TRAIN_MODEL = "TRAIN_MODEL"
    SHOW_RESULTS = "SHOW_RESULTS"
    EVAL_MODEL = "EVAL_MODEL"
    GENE_SUBMISSION = "GENE_SUBMISSION"
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value


@dataclass
class BaseConfig(object):

    def __str__(self) -> str:
        return object_dict_print(self)


class TaskLogger(object):
    """
    输出日志
    Args:
        log_path: 日志的路径
    """

    def __init__(self, log_path: str):
        super(TaskLogger, self).__init__()
        # 创建一个日志器
        self.logger = logging.getLogger("logger")

        # 设置日志输出的最低等级,低于当前等级则会被忽略
        self.logger.setLevel(logging.INFO)

        # 创建处理器：sh为控制台处理器，fh为文件处理器
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_path, encoding="UTF-8", mode='w')

        # 创建格式器,并将sh，fh设置对应的格式
        format_str = "%(asctime)s -%(name)s -%(levelname)-8s -%(filename)s(line: %(lineno)s):  %(message)s"
        formatter = logging.Formatter(fmt=format_str, datefmt="%Y/%m/%d %X")
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)

        # 将处理器，添加至日志器中
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def get_logger(self) -> logging.Logger:
        return self.logger


if __name__ == "__main__":
    pass
