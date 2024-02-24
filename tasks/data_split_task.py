#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_split_task.py
@Author: YangChen
@Date: 2023/12/24
"""
import os
import pickle
from multiprocessing import Pool
from typing import Any, Dict

from tqdm import tqdm

from common import TaskType, LoadConfigResultDate
from tasks import BaseTask
from utils import DataUtil


class DataSplitTask(BaseTask):
    def __init__(self):
        super(DataSplitTask, self).__init__()
        self.task_type = TaskType.DATA_SPLIT

    def execute(self, result_info: LoadConfigResultDate):
        logger = result_info.task_logger.get_logger()
        train_dir = result_info.task_config.train_dir
        self.rebuild_dir(train_dir)
        data_preprocess_dir = result_info.task_config.data_preprocess_dir
        self.check_dir_exist(data_preprocess_dir)
        pkl_list = sorted(os.listdir(data_preprocess_dir),
                          key=lambda x: int(x[:-4].split('_')[-1]))
        data_set_num = 0
        his_step = result_info.train_model_config.his_step
        for pkl_path in pkl_list:
            pool = Pool(result_info.data_preprocess_config.num_works)
            with open(os.path.join(data_preprocess_dir, pkl_path), "rb") as f:
                pkl_obj = pickle.load(f)
            process_list = list()
            for one_pkl_dict in pkl_obj:
                data_set_num += 1
                one_pkl_path = os.path.join(train_dir, f"dataset_{data_set_num}.pkl")
                process_list.append(
                    pool.apply_async(
                        self.save_split_data,
                        kwds=dict(
                            one_pkl_dict=one_pkl_dict,
                            his_step=his_step,
                            one_pkl_path=one_pkl_path
                        )
                    )
                )
            for process in tqdm(process_list):
                try:
                    process.get()
                except Exception as e:
                    logger.error(e)
                finally:
                    continue
            pool.close()

    @staticmethod
    def save_split_data(one_pkl_dict: Dict[str, Any], his_step: int, one_pkl_path: str):
        pkl_data = DataUtil.split_pkl_data(one_pkl_dict, his_step)
        if pkl_data and len(pkl_data) > 0:
            with open(one_pkl_path, "wb") as f:
                pickle.dump(pkl_data, f)
