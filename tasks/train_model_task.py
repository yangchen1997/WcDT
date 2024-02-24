#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: train_model_task.py
@Author: YangChen
@Date: 2023/12/26
"""
import os
import shutil
from typing import Union

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import TaskType, LoadConfigResultDate
from common import WaymoDataset
from net_works import BackBone
from tasks import BaseTask
from utils import MathUtil, VisualizeUtil


class TrainModelTask(BaseTask):

    def __init__(self):
        super(TrainModelTask, self).__init__()
        self.task_type = TaskType.TRAIN_MODEL
        self.device = torch.device("cpu")
        self.multi_gpus = False
        self.gpu_ids = list()

    def execute(self, result_info: LoadConfigResultDate):
        train_dir = result_info.task_config.train_dir
        train_model_config = result_info.train_model_config
        self.init_dirs(result_info)
        # 初始化device
        if train_model_config.use_gpu:
            if train_model_config.gpu_ids:
                self.device = torch.device(f"cuda:{train_model_config.gpu_ids[0]}")
                self.multi_gpus = True
                self.gpu_ids = train_model_config.gpu_ids
            else:
                self.device = torch.device('cuda')
        # 初始化dataloader
        waymo_dataset = WaymoDataset(
            train_dir, train_model_config.his_step, train_model_config.max_pred_num,
            train_model_config.max_other_num, train_model_config.max_traffic_light,
            train_model_config.max_lane_num, train_model_config.max_point_num
        )
        data_loader = DataLoader(
            waymo_dataset,
            shuffle=False,
            batch_size=train_model_config.batch_size,
            num_workers=train_model_config.num_works,
            pin_memory=True,
            drop_last=False
        )
        model = self.init_model(result_info)
        model_train = model.train()
        optimizer = optim.Adam(model_train.parameters(), lr=train_model_config.init_lr,
                               betas=(0.9, 0.999), weight_decay=0)
        epoch_step = len(waymo_dataset) // train_model_config.batch_size
        if epoch_step == 0:
            raise ValueError("dataset is too small, epoch_step = 0")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_model_config.num_epoch, eta_min=1e-6)
        for epoch in range(train_model_config.num_epoch):
            self.fit_one_epoch(epoch, epoch_step, optimizer, model, data_loader, result_info)
            scheduler.step()

    def fit_one_epoch(
            self, epoch_num: int, epoch_step: int,
            optimizer: Optimizer, model: Union[BackBone, nn.DataParallel],
            data_loader: DataLoader, result_info: LoadConfigResultDate
    ):
        diffusion_losses = 0
        traj_losses = 0
        confidence_losses = 0
        total_losses = 0
        train_model_config = result_info.train_model_config
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch_num + 1}/{train_model_config.num_epoch}', mininterval=0.3)
        torch.autograd.set_detect_anomaly(True)
        for iteration, data in enumerate(data_loader):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device).to(torch.float32)
            optimizer.zero_grad()
            diffusion_loss, traj_loss, confidence_loss, min_loss_traj = model(data)
            diffusion_loss = diffusion_loss.mean()
            traj_loss = traj_loss.mean()
            confidence_loss = confidence_loss.mean()
            total_loss = diffusion_loss + traj_loss + confidence_loss
            total_loss.backward()
            optimizer.step()
            total_losses += total_loss.item()
            diffusion_losses += diffusion_loss.item()
            traj_losses += traj_loss.item()
            confidence_losses += confidence_loss.item()
            pbar.set_postfix(**{'total_loss': total_losses / (iteration + 1),
                                'diffusion_loss': diffusion_losses / (iteration + 1),
                                'traj_losses': traj_losses / (iteration + 1),
                                'confidence_losses': confidence_losses / (iteration + 1)})
            pbar.update()
            if iteration % 10 == 0:
                image_path = os.path.join(result_info.task_config.image_dir,
                                          f"epoch_{epoch_num}_batch_num_{iteration}_image.png")
                VisualizeUtil.show_result(image_path, min_loss_traj, data)

    @staticmethod
    def init_dirs(result_info: LoadConfigResultDate):
        task_config = result_info.task_config
        if os.path.exists(task_config.image_dir):
            shutil.rmtree(task_config.image_dir)
        os.makedirs(task_config.image_dir, exist_ok=True)
        os.makedirs(task_config.model_dir, exist_ok=True)

    def init_model(self, result_info: LoadConfigResultDate) -> Union[BackBone, nn.DataParallel]:
        train_model_config = result_info.train_model_config
        task_config = result_info.task_config
        # 初始化diffusion的betas
        if train_model_config.schedule == "cosine":
            betas = MathUtil.generate_cosine_schedule(train_model_config.time_steps)
        else:
            schedule_low = 1e-4
            schedule_high = 0.008
            betas = MathUtil.generate_linear_schedule(
                train_model_config.time_steps,
                schedule_low * 1000 / train_model_config.time_steps,
                schedule_high * 1000 / train_model_config.time_steps,
            )
        model = BackBone(betas)
        if task_config.pre_train_model:
            pre_train_model_path = task_config.pre_train_model
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pre_train_model_path)
            # 模型参数赋值
            new_model_dict = dict()
            for key in model_dict.keys():
                if ("module." + key) in pretrained_dict:
                    new_model_dict[key] = pretrained_dict["module." + key]
                elif key in pretrained_dict:
                    new_model_dict[key] = pretrained_dict[key]
                else:
                    print("key: ", key, ", not in pretrained")
            model.load_state_dict(new_model_dict)
            result_info.task_logger.logger.info("load pre_train_model success")
        model = model.to(self.device)
        if self.multi_gpus:
            model = nn.DataParallel(model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
        return model
