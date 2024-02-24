#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: map_utils.py
@Author: YangChen
@Date: 2023/12/20
"""
import math
from typing import Tuple

import numpy as np


class MapUtil:
    @staticmethod
    def local_to_global(ego_heading: float, position_x: np.ndarray,
                        position_y: np.ndarray, ego_local_x: float,
                        ego_local_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        将坐标系从自车系转为世界系
        @param ego_heading: 自车在世界系的转角
        @param position_x: 要转换的x坐标
        @param position_y: 要转换的y坐标
        @param ego_local_x: 自车在世界系的位置
        @param ego_local_y: 自车在世界系的位置
        @return: 世界坐标系的x坐标, 世界坐标系的y坐标
        """
        yaw = ego_heading
        global_x = ego_local_x + position_x * math.cos(yaw) - position_y * math.sin(yaw)
        global_y = ego_local_y + position_x * math.sin(yaw) + position_y * math.cos(yaw)
        return global_x, global_y

    @classmethod
    def theta_local_to_global(cls, ego_heading: float, heading: np.ndarray) -> np.ndarray:
        """
        将自车坐标系下的角度转成世界坐标系下的角度
        @param ego_heading: 自车在世界系的转角
        @param heading: 要转换的heading
        @return: 世界标系下的heading
        """
        heading = heading.tolist()
        heading_list = []
        for one_heading in heading:
            heading_list.append(cls.normalize_angle(ego_heading + one_heading))
        return np.array(heading_list)

    @staticmethod
    def global_to_local(curr_x: float, curr_y: float, curr_heading: float,
                        point_x: float, point_y: float) -> Tuple[float, float]:
        """
        将世界系的坐标转成自车系
        @param curr_x:
        @param curr_y:
        @param curr_heading:
        @param point_x:
        @param point_y:
        @return:
        """
        delta_x = point_x - curr_x
        delta_y = point_y - curr_y
        return delta_x * math.cos(curr_heading) + delta_y * math.sin(curr_heading),\
            delta_y * math.cos(curr_heading) - delta_x * math.sin(curr_heading)

    @classmethod
    def theta_global_to_local(cls, curr_heading: float, heading: float) -> float:
        """
        将世界坐标系下的角度转成自车坐标系下的角度
        @param curr_heading: 自车在世界系的转角
        @param heading: 要转换的heading
        @return: 自车坐标系下的heading
        """
        return cls.normalize_angle(-curr_heading + heading)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        归一化弧度值, 使其范围在[-pi, pi]之间
        @param angle: 输入的弧度
        @return: 归一化之后的弧度
        """
        angle = (angle + math.pi) % (2 * math.pi)
        if angle < .0:
            angle += 2 * math.pi
        return angle - math.pi
