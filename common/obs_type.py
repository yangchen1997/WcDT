#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: obs_type.py
@Author: YangChen
@Date: 2023/12/24
"""

ObjectType = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
    4: "TYPE_OTHER"
}
MapState = {
    0: "LANE_STATE_UNKNOWN",
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION"
}
