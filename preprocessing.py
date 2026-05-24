#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理模块
  - butter_lowpass_filter: Butterworth 零相位低通滤波器
  - preprocess_wisdm: WISDM 预处理流水线 (10Hz 滤波 → 0.3Hz 去重力 → 滑动窗口)
"""

import numpy as np
from scipy.signal import butter, sosfilt


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Butterworth 低通滤波器 (零相位)"""
    nyq = 0.5 * fs
    # 防止截止频率等于或超过奈奎斯特频率
    norm_cutoff = min(cutoff / nyq, 1.0 - 1e-6)
    sos = butter(order, norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, data)


def preprocess_wisdm(data, labels, fs=20.0, win=40, step=20):
    """WISDM 预处理: 低通滤波(10Hz) → 去重力 → 滑动窗口

    Args:
        data: N×3 原始加速度数组 (x, y, z)
        labels: N 个活动标签
        fs: 采样率 (Hz)
        win: 窗口长度 (样本数)
        step: 滑动步长 (样本数)

    Returns:
        segments: M×win×3 数组
        seg_labels: M 个活动标签
    """
    # 1. 低通滤波 10Hz
    data_filt = np.column_stack([
        butter_lowpass_filter(data[:, i], cutoff=10, fs=fs, order=4)
        for i in range(3)
    ])

    # 2. 去重力: 0.3Hz 低通提取重力 → 原始减重力
    gravity = np.column_stack([
        butter_lowpass_filter(data_filt[:, i], cutoff=0.3, fs=fs, order=4)
        for i in range(3)
    ])
    body_acc = data_filt - gravity

    # 3. 按活动分别滑动窗口
    X_seg, y_seg = [], []
    for act in np.unique(labels):
        mask = labels == act
        act_data = body_acc[mask]
        for start in range(0, len(act_data) - win + 1, step):
            seg = act_data[start:start + win]
            X_seg.append(seg)
            y_seg.append(act)

    return np.array(X_seg), np.array(y_seg)
