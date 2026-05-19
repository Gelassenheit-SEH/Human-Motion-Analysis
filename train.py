#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块: 数据加载、特征提取、模型训练
"""

import numpy as np
import zipfile
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_wisdm
from feature_extraction import extract_all_features


# ========== UCI HAR 参数 ==========
HAR_FS = 50.0         # 采样率 50Hz
HAR_WIN = 128          # 窗口长度 (2.56s)
HAR_BANDS = [(0, 3), (3, 8), (8, 15)]  # Hz
HAR_AXIS_NAMES = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
]

# ========== WISDM 参数 ==========
WISDM_FS = 20.0        # 采样率 ~20Hz
WISDM_WIN = 40          # 2s × 20Hz
WISDM_STEP = 20         # 50% 重叠
WISDM_BANDS = [(0, 2), (2, 6), (6, 10)]  # Hz
WISDM_AXIS_NAMES = ['x', 'y', 'z']

WISDM_ACT_MAP = {
    'A': 'walking', 'B': 'jogging', 'C': 'stairs',
    'D': 'sitting', 'E': 'standing', 'F': 'typing',
    'G': 'teeth', 'H': 'soup', 'I': 'chips',
    'J': 'pasta', 'K': 'drinking', 'L': 'sandwich',
    'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
    'Q': 'writing', 'R': 'clapping', 'S': 'folding',
}

WISDM_MAJOR_ACTS = ['walking', 'jogging', 'sitting', 'standing', 'stairs']


def load_har_data(data_dir='./dataset'):
    """加载 UCI HAR 数据集

    从 zip 中读取预分割的 train/test 惯性信号和标签。

    Returns:
        train_inertial: dict {axis_name: ndarray (n_windows × 128)}
        test_inertial:  dict {axis_name: ndarray (n_windows × 128)}
        y_train:  1D array, 活动标签 (int 1-6)
        y_test:   1D array, 活动标签 (int 1-6)
        activities: dict {int: activity_name}
    """
    zip_path = os.path.join(data_dir, 'har', 'UCI HAR Dataset.zip')
    if not os.path.exists(zip_path):
        zip_path = os.path.join(
            os.path.dirname(data_dir), 'har_raw', 'UCI HAR Dataset.zip'
        )

    activities = {}
    data = {}

    with zipfile.ZipFile(zip_path) as zf:
        for line in zf.read('UCI HAR Dataset/activity_labels.txt').decode().strip().split('\n'):
            idx, name = line.strip().split()
            activities[int(idx)] = name

        for subset in ['train', 'test']:
            y = np.loadtxt(zf.open(f'UCI HAR Dataset/{subset}/y_{subset}.txt'), dtype=int)

            inertial = {}
            for st in ['body_acc', 'body_gyro']:
                for ax in ['x', 'y', 'z']:
                    key = f'{st}_{ax}'
                    inertial[key] = np.loadtxt(
                        zf.open(f'UCI HAR Dataset/{subset}/Inertial Signals/{key}_{subset}.txt')
                    )

            data[subset] = {'y': y, 'inertial': inertial}

    return (data['train']['inertial'], data['test']['inertial'],
            data['train']['y'], data['test']['y'], activities)


def extract_har_features(inertial_dict, axis_names=None, fs=None, bands=None):
    """对 HAR 惯性信号中所有窗口提取特征

    Args:
        inertial_dict: dict {axis_name: ndarray (n_windows × HAR_WIN)}

    Returns:
        X: ndarray (n_windows × n_features)
        feature_names: list of feature name strings
    """
    if axis_names is None:
        axis_names = HAR_AXIS_NAMES
    if fs is None:
        fs = HAR_FS
    if bands is None:
        bands = HAR_BANDS

    n = len(inertial_dict[axis_names[0]])
    feat_list, name_list = [], None
    for i in range(n):
        data = {ax: inertial_dict[ax][i] for ax in axis_names}
        vec, names = extract_all_features(data, axis_names, fs, bands)
        feat_list.append(vec)
        if name_list is None:
            name_list = names
    return np.array(feat_list), name_list


def load_wisdm_raw(data_dir='./dataset'):
    """加载 WISDM 原始加速度数据

    从前 20 个 subject 文件中解析原始时间序列。

    Returns:
        raw_signals: ndarray (N, 3) — x, y, z 加速度
        raw_labels:  ndarray (N,) — 活动名称字符串
    """
    zip_path = os.path.join(data_dir, 'wisdm', 'wisdm-dataset.zip')
    if not os.path.exists(zip_path):
        zip_path = os.path.join(
            os.path.dirname(data_dir), 'wisdm_raw', 'wisdm-dataset.zip'
        )

    raw_signals, raw_labels = [], []

    with zipfile.ZipFile(zip_path) as zf:
        files = sorted([
            f for f in zf.namelist()
            if f.startswith('wisdm-dataset/raw/phone/accel/') and f.endswith('.txt')
        ])[:20]

        for fname in files:
            for line in zf.read(fname).decode().strip().split('\n'):
                line = line.strip().rstrip(';')
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        raw_signals.append([float(parts[3]), float(parts[4]), float(parts[5])])
                        raw_labels.append(WISDM_ACT_MAP.get(parts[1], parts[1]))
                    except ValueError:
                        pass

    return np.array(raw_signals), np.array(raw_labels)


def extract_wisdm_features(segments, axis_names=None, fs=None, bands=None):
    """对 WISDM 分段数据 (M×win×3) 提取每个窗口的特征

    Returns:
        X: ndarray (M, n_features)
        feature_names: list of str
    """
    if axis_names is None:
        axis_names = WISDM_AXIS_NAMES
    if fs is None:
        fs = WISDM_FS
    if bands is None:
        bands = WISDM_BANDS

    X_list, name_list = [], None
    for i in range(len(segments)):
        data_dict = {ax: segments[i, :, j] for j, ax in enumerate(axis_names)}
        vec, names = extract_all_features(data_dict, axis_names, fs, bands)
        X_list.append(vec)
        if name_list is None:
            name_list = names

    return np.array(X_list), name_list


def filter_major_activities(X, y, major_acts=None):
    """只保留主要活动类别"""
    if major_acts is None:
        major_acts = WISDM_MAJOR_ACTS
    mask = np.isin(y, major_acts)
    return X[mask], y[mask]


def train_classifiers(X_train, y_train, scaler_range=(-1, 1)):
    """MinMax 归一化 + 训练 KNN(k=5) 和 SVM(RBF)

    Returns:
        scaler: 训练好的 MinMaxScaler
        knn:    训练好的 KNeighborsClassifier
        svm:    训练好的 SVC
    """
    scaler = MinMaxScaler(feature_range=scaler_range)
    X_train_s = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(5)
    knn.fit(X_train_s, y_train)

    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_s, y_train)

    return scaler, knn, svm


# ========== WISDM 全流程便利函数 ==========

def prepare_wisdm(data_dir='./dataset', test_size=0.3, random_state=42):
    """加载 → 预处理 → 特征提取 → 筛选主要活动 → train/test 分割

    Returns:
        X_train, X_test, y_train, y_test, scaler, knn, svm
    """
    # 加载原始数据
    raw_signals, raw_labels = load_wisdm_raw(data_dir)
    print(f"WISDM raw samples: {len(raw_signals)}")

    # 预处理 (滤波 + 去重力 + 滑动窗口)
    segments, seg_labels = preprocess_wisdm(
        raw_signals, raw_labels, fs=WISDM_FS, win=WISDM_WIN, step=WISDM_STEP
    )
    print(f"WISDM segments: {len(segments)}, shape={segments.shape}")

    # 特征提取
    X_all, _ = extract_wisdm_features(segments)
    y_all = seg_labels
    print(f"WISDM features per window: {X_all.shape[1]}")

    # 筛选主要活动
    X_major, y_major = filter_major_activities(X_all, y_all)
    acts = np.unique(y_major)
    print(f"Major activities ({len(acts)}): {list(acts)}")

    # 分割训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X_major, y_major, test_size=test_size, random_state=random_state, stratify=y_major
    )
    print(f"WISDM train: {len(X_train)}, test: {len(X_test)}")

    # 训练
    scaler, knn, svm = train_classifiers(X_train, y_train)

    return scaler, knn, svm, X_test, y_test


if __name__ == '__main__':
    # 快速验证 WISDM 训练流程
    print("Testing WISDM training pipeline...")
    result = prepare_wisdm()
    print("Done!")
