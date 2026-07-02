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
from sklearn.ensemble import RandomForestClassifier

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

    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train_s, y_train)
    # 修改
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)

    return scaler, knn, svm,rf


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

    # 同步分割原始分段数据（供 CNN 使用）
    major_mask = np.isin(seg_labels, WISDM_MAJOR_ACTS)
    seg_major = segments[major_mask]
    seg_train, seg_test, _, _ = train_test_split(
        seg_major, y_major, test_size=test_size,
        random_state=random_state, stratify=y_major,
    )

    # 训练
    scaler, knn, svm, rf = train_classifiers(X_train, y_train)

    return scaler, knn, svm, rf, X_test, y_test, seg_train, seg_test, X_train, y_train


# ========== 深度学习集成 ==========

def prepare_har_raw_for_cnn(inertial_dict, axis_names=None):
    """将 HAR 惯性信号字典转换为 CNN 输入格式 (N, channels, timesteps)

    Args:
        inertial_dict: dict {axis_name: ndarray (n_windows × 128)}
        axis_names:    轴名称列表（决定通道顺序）

    Returns:
        X_cnn: ndarray (n_windows, n_channels, n_timesteps)
    """
    if axis_names is None:
        axis_names = HAR_AXIS_NAMES
    channels = [inertial_dict[ax] for ax in axis_names]
    X_cnn = np.stack(channels, axis=1).astype(np.float32)
    return X_cnn


def prepare_wisdm_raw_for_cnn(segments):
    """将 WISDM 分段数据转换为 CNN 输入格式 (M, channels, timesteps)

    Args:
        segments: ndarray (M, win, 3)  → 转置为 (M, 3, win)

    Returns:
        X_cnn: ndarray (M, 3, win)
    """
    return np.transpose(segments, (0, 2, 1)).astype(np.float32)


def train_dl_har(X_features, y_train, X_test_features, y_test,
                 train_inertial, test_inertial, activities,
                 model_types=('mlp', 'cnn', 'cnnlstm'),
                 epochs=80, verbose=True):
    """在 HAR 数据上训练深度学习模型

    Args:
        X_features:     手工特征训练集 (N_train, n_features)
        y_train:        训练标签
        X_test_features: 手工特征测试集
        y_test:         测试标签
        train_inertial: 训练集惯性信号 dict
        test_inertial:  测试集惯性信号 dict
        activities:     {int: name} 活动映射
        model_types:    要训练的模型类型元组
        epochs:         最大训练轮数
        verbose:        是否打印详情

    Returns:
        results: dict {model_name: {'model': nn.Module, 'acc': float, 'y_pred': array}}
        X_cnn_test: CNN 格式的测试数据（用于后续可视化）
    """
    from dl_models import (create_model, train_dl_model, evaluate_dl_model,
                           get_device)

    device = get_device()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Deep Learning on HAR (device: {device})")
        print(f"{'=' * 60}")

    n_classes = len(activities)
    results = {}

    # ---- 准备 CNN 原始信号数据 ----
    if 'cnn' in model_types or 'cnnlstm' in model_types:
        X_cnn_train = prepare_har_raw_for_cnn(train_inertial)
        X_cnn_test = prepare_har_raw_for_cnn(test_inertial)
        n_channels, n_timesteps = X_cnn_train.shape[1], X_cnn_train.shape[2]
        if verbose:
            print(f"CNN input shape: ({n_channels} channels × {n_timesteps} timesteps)")
    else:
        X_cnn_test = None

    # ---- MLP: 基于增强手工特征 ----
    if 'mlp' in model_types:
        if verbose:
            print("\n--- MLP (on enhanced hand-crafted features) ---")
        mlp = create_model('mlp', X_features.shape[1], n_classes,
                           hidden_dims=(256, 128, 64), dropout=0.3)
        mlp, hist, le = train_dl_model(
            mlp, X_features, y_train, model_type='mlp',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        mlp_acc, mlp_pred = evaluate_dl_model(
            mlp, X_test_features, y_test, model_type='mlp',
            label_map=activities,
        )
        results['DL-MLP'] = {'model': mlp, 'acc': mlp_acc, 'y_pred': mlp_pred}

    # ---- CNN1D: 基于原始信号 ----
    if 'cnn' in model_types:
        if verbose:
            print("\n--- CNN1D (on raw inertial signals) ---")
        cnn = create_model('cnn', (n_channels, n_timesteps), n_classes,
                           conv_filters=(64, 128, 256), kernel_size=5, dropout=0.3)
        cnn, hist, le = train_dl_model(
            cnn, X_cnn_train, y_train, model_type='cnn',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        cnn_acc, cnn_pred = evaluate_dl_model(
            cnn, X_cnn_test, y_test, model_type='cnn',
            label_map=activities,
        )
        results['DL-CNN'] = {'model': cnn, 'acc': cnn_acc, 'y_pred': cnn_pred}

    # ---- CNN-LSTM: 基于原始信号 ----
    if 'cnnlstm' in model_types:
        if verbose:
            print("\n--- CNN-LSTM (on raw inertial signals) ---")
        cnnlstm = create_model('cnnlstm', (n_channels, n_timesteps), n_classes,
                               conv_filters=(64, 128, 256),
                               lstm_hidden=128, lstm_layers=2,
                               kernel_size=5, dropout=0.3)
        cnnlstm, hist, le = train_dl_model(
            cnnlstm, X_cnn_train, y_train, model_type='cnnlstm',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        cl_acc, cl_pred = evaluate_dl_model(
            cnnlstm, X_cnn_test, y_test, model_type='cnnlstm',
            label_map=activities,
        )
        results['DL-CNNLSTM'] = {'model': cnnlstm, 'acc': cl_acc, 'y_pred': cl_pred}

    return results, X_cnn_test


def train_dl_wisdm(X_features, y_train, X_test_features, y_test,
                   segments_train, segments_test,
                   model_types=('mlp', 'cnn', 'cnnlstm'),
                   epochs=80, verbose=True):
    """在 WISDM 数据上训练深度学习模型

    Args:
        X_features:      手工特征训练集
        y_train:         训练标签
        X_test_features: 手工特征测试集
        y_test:          测试标签
        segments_train:  训练集原始分段 (M_train, win, 3)
        segments_test:   测试集原始分段 (M_test, win, 3)
        model_types:     模型类型元组
        epochs:          最大训练轮数
        verbose:         是否打印详情

    Returns:
        results: dict {model_name: {'model': nn.Module, 'acc': float, 'y_pred': array}}
    """
    from dl_models import (create_model, train_dl_model, evaluate_dl_model,
                           get_device)

    device = get_device()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Deep Learning on WISDM (device: {device})")
        print(f"{'=' * 60}")

    n_classes = len(np.unique(y_train))
    results = {}

    # ---- 准备 CNN 原始信号数据 ----
    if 'cnn' in model_types or 'cnnlstm' in model_types:
        X_cnn_train = prepare_wisdm_raw_for_cnn(segments_train)
        X_cnn_test = prepare_wisdm_raw_for_cnn(segments_test)
        n_channels, n_timesteps = X_cnn_train.shape[1], X_cnn_train.shape[2]
        if verbose:
            print(f"CNN input shape: ({n_channels} channels × {n_timesteps} timesteps)")
    else:
        X_cnn_test = None

    # ---- MLP ----
    if 'mlp' in model_types:
        if verbose:
            print("\n--- MLP (on enhanced hand-crafted features) ---")
        mlp = create_model('mlp', X_features.shape[1], n_classes,
                           hidden_dims=(256, 128, 64), dropout=0.3)
        mlp, hist, le = train_dl_model(
            mlp, X_features, y_train, model_type='mlp',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        mlp_acc, mlp_pred = evaluate_dl_model(
            mlp, X_test_features, y_test, model_type='mlp', label_map=None,
        )
        results['DL-MLP'] = {'model': mlp, 'acc': mlp_acc, 'y_pred': mlp_pred}

    # ---- CNN1D ----
    if 'cnn' in model_types:
        if verbose:
            print("\n--- CNN1D (on raw inertial signals) ---")
        cnn = create_model('cnn', (n_channels, n_timesteps), n_classes,
                           conv_filters=(64, 128, 256), kernel_size=5, dropout=0.3)
        cnn, hist, le = train_dl_model(
            cnn, X_cnn_train, y_train, model_type='cnn',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        cnn_acc, cnn_pred = evaluate_dl_model(
            cnn, X_cnn_test, y_test, model_type='cnn', label_map=None,
        )
        results['DL-CNN'] = {'model': cnn, 'acc': cnn_acc, 'y_pred': cnn_pred}

    # ---- CNN-LSTM ----
    if 'cnnlstm' in model_types:
        if verbose:
            print("\n--- CNN-LSTM (on raw inertial signals) ---")
        cnnlstm = create_model('cnnlstm', (n_channels, n_timesteps), n_classes,
                               conv_filters=(64, 128, 256),
                               lstm_hidden=128, lstm_layers=2,
                               kernel_size=5, dropout=0.3)
        cnnlstm, hist, le = train_dl_model(
            cnnlstm, X_cnn_train, y_train, model_type='cnnlstm',
            batch_size=64, epochs=epochs, lr=0.001,
            early_stopping_patience=15, verbose=verbose,
        )
        cl_acc, cl_pred = evaluate_dl_model(
            cnnlstm, X_cnn_test, y_test, model_type='cnnlstm', label_map=None,
        )
        results['DL-CNNLSTM'] = {'model': cnnlstm, 'acc': cl_acc, 'y_pred': cl_pred}

    return results


if __name__ == '__main__':
    # 快速验证 WISDM 训练流程
    print("Testing WISDM training pipeline...")
    result = prepare_wisdm()
    print("Done!")
