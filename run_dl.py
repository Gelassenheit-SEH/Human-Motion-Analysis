#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅运行深度学习模型（MLP / CNN1D / CNN-LSTM）并输出结果
用法: python run_dl.py [--har] [--wisdm] [--epochs 80] [--all]
     默认 --all，即同时跑 HAR 和 WISDM
"""

import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from train import (
    load_har_data, extract_har_features,
    load_wisdm_raw, extract_wisdm_features,
    HAR_FS, HAR_BANDS, HAR_AXIS_NAMES, HAR_WIN,
    WISDM_FS, WISDM_WIN, WISDM_BANDS, WISDM_AXIS_NAMES, WISDM_STEP,
    WISDM_MAJOR_ACTS, filter_major_activities,
    prepare_har_raw_for_cnn, prepare_wisdm_raw_for_cnn,
)
from preprocessing import preprocess_wisdm
from dl_models import (
    create_model, train_dl_model, evaluate_dl_model, get_device,
)

# ==================== HAR ====================

def run_har_dl(epochs=80, model_types=('mlp', 'cnn', 'cnnlstm', 'resnet')):
    print("=" * 60)
    print("  HAR — Deep Learning")
    print("=" * 60)

    # 加载
    train_inertial, test_inertial, y_train, y_test, activities = load_har_data(DATA_DIR)
    n_classes = len(activities)
    print(f"Activities: {activities}")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # 特征提取（已含姿态角特征）
    X_train, feat_names = extract_har_features(train_inertial)
    X_test, _ = extract_har_features(test_inertial)
    print(f"Features: {X_train.shape[1]} dims (含姿态角特征)")

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # CNN 原始信号
    X_cnn_train = prepare_har_raw_for_cnn(train_inertial)
    X_cnn_test = prepare_har_raw_for_cnn(test_inertial)
    n_channels, n_timesteps = X_cnn_train.shape[1], X_cnn_train.shape[2]
    device = get_device()
    print(f"CNN input: ({n_channels} ch × {n_timesteps} steps) | Device: {device}")

    results = {}
    for mt in model_types:
        print(f"\n{'─' * 40}")
        print(f"  Training {mt.upper()} ...")
        print(f"{'─' * 40}")

        if mt == 'mlp':
            model = create_model('mlp', X_train_s.shape[1], n_classes,
                                 hidden_dims=(256, 128, 64), dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_train_s, y_train, model_type='mlp',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_test_s, y_test, model_type='mlp', label_map=activities,
            )
        elif mt == 'cnn':
            model = create_model('cnn', (n_channels, n_timesteps), n_classes,
                                 conv_filters=(64, 128, 256), kernel_size=5, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='cnn',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='cnn', label_map=activities,
            )
        elif mt == 'cnnlstm':
            model = create_model('cnnlstm', (n_channels, n_timesteps), n_classes,
                                 conv_filters=(64, 128, 256),
                                 lstm_hidden=128, lstm_layers=2,
                                 kernel_size=5, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='cnnlstm',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='cnnlstm', label_map=activities,
            )
        elif mt == 'resnet':
            model = create_model('resnet', (n_channels, n_timesteps), n_classes,
                                 base_ch=64, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='resnet',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='resnet', label_map=activities,
            )
        else:
            continue

        results[mt.upper()] = acc
        print(f"  >>> {mt.upper()} Final Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    return results


# ==================== WISDM ====================

def run_wisdm_dl(epochs=80, model_types=('mlp', 'cnn', 'cnnlstm', 'resnet')):
    print("\n" + "=" * 60)
    print("  WISDM — Deep Learning")
    print("=" * 60)

    # 加载 + 预处理
    raw_signals, raw_labels = load_wisdm_raw(DATA_DIR)
    print(f"Raw samples: {len(raw_signals)}")

    segments, seg_labels = preprocess_wisdm(
        raw_signals, raw_labels, fs=WISDM_FS, win=WISDM_WIN, step=WISDM_STEP,
    )
    print(f"Segments: {len(segments)} | shape: {segments.shape}")

    # 特征提取（已含姿态角特征）
    X_all, feat_names = extract_wisdm_features(segments)
    print(f"Features: {X_all.shape[1]} dims (含姿态角特征)")

    # 筛选主要活动 + 分割
    X_major, y_major = filter_major_activities(X_all, seg_labels)
    major_mask = np.isin(seg_labels, WISDM_MAJOR_ACTS)
    seg_major = segments[major_mask]
    acts = np.unique(y_major)
    print(f"Major activities: {list(acts)}")

    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X_major, y_major, seg_major,
        test_size=0.3, random_state=42, stratify=y_major,
    )
    n_classes = len(acts)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # CNN 原始信号
    X_cnn_train = prepare_wisdm_raw_for_cnn(seg_train)
    X_cnn_test = prepare_wisdm_raw_for_cnn(seg_test)
    n_channels, n_timesteps = X_cnn_train.shape[1], X_cnn_train.shape[2]
    device = get_device()
    print(f"CNN input: ({n_channels} ch × {n_timesteps} steps) | Device: {device}")

    results = {}
    for mt in model_types:
        print(f"\n{'─' * 40}")
        print(f"  Training {mt.upper()} ...")
        print(f"{'─' * 40}")

        if mt == 'mlp':
            model = create_model('mlp', X_train_s.shape[1], n_classes,
                                 hidden_dims=(256, 128, 64), dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_train_s, y_train, model_type='mlp',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_test_s, y_test, model_type='mlp',
            )
        elif mt == 'cnn':
            model = create_model('cnn', (n_channels, n_timesteps), n_classes,
                                 conv_filters=(64, 128, 256), kernel_size=5, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='cnn',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='cnn',
            )
        elif mt == 'cnnlstm':
            model = create_model('cnnlstm', (n_channels, n_timesteps), n_classes,
                                 conv_filters=(64, 128, 256),
                                 lstm_hidden=128, lstm_layers=2,
                                 kernel_size=5, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='cnnlstm',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='cnnlstm',
            )
        elif mt == 'resnet':
            model = create_model('resnet', (n_channels, n_timesteps), n_classes,
                                 base_ch=64, dropout=0.3)
            model, hist, le = train_dl_model(
                model, X_cnn_train, y_train, model_type='resnet',
                batch_size=64, epochs=epochs, lr=0.001,
                early_stopping_patience=15, verbose=True,
            )
            acc, y_pred = evaluate_dl_model(
                model, X_cnn_test, y_test, model_type='resnet',
            )
        else:
            continue

        results[mt.upper()] = acc
        print(f"  >>> {mt.upper()} Final Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    return results


# ==================== 汇总 ====================

def print_summary(har_results, wisdm_results):
    print("\n")
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<14} {'HAR':>10} {'WISDM':>10}")
    print("-" * 36)
    all_models = ['MLP', 'CNN', 'CNNLSTM', 'RESNET']
    for m in all_models:
        h = har_results.get(m, 0) if har_results else 0
        w = wisdm_results.get(m, 0) if wisdm_results else 0
        print(f"{'DL-' + m:<14} {h:>9.2%} {w:>9.2%}" if h or w else f"{'DL-' + m:<14} {'—':>10} {'—':>10}")
    print("-" * 36)
    print()


# ==================== Main ====================

if __name__ == '__main__':
    has_ds_flag = '--har' in sys.argv or '--wisdm' in sys.argv or '--all' in sys.argv
    run_har = '--har' in sys.argv or '--all' in sys.argv or not has_ds_flag
    run_wisdm = '--wisdm' in sys.argv or '--all' in sys.argv or not has_ds_flag

    epochs = 80
    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])

    model_types = ('mlp', 'cnn', 'cnnlstm', 'resnet')
    if '--mlp' in sys.argv:
        model_types = ('mlp',)
    elif '--cnn' in sys.argv:
        model_types = ('cnn',)
    elif '--cnnlstm' in sys.argv:
        model_types = ('cnnlstm',)
    elif '--resnet' in sys.argv:
        model_types = ('resnet',)

    har_results = run_har_dl(epochs, model_types) if run_har else {}
    wisdm_results = run_wisdm_dl(epochs, model_types) if run_wisdm else {}

    print_summary(har_results, wisdm_results)
