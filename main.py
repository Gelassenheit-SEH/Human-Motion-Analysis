#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运动数据分析与分类 — 主流程

数据集: UCI HAR (ID 240) & WISDM (ID 507)
流程: 预处理 → 时域+频域特征提取 → MinMax归一化 → KNN/SVM分类 → 评估 → 频谱可视化
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal, fft
import zipfile
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from preprocessing import butter_lowpass_filter
from feature_extraction import extract_all_features
from train import (
    load_har_data, extract_har_features, train_classifiers,
    load_wisdm_raw, prepare_wisdm,
    HAR_FS, HAR_BANDS, HAR_AXIS_NAMES, HAR_WIN,
    WISDM_FS, WISDM_WIN, WISDM_BANDS, WISDM_AXIS_NAMES, WISDM_STEP,
)
from test import evaluate_har, evaluate_wisdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def feature_comparison_har(X_train, y_train, activities, feat_names):
    """打印 HAR 各活动的特征均值对比"""
    print("\nFeature comparison across activities (mean values):")
    check_feats = ['body_acc_x_mean', 'body_acc_x_zero_crossing', 'body_acc_x_rms',
                   'body_acc_x_spectral_centroid', 'body_gyro_x_mean']
    for feat in check_feats:
        if feat in feat_names:
            idx = feat_names.index(feat)
            print(f"  {feat}:")
            for act_id in range(1, 7):
                mask = y_train == act_id
                print(f"    {activities[act_id]:20s}: {np.mean(X_train[mask, idx]):.4f}")


def feature_comparison_wisdm(X, y):
    """打印 WISDM 各活动的特征均值对比"""
    print("\nFeature comparison across activities (mean values):")
    for feat_idx in [0, 2, 3, 6]:  # x_mean, x_zero_crossing, x_rms, x_centroid
        if feat_idx < X.shape[1]:
            print(f"  Feature col {feat_idx}:")
            for act in np.unique(y):
                mask = y == act
                if mask.sum() > 0:
                    print(f"    {act:12s}: {np.mean(X[mask, feat_idx]):.4f}")


TIME_FEATURE_TYPES = ['mean', 'var', 'zero_crossing', 'rms', 'peak', 'peak_to_peak', 'waveform_factor']
TIME_FEATURE_LABELS = ['Mean', 'Variance', 'Zero Crossing', 'RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor']


def plot_time_features(X, y, feature_names, activities, axis_names, output_path, title):
    """时域特征可视化: 7 个子图，每个子图展示一种特征类型在各类活动 / 各轴上的均值"""
    # 建立特征类型 → [(axis_name, col_index)] 的映射
    feat_map = {}
    for ft in TIME_FEATURE_TYPES:
        cols = []
        for axis in axis_names:
            target = f'{axis}_{ft}'
            try:
                cols.append((axis, feature_names.index(target)))
            except ValueError:
                pass
        feat_map[ft] = cols

    if isinstance(activities, dict):
        unique_acts = sorted(activities.keys())
        act_labels = [activities[k] for k in unique_acts]
    else:
        unique_acts = sorted(np.unique(y))
        act_labels = list(unique_acts)

    n_acts = len(unique_acts)
    n_axes = len(axis_names)
    bar_width = 0.8 / n_axes
    x = np.arange(n_acts)
    colors = plt.cm.tab10(np.linspace(0, 1, n_axes))

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()

    for fi, ft in enumerate(TIME_FEATURE_TYPES):
        ax = axes[fi]
        cols = feat_map.get(ft, [])
        for ai, (axis_name, col_idx) in enumerate(cols):
            values = [np.mean(X[y == act, col_idx]) for act in unique_acts]
            offset = (ai - n_axes / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=axis_name,
                   color=colors[ai], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(act_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(TIME_FEATURE_LABELS[fi], fontsize=11)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3, axis='y')

    axes[-1].set_visible(False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def run_har_pipeline():
    """HAR 完整流程: 加载 → 特征提取 → 训练 → 评估"""
    print("=" * 60)
    print("PART 1: UCI HAR DATASET")
    print("=" * 60)

    # 加载
    train_inertial, test_inertial, y_train, y_test, activities = load_har_data(DATA_DIR)
    print(f"Activities: {activities}")
    print(f"Train: {len(y_train)} windows")
    print(f"Test:  {len(y_test)} windows")

    # 特征提取
    print("\nExtracting HAR features...")
    X_train, feat_names = extract_har_features(train_inertial)
    X_test, _ = extract_har_features(test_inertial)
    print(f"Features per window: {len(feat_names)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 特征对比
    feature_comparison_har(X_train, y_train, activities, feat_names)

    # 时域特征可视化
    plot_time_features(
        X_train, y_train, feat_names, activities,
        ['body_acc_x', 'body_acc_y', 'body_acc_z',
         'body_gyro_x', 'body_gyro_y', 'body_gyro_z'],
        os.path.join(OUTPUT_DIR, 'har_time_features.png'),
        'HAR - Time-Domain Features by Activity',
    )

    # 训练
    scaler, knn, svm , rf = train_classifiers(X_train, y_train)
    X_test_s = scaler.transform(X_test)

    # 评估
    evaluate_har(knn, svm, rf, X_test_s, y_test, activities, OUTPUT_DIR)

    print("\nHAR feature names:", feat_names[:10], "...")

    return activities


def run_wisdm_pipeline():
    """WISDM 完整流程: 加载 → 预处理 → 特征提取 → 训练 → 评估"""
    print("\n" + "=" * 60)
    print("PART 2: WISDM DATASET")
    print("=" * 60)

    # 加载原始数据 (用于后续特征对比)
    raw_signals, raw_labels = load_wisdm_raw(DATA_DIR)
    print(f"Raw samples: {len(raw_signals)}")

    # 完整流程 (prepare_wisdm 包含预处理 → 特征提取 → 分割 → 训练)
    scaler, knn, svm, rf , X_test, y_test = prepare_wisdm(DATA_DIR)

    # 特征对比 (使用 scaler 之前的训练数据)
    from train import (
        load_wisdm_raw as _lwr, extract_wisdm_features,
        filter_major_activities, WISDM_MAJOR_ACTS
    )
    from preprocessing import preprocess_wisdm
    raw_sig, raw_lbl = _lwr(DATA_DIR)
    segs, seg_lbls = preprocess_wisdm(raw_sig, raw_lbl, fs=WISDM_FS, win=WISDM_WIN, step=WISDM_STEP)
    X_all, wisdm_feat_names = extract_wisdm_features(segs)
    X_major, y_major = filter_major_activities(X_all, seg_lbls)
    feature_comparison_wisdm(X_major, y_major)

    # 时域特征可视化
    plot_time_features(
        X_major, y_major, wisdm_feat_names, None,
        WISDM_AXIS_NAMES,
        os.path.join(OUTPUT_DIR, 'wisdm_time_features.png'),
        'WISDM - Time-Domain Features by Activity',
    )

    # 评估 (测试集需要归一化)
    X_test_s = scaler.transform(X_test)
    evaluate_wisdm(knn, svm, rf,X_test_s, y_test, OUTPUT_DIR)


def run_har_visualization(activities):
    """HAR 频谱可视化: DFT + STFT + 频谱特征量化"""
    print("\n" + "=" * 60)
    print("PART 3: STFT / DFT FREQUENCY ANALYSIS")
    print("=" * 60)

    zip_path = os.path.join(DATA_DIR, 'har', 'UCI HAR Dataset.zip')
    if not os.path.exists(zip_path):
        zip_path = os.path.join(BASE_DIR, 'har_raw', 'UCI HAR Dataset.zip')

    # --- HAR DFT ---
    print("\n--- HAR DFT & STFT ---")
    with zipfile.ZipFile(zip_path) as zf:
        bax = np.loadtxt(zf.open('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt'))
        yh = np.loadtxt(zf.open('UCI HAR Dataset/train/y_train.txt'), dtype=int)

    sig_samples = {}
    for aid in range(1, 7):
        idx = np.where(yh == aid)[0]
        if len(idx) > 0:
            sig_samples[activities[aid]] = bax[idx[0]]

    # DFT 图
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (an, sig) in enumerate(sig_samples.items()):
        ax = axes[idx]
        n = len(sig)
        dft = fft.fft(sig)
        freq = fft.fftfreq(n, d=1 / HAR_FS)
        pos = freq >= 0
        ax.plot(freq[pos], np.abs(dft[pos]) / n, 'b-', lw=1)
        ax.set_title(an)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([0, HAR_FS / 2])
        ax.grid(alpha=0.3)
    fig.suptitle('HAR - DFT (body_acc_x)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'har_dft.png'), dpi=150)
    plt.close()
    print('[Saved] har_dft.png')

    # --- HAR STFT ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (an, sig) in enumerate(sig_samples.items()):
        ax = axes[idx]
        f, t, Zxx = scipy_signal.stft(sig, fs=HAR_FS, nperseg=32, noverlap=16)
        Zxx_dB = 20 * np.log10(np.abs(Zxx) + 1e-10)
        ax.pcolormesh(t, f, Zxx_dB, shading='gouraud', cmap='viridis')
        ax.set_title(an)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    fig.suptitle('HAR - STFT Spectrograms')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'har_stft.png'), dpi=150)
    plt.close()
    print('[Saved] har_stft.png')

    # --- 频谱特征量化 ---
    print("\n--- Spectral Feature Quantification ---")
    specf = {}
    for an, sig in sig_samples.items():
        n = len(sig)
        window = np.hanning(n)
        sig_w = sig * window
        dft = fft.fft(sig_w)
        freq = fft.fftfreq(n, d=1 / HAR_FS)
        pos = freq >= 0
        fp = freq[pos]
        mp = np.abs(dft[pos])

        sc = np.sum(fp * mp) / (np.sum(mp) + 1e-10)
        se = np.sum(mp ** 2)
        prob = mp / (np.sum(mp) + 1e-10)
        sent = -np.sum(prob * np.log2(prob + 1e-10))

        bands = [(0, 3), (3, 8), (8, 15)]
        energies = {}
        for low, high in bands:
            mask = (fp >= low) & (fp < high)
            energies[f'{low}-{high}Hz'] = np.sum(mp[mask] ** 2)

        specf[an] = {
            'Centroid(Hz)': f'{sc:.2f}',
            'Entropy(bits)': f'{sent:.2f}',
            'Energy': f'{se:.2f}',
            **{k: f'{v:.2f}' for k, v in energies.items()},
        }

    print(pd.DataFrame(specf).T.to_string())

    # --- WISDM DFT ---
    print("\n--- WISDM DFT ---")
    raw_signals, raw_labels = load_wisdm_raw(DATA_DIR)

    wisdm_segments = {}
    for act in ['walking', 'jogging', 'sitting', 'standing', 'stairs']:
        mask = raw_labels == act
        act_data = raw_signals[mask]
        if len(act_data) >= WISDM_WIN:
            wisdm_segments[act] = act_data[:WISDM_WIN].T  # 3 × WISDM_WIN

    fig, axes = plt.subplots(len(wisdm_segments), 2,
                             figsize=(14, 3 * len(wisdm_segments)))
    for idx, (an, seg) in enumerate(wisdm_segments.items()):
        if len(wisdm_segments) == 1:
            axt, axf = axes[0], axes[1]
        else:
            axt, axf = axes[idx, 0], axes[idx, 1]

        t = np.arange(seg.shape[1]) / WISDM_FS
        axt.plot(t, seg[0], label='X', alpha=0.7)
        axt.plot(t, seg[1], label='Y', alpha=0.7)
        axt.plot(t, seg[2], label='Z', alpha=0.7)
        axt.set_title(f'{an} - Time')
        axt.set_xlabel('Time (s)')
        axt.legend()
        axt.grid(alpha=0.3)

        mag = np.sqrt(seg[0] ** 2 + seg[1] ** 2 + seg[2] ** 2)
        dft = fft.fft(mag)
        n = len(mag)
        freq = fft.fftfreq(n, d=1 / WISDM_FS)
        pos = freq >= 0
        axf.plot(freq[pos], np.abs(dft[pos]) / n, 'b-', lw=1)
        axf.set_title(f'{an} - DFT')
        axf.set_xlabel('Frequency (Hz)')
        axf.set_xlim([0, WISDM_FS / 2])
        axf.grid(alpha=0.3)

    fig.suptitle('WISDM - Time Domain & DFT')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wisdm_dft.png'), dpi=150)
    plt.close()
    print('[Saved] wisdm_dft.png')

    # --- WISDM STFT ---
    print("\n--- WISDM STFT ---")
    fig, axes = plt.subplots(len(wisdm_segments), 1,
                             figsize=(14, 3 * len(wisdm_segments)))
    if len(wisdm_segments) == 1:
        axes = [axes]
    for idx, (an, seg) in enumerate(wisdm_segments.items()):
        ax = axes[idx]
        mag = np.sqrt(seg[0] ** 2 + seg[1] ** 2 + seg[2] ** 2)
        f, t, Zxx = scipy_signal.stft(mag, fs=WISDM_FS, nperseg=16, noverlap=8)
        Zxx_dB = 20 * np.log10(np.abs(Zxx) + 1e-10)
        ax.pcolormesh(t, f, Zxx_dB, shading='gouraud', cmap='magma')
        ax.set_title(f'{an} - STFT')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
    fig.suptitle('WISDM - STFT Spectrograms')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wisdm_stft.png'), dpi=150)
    plt.close()
    print('[Saved] wisdm_stft.png')


def print_summary():
    """打印输出文件列表"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:40s} ({size / 1024:.1f} KB)")
    print()
    print("DONE!")


def main():
    activities = run_har_pipeline()
    run_wisdm_pipeline()
    run_har_visualization(activities)
    print_summary()


if __name__ == '__main__':
    main()
