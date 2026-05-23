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
import os, sys, pickle, warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== Morandi Color Palettes ==========
MORANDI = [
    '#B3C4D1', '#C4A8B8', '#A8C4B0', '#D1C4A8',
    '#C4B8D1', '#B8D1C4', '#D1B8A8', '#B8A8C4',
    '#A8B8C4', '#C4A8A8',
]

MORANDI_LINE = [
    '#7A9BB5', '#B57A9B', '#9BB57A', '#C49B6C',
    '#8F9BB5', '#B59B6C',
]

# Global style settings
plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

from preprocessing import butter_lowpass_filter
from feature_extraction import extract_all_features
from train import (
    load_har_data, extract_har_features, train_classifiers,
    load_wisdm_raw, prepare_wisdm,
    HAR_FS, HAR_BANDS, HAR_AXIS_NAMES, HAR_WIN,
    WISDM_FS, WISDM_WIN, WISDM_BANDS, WISDM_AXIS_NAMES, WISDM_STEP,
)
from sklearn.metrics import accuracy_score
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


TIME_FEATURE_TYPES = ['mean', 'var', 'zero_crossing']
TIME_FEATURE_LABELS = ['Mean', 'Variance', 'Zero Crossing']

# ========== Classifier Visualization Functions ==========

def plot_classifier_comparison(results_dict, output_path, title='Classifier Accuracy Comparison'):
    """分组柱状图对比各分类器在各数据集上的准确率"""
    datasets = list(results_dict.keys())
    classifiers = list(results_dict[datasets[0]].keys())
    n_datasets = len(datasets)
    n_clfs = len(classifiers)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.7 / n_clfs
    x = np.arange(n_datasets)

    for i, clf in enumerate(classifiers):
        values = [results_dict[ds][clf] for ds in datasets]
        offset = (i - n_clfs / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=clf,
                       color=MORANDI[i % len(MORANDI)], edgecolor='white',
                       linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#3D3D3D')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, facecolor='#F8F6F4', edgecolor='#D5C8C0',
              fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def plot_rf_feature_importance(rf_model, feature_names, output_path, n_top=15,
                                title='Random Forest - Top Feature Importance'):
    """RF 特征重要性水平柱状图"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(n_top)
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, n_top))

    ax.barh(y_pos, importances[indices][::-1], color=colors,
            edgecolor='white', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def plot_permutation_importance(model, X, y, feature_names, output_path, n_top=15,
                                 title='Permutation Feature Importance', clf_label=''):
    """基于置换检验的特征重要性（适用于 SVM / KNN 等无原生 feature_importance 的模型）"""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(n_top)

    # 使用与 confusion matrix 对应的配色
    if clf_label == 'SVM':
        cmap_colors = plt.cm.Purples(np.linspace(0.4, 0.85, n_top))
    else:
        cmap_colors = plt.cm.Oranges(np.linspace(0.4, 0.85, n_top))

    ax.barh(y_pos, importances[indices][::-1], color=cmap_colors,
            edgecolor='white', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Permutation Importance (Δ score)', fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


# ========== 7-Feature Time-Domain Grouped Bar ==========
TIME_FEAT7 = ['mean', 'var', 'zero_crossing', 'rms', 'peak', 'peak_to_peak', 'waveform_factor']
TIME_LABEL7 = ['Mean', 'Variance', 'Zero Crossing', 'RMS', 'Peak', 'Peak-to-Peak', 'Waveform Factor']


def plot_time_features(X, y, feature_names, activities, axis_names, output_path, title):
    """7 个子图，每个子图展示一种特征在各类活动/各轴上的均值"""
    feat_map = {}
    for ft in TIME_FEAT7:
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

    for fi, ft in enumerate(TIME_FEAT7):
        ax = axes[fi]
        cols = feat_map.get(ft, [])
        for ai, (axis_name, col_idx) in enumerate(cols):
            values = [np.mean(X[y == act, col_idx]) for act in unique_acts]
            offset = (ai - n_axes / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=axis_name,
                   color=colors[ai], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(act_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(TIME_LABEL7[fi], fontsize=11)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3, axis='y')

    axes[-1].set_visible(False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def plot_violin_features(X, y, feature_names, activities, axis_names, output_path, title):
    """小提琴图：展示各活动在 Mean / Variance / Zero Crossing 上的分布

    每个子图 = 一种特征，跨指定轴取平均，按活动分组画小提琴图。
    """
    # 对每种特征类型，收集跨轴平均后的每样本值
    feat_data = {ft: [] for ft in TIME_FEATURE_TYPES}
    for ft in TIME_FEATURE_TYPES:
        cols = []
        for axis in axis_names:
            target = f'{axis}_{ft}'
            try:
                cols.append(feature_names.index(target))
            except ValueError:
                pass
        if cols:
            # 跨轴平均
            feat_data[ft] = np.mean(X[:, cols], axis=1)

    if isinstance(activities, dict):
        unique_acts = sorted(activities.keys())
        act_labels = [activities[k] for k in unique_acts]
    else:
        unique_acts = sorted(np.unique(y))
        act_labels = list(unique_acts)

    n_acts = len(unique_acts)
    # 从 experiment_violin.py 借用的鲜亮配色
    act_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7'][:n_acts]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for fi, ft in enumerate(TIME_FEATURE_TYPES):
        ax = axes[fi]
        values = feat_data.get(ft)
        if values is None or len(values) == 0:
            ax.set_title(TIME_FEATURE_LABELS[fi])
            continue

        data = [values[y == act] for act in unique_acts]
        vp = ax.violinplot(data, positions=range(n_acts), showmeans=True,
                           showmedians=False, widths=0.6)

        for body, color in zip(vp['bodies'], act_colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        if vp['cmeans']:
            vp['cmeans'].set_color('darkred')
            vp['cmeans'].set_linewidth(1.5)

        ax.set_xticks(range(n_acts))
        ax.set_xticklabels(act_labels, rotation=25, ha='right', fontsize=8)
        ax.set_title(TIME_FEATURE_LABELS[fi], fontsize=12)
        ax.grid(alpha=0.2, axis='y')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def plot_radar_features(X, y, feature_names, activities, axis_names, output_path, title):
    """雷达图：各活动在过零率与峰值上的特征指纹"""
    # 取过零率 + 峰值 (peak, 而非 peak_to_peak) 的每轴特征
    radar_feats = []
    for ax in axis_names:
        for suffix in ['zero_crossing', 'peak']:
            target = f'{ax}_{suffix}'
            try:
                radar_feats.append((target, feature_names.index(target)))
            except ValueError:
                pass

    if isinstance(activities, dict):
        unique_acts = sorted(activities.keys())
        act_labels = [activities[k] for k in unique_acts]
    else:
        unique_acts = sorted(np.unique(y))
        act_labels = list(unique_acts)

    n_feats = len(radar_feats)
    if n_feats == 0:
        return  # 没有所需特征时跳过

    # 为每个活动计算各特征的均值，并归一化 [0, 1]
    raw = np.zeros((len(unique_acts), n_feats))
    for ai, act in enumerate(unique_acts):
        for fi, (_, col_idx) in enumerate(radar_feats):
            raw[ai, fi] = np.mean(X[y == act, col_idx])
    # 逐列 Min-Max 归一化
    rmin, rmax = raw.min(axis=0), raw.max(axis=0)
    normed = (raw - rmin) / (rmax - rmin + 1e-10)

    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for ai, act in enumerate(unique_acts):
        values = normed[ai].tolist() + normed[ai, :1].tolist()
        ax.plot(angles, values, 'o-', linewidth=2, color=MORANDI[ai], label=act_labels[ai], markersize=4)
        ax.fill(angles, values, alpha=0.08, color=MORANDI[ai])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f[0] for f in radar_feats], fontsize=8)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')


def plot_top_feature_per_classifier(knn_model, svm_model, rf_model, X_test, y_test,
                                     feature_names, output_path, title_prefix=''):
    """柱状图：三种训练方法各自最重要的 feature 及其重要度"""
    from sklearn.inspection import permutation_importance

    # RF: native feature_importances_
    rf_imp = rf_model.feature_importances_
    rf_top_idx = np.argmax(rf_imp)
    rf_top_val = rf_imp[rf_top_idx]

    # SVM / KNN: permutation importance
    svm_result = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    knn_result = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    svm_imp = svm_result.importances_mean
    knn_imp = knn_result.importances_mean

    svm_top_idx = np.argmax(svm_imp)
    knn_top_idx = np.argmax(knn_imp)
    svm_top_val = svm_imp[svm_top_idx]
    knn_top_val = knn_imp[knn_top_idx]

    classifiers = ['KNN', 'SVM', 'RF']
    top_indices = [knn_top_idx, svm_top_idx, rf_top_idx]
    top_values = [knn_top_val, svm_top_val, rf_top_val]
    top_names = [feature_names[i] for i in top_indices]
    colors = [MORANDI[2], MORANDI[1], MORANDI[0]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(classifiers, top_values, color=colors, edgecolor='white', linewidth=0.8, width=0.5)
    for bar, name, val in zip(bars, top_names, top_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{name}\n({val:.4f})', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Importance', fontsize=11)
    ax.set_title(f'{title_prefix} Top Feature per Classifier', fontsize=13, pad=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(top_values) * 1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {output_path}')

def run_har_ablation(X_train, y_train, X_test, y_test, feature_names, OUTPUT_DIR):
    """挑战一：加速度计与陀螺仪消融实验"""
    print("\n" + "=" * 60)
    print("CHALLENGE 1: HAR SENSOR ABLATION STUDY")
    print("=" * 60)

    # 动态匹配特征列索引
    acc_indices = [i for i, name in enumerate(feature_names) if "acc" in name]
    gyro_indices = [i for i, name in enumerate(feature_names) if "gyro" in name]

    configs = {
        "Acc Only": acc_indices,
        "Gyro Only": gyro_indices,
        "Acc + Gyro (Fused)": list(range(len(feature_names))),
    }

    ablation_results = {}
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    for name, indices in configs.items():
        print(f"Training RF for {name}...")
        X_tr = X_train[:, indices]
        X_te = X_test[:, indices]

        # 强制单核 n_jobs=1
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_train)
        acc = accuracy_score(y_test, rf.predict(X_te))
        ablation_results[name] = acc
        print(f"  -> {name} RF Accuracy: {acc:.4f}")

    # 绘制消融实验对比图
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(ablation_results.keys())
    accs = list(ablation_results.values())

    bars = ax.bar(
        names,
        accs,
        color=["#7A9BB5", "#B57A9B", "#9BB57A"],
        width=0.5,
        edgecolor="white",
    )
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Ablation Study: Sensor Fusion (Random Forest)", fontsize=13, pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    out_path = os.path.join(OUTPUT_DIR, "ablation_sensor_fusion.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved Ablation Plot] {out_path}")
    return ablation_results


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

    # 时域特征可视化 — 小提琴图 + 分组柱状图
    acc_axes = ['body_acc_x', 'body_acc_y', 'body_acc_z']
    plot_violin_features(
        X_train, y_train, feat_names, activities, acc_axes,
        os.path.join(OUTPUT_DIR, 'har_violin.png'),
        'HAR - Feature Distributions by Activity',
    )
    plot_time_features(
        X_train, y_train, feat_names, activities,
        ['body_acc_x', 'body_acc_y', 'body_acc_z',
         'body_gyro_x', 'body_gyro_y', 'body_gyro_z'],
        os.path.join(OUTPUT_DIR, 'har_time_features.png'),
        'HAR - Time-Domain Features by Activity',
    )
    plot_radar_features(
        X_train, y_train, feat_names, activities, acc_axes,
        os.path.join(OUTPUT_DIR, 'har_radar.png'),
        'HAR - Feature Radar by Activity',
    )

    # 训练
    scaler, knn, svm, rf = train_classifiers(X_train, y_train)
    X_test_s = scaler.transform(X_test)

    # 评估（获取预测结果）
    (_, knn_pred), (_, svm_pred), (_, rf_pred) = evaluate_har(
        knn, svm, rf, X_test_s, y_test, activities, OUTPUT_DIR
    )

    # RF 特征重要性
    plot_rf_feature_importance(
        rf, feat_names,
        os.path.join(OUTPUT_DIR, 'har_rf_importance.png'),
        n_top=15,
        title='HAR - Random Forest Feature Importance',
    )

    # SVM / KNN 置换特征重要性
    plot_permutation_importance(
        svm, X_test_s, y_test, feat_names,
        os.path.join(OUTPUT_DIR, 'har_svm_importance.png'),
        n_top=15,
        title='HAR - SVM Permutation Feature Importance',
        clf_label='SVM',
    )
    plot_permutation_importance(
        knn, X_test_s, y_test, feat_names,
        os.path.join(OUTPUT_DIR, 'har_knn_importance.png'),
        n_top=15,
        title='HAR - KNN Permutation Feature Importance',
        clf_label='KNN',
    )

    # 各分类器最重要特征对比
    plot_top_feature_per_classifier(
        knn, svm, rf, X_test_s, y_test, feat_names,
        os.path.join(OUTPUT_DIR, 'har_top_feature_per_clf.png'),
        title_prefix='HAR',
    )

    print("\nHAR feature names:", feat_names[:10], "...")

    # 收集准确率（用 evaluate_har 已返回的结果）
    har_acc = {
        'KNN': accuracy_score(y_test, knn_pred),
        'SVM': accuracy_score(y_test, svm_pred),
        'RF':  accuracy_score(y_test, rf_pred),
    }
    return activities, har_acc, rf, feat_names, svm, knn, scaler, X_test_s, y_test, X_train, y_train


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

    # 时域特征可视化 — 小提琴图
    plot_violin_features(
        X_major, y_major, wisdm_feat_names, None, WISDM_AXIS_NAMES,
        os.path.join(OUTPUT_DIR, 'wisdm_violin.png'),
        'WISDM - Feature Distributions by Activity',
    )
    plot_radar_features(
        X_major, y_major, wisdm_feat_names, None, WISDM_AXIS_NAMES,
        os.path.join(OUTPUT_DIR, 'wisdm_radar.png'),
        'WISDM - Feature Radar by Activity',
    )
    plot_time_features(
        X_major, y_major, wisdm_feat_names, None, WISDM_AXIS_NAMES,
        os.path.join(OUTPUT_DIR, 'wisdm_time_features.png'),
        'WISDM - Time-Domain Features by Activity',
    )

    # 评估 (测试集需要归一化)
    X_test_s = scaler.transform(X_test)
    (_, knn_pred), (_, svm_pred), (_, rf_pred) = evaluate_wisdm(
        knn, svm, rf, X_test_s, y_test, OUTPUT_DIR
    )

    # WISDM RF 特征重要性
    plot_rf_feature_importance(
        rf, wisdm_feat_names,
        os.path.join(OUTPUT_DIR, 'wisdm_rf_importance.png'),
        n_top=15,
        title='WISDM - Random Forest Feature Importance',
    )

    # SVM / KNN 置换特征重要性
    plot_permutation_importance(
        svm, X_test_s, y_test, wisdm_feat_names,
        os.path.join(OUTPUT_DIR, 'wisdm_svm_importance.png'),
        n_top=15,
        title='WISDM - SVM Permutation Feature Importance',
        clf_label='SVM',
    )
    plot_permutation_importance(
        knn, X_test_s, y_test, wisdm_feat_names,
        os.path.join(OUTPUT_DIR, 'wisdm_knn_importance.png'),
        n_top=15,
        title='WISDM - KNN Permutation Feature Importance',
        clf_label='KNN',
    )

    # 各分类器最重要特征对比
    plot_top_feature_per_classifier(
        knn, svm, rf, X_test_s, y_test, wisdm_feat_names,
        os.path.join(OUTPUT_DIR, 'wisdm_top_feature_per_clf.png'),
        title_prefix='WISDM',
    )

    # 收集准确率
    wisdm_acc = {
        'KNN': accuracy_score(y_test, knn_pred),
        'SVM': accuracy_score(y_test, svm_pred),
        'RF':  accuracy_score(y_test, rf_pred),
    }
    return wisdm_acc, X_major, y_major, wisdm_feat_names, svm, knn, rf, scaler, X_test_s, y_test


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
        x_fill = freq[pos]
        y_fill = np.abs(dft[pos]) / n
        # 渐变填充：从折线下方到 x 轴
        for frac in np.linspace(0, 1, 30):
            ax.fill_between(x_fill, y_fill * frac, y_fill,
                            alpha=0.012, color=MORANDI_LINE[0])
        ax.plot(x_fill, y_fill, color=MORANDI_LINE[0], linewidth=2.0)
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
        x_fill = freq[pos]
        y_fill = np.abs(dft[pos]) / n
        # 渐变填充
        for frac in np.linspace(0, 1, 30):
            axf.fill_between(x_fill, y_fill * frac, y_fill,
                             alpha=0.012, color=MORANDI_LINE[0])
        axf.plot(x_fill, y_fill, color=MORANDI_LINE[0], linewidth=2.0)
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


CACHE_FILE = os.path.join(OUTPUT_DIR, 'cache', 'pipeline_data.pkl')


def save_pipeline_cache(data):
    """将流水线中间结果保存到缓存，供 --viz-only 使用"""
    os.makedirs(os.path.join(OUTPUT_DIR, 'cache'), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'[Cache saved] {CACHE_FILE}')


def load_pipeline_cache():
    """加载缓存中间结果"""
    with open(CACHE_FILE, 'rb') as f:
        return pickle.load(f)

def run_har_decision_fusion(
    X_train, y_train, X_test, y_test, feature_names, OUTPUT_DIR
):
    """挑战一：加速度计与陀螺仪的决策级融合"""
    print("\n" + "=" * 60)
    print("CHALLENGE 1: HAR DECISION-LEVEL FUSION (LATE FUSION)")
    print("=" * 60)

    acc_indices = [i for i, name in enumerate(feature_names) if "acc" in name]
    gyro_indices = [i for i, name in enumerate(feature_names) if "gyro" in name]

    X_train_acc = X_train[:, acc_indices]
    X_test_acc = X_test[:, acc_indices]
    X_train_gyro = X_train[:, gyro_indices]
    X_test_gyro = X_test[:, gyro_indices]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print("Training independent Random Forest classifiers...")
    rf_acc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_acc.fit(X_train_acc, y_train)

    rf_gyro = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_gyro.fit(X_train_gyro, y_train)

    print("Predicting probabilities on test set...")
    proba_acc = rf_acc.predict_proba(X_test_acc)
    proba_gyro = rf_gyro.predict_proba(X_test_gyro)

    # 决策融合：1:1 软投票
    proba_fused = (proba_acc + proba_gyro) / 2.0
    y_pred_fused = rf_acc.classes_[np.argmax(proba_fused, axis=1)]

    acc_only_score = accuracy_score(y_test, rf_acc.predict(X_test_acc))
    gyro_only_score = accuracy_score(y_test, rf_gyro.predict(X_test_gyro))
    fused_score = accuracy_score(y_test, y_pred_fused)

    print(f"\n--- Decision Fusion Results ---")
    print(f"  - Accelerometer Only RF Acc: {acc_only_score:.4%}")
    print(f"  - Gyroscope Only RF Acc:     {gyro_only_score:.4%}")
    print(f"  - Decision-Level Fused Acc:  {fused_score:.4%}")

    # 绘制决策级融合结果柱状图
    results = {
        "Acc Only": acc_only_score,
        "Gyro Only": gyro_only_score,
        "Fused": fused_score,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        results.keys(),
        results.values(),
        color=["#7A9BB5", "#B57A9B", "#C49B6C"],
        width=0.5,
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Decision Fusion: Late Fusion Accuracy", fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, acc in zip(bars, results.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    out_path = os.path.join(OUTPUT_DIR, "decision_fusion_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved Decision Fusion Plot] {out_path}")

    return fused_score

def run_challenge_enrichment_plots(
    X_train, y_train, X_test, y_test, feature_names, activities, OUTPUT_DIR
):
    """生成挑战一的高级可视化图表 (包含 Gyro, Acc, Fused 三者对比)"""
    print("\n" + "=" * 60)
    print("CHALLENGE 1 ENRICHMENT: Generating Advanced Visualizations...")
    print("=" * 60)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    labels = sorted(list(activities.keys()))
    target_names = [activities[l] for l in labels]

    # 1. 准备三个模型的特征索引与训练预测
    acc_idx = [i for i, n in enumerate(feature_names) if "acc" in n]
    gyro_idx = [i for i, n in enumerate(feature_names) if "gyro" in n]

    print("Fitting RF (Gyro Only)...")
    rf_gyro = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_gyro.fit(X_train[:, gyro_idx], y_train)
    y_pred_gyro = rf_gyro.predict(X_test[:, gyro_idx])

    print("Fitting RF (Acc Only)...")
    rf_acc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_acc.fit(X_train[:, acc_idx], y_train)
    y_pred_acc = rf_acc.predict(X_test[:, acc_idx])

    print("Fitting RF (Fused)...")
    rf_fused = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_fused.fit(X_train, y_train)
    y_pred_fused = rf_fused.predict(X_test)

    # ==========================================
    # 图表 1: 各动作类别准确率提升对比图 (包含 Gyro, Acc, Fused)
    # ==========================================
    cm_gyro = confusion_matrix(y_test, y_pred_gyro, labels=labels)
    cm_acc = confusion_matrix(y_test, y_pred_acc, labels=labels)
    cm_fused = confusion_matrix(y_test, y_pred_fused, labels=labels)

    gyro_class_acc = cm_gyro.diagonal() / cm_gyro.sum(axis=1)
    acc_class_acc = cm_acc.diagonal() / cm_acc.sum(axis=1)
    fused_class_acc = cm_fused.diagonal() / cm_fused.sum(axis=1)

    x = np.arange(len(target_names))
    width = 0.25  # 缩小柱子宽度以容纳三根柱体

    fig, ax = plt.subplots(figsize=(12, 6))  # 稍微加宽画布
    rects1 = ax.bar(
        x - width,
        gyro_class_acc,
        width,
        label="Gyro Only",
        color="#B57A9B",
        edgecolor="white",
    )
    rects2 = ax.bar(
        x, acc_class_acc, width, label="Acc Only", color="#7A9BB5", edgecolor="white"
    )
    rects3 = ax.bar(
        x + width,
        fused_class_acc,
        width,
        label="Acc + Gyro (Fused)",
        color="#9BB57A",
        edgecolor="white",
    )

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        "Per-Class Accuracy: Gyro vs Acc vs Fused (Random Forest)", fontsize=14, pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)  # 提高上限防遮挡
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 添加数值标签 (为了防拥挤，字体稍微调小一点)
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.tight_layout()
    p1_path = os.path.join(OUTPUT_DIR, "enrichment_per_class_acc.png")
    plt.savefig(p1_path, dpi=150)
    plt.close()
    print(f"[Saved] Per-Class Accuracy Plot -> {p1_path}")

    # ==========================================
    # 图表 2: 差值混淆矩阵 (Fused 减去最强基线 Acc Only)
    # 注: 这里保留 Fused - Acc，因为 Acc 是主导特征，我们想看加了 Gyro 后修复了哪些错误
    # ==========================================
    cm_diff = cm_fused - cm_acc

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_diff,
        annot=True,
        fmt="d",
        cmap="RdBu_r",
        center=0,
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={"label": "Performance Change (Δ)"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        "Differential Confusion Matrix (Fused - Acc Only)", fontsize=14, pad=15
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    p2_path = os.path.join(OUTPUT_DIR, "enrichment_diff_cm.png")
    plt.savefig(p2_path, dpi=150)
    plt.close()
    print(f"[Saved] Differential Confusion Matrix -> {p2_path}")

    # ==========================================
    # 图表 3: 融合模型双色特征重要性 TOP 15
    # ==========================================
    importances = rf_fused.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    colors = ["#D489A1" if "gyro" in name else "#658EAD" for name in top_features]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top_features))

    ax.barh(y_pos, top_importances[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title("Top 15 Feature Importances in Fused Model", fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#658EAD", edgecolor="white", label="Accelerometer Feature"),
        Patch(facecolor="#D489A1", edgecolor="white", label="Gyroscope Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    p3_path = os.path.join(OUTPUT_DIR, "enrichment_feature_importance.png")
    plt.savefig(p3_path, dpi=150)
    plt.close()
    print(f"[Saved] Colored Feature Importance -> {p3_path}")

def run_challenge_tradeoff_and_tuning(
    X_train, y_train, X_test, y_test, feature_names, OUTPUT_DIR
):
    """方向一与方向二结合：运行超参数调优曲线与工程推理延迟折中分析"""
    print("\n" + "=" * 60)
    print("CHALLENGE 1: Running Trade-off Analysis & Hyperparameter Tuning...")
    print("=" * 60)

    import time

    acc_idx = [i for i, n in enumerate(feature_names) if "acc" in n]
    gyro_idx = [i for i, n in enumerate(feature_names) if "gyro" in n]

    # --------------------------------------------------------
    # 【方向二】超参数演进：测试不同树的数量对全面融合模型的影响
    # --------------------------------------------------------
    n_estimators_list = [10, 30, 50, 100, 150, 200]
    tuning_accuracies = []
    tuning_times = []

    print("1. Evaluatng n_estimators tuning for Fused Model...")
    for n_est in n_estimators_list:
        t0 = time.time()
        # 建立不同规模的随机森林
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        train_time = time.time() - t0

        acc = accuracy_score(y_test, rf.predict(X_test))
        tuning_accuracies.append(acc)
        tuning_times.append(train_time)
        print(
            f"  - Trees: {n_est:3d} | Test Acc: {acc:.4%} | Train Time: {train_time:.3f}s"
        )

    # 绘制图 1：超参数调优双 Y 轴图（准确率 vs 训练时间）
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        n_estimators_list,
        tuning_accuracies,
        marker="o",
        color="#9BB57A",
        linewidth=2,
        label="Test Accuracy",
    )
    ax1.set_xlabel("Number of Trees (n_estimators)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(
        "Hyperparameter Tuning: Accuracy vs Training Cost", fontsize=13, pad=15
    )
    ax1.grid(axis="both", alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(
        n_estimators_list,
        tuning_times,
        marker="s",
        color="#C49B6C",
        linestyle="--",
        linewidth=1.5,
        label="Training Time",
    )
    ax2.set_ylabel("Training Time (seconds)", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    p1_path = os.path.join(OUTPUT_DIR, "challenge_tuning_curve.png")
    plt.tight_layout()
    plt.savefig(p1_path, dpi=150)
    plt.close()
    print(f"[Saved] Tuning Curve Plot -> {p1_path}")

    # --------------------------------------------------------
    # 【方向一】工程落地折中：固定 100 棵树，对比三种配置的单样本预测延迟
    # --------------------------------------------------------
    configs = {
        "Gyro Only": gyro_idx,
        "Acc Only": acc_idx,
        "Acc + Gyro (Fused)": list(range(len(feature_names))),
    }

    tradeoff_accs = []
    inference_delays = []  # 单位：微秒 (μs)

    print(
        "\n2. Evaluating Single-Sample Inference Latency Trade-off (n_estimators=100)..."
    )
    for name, idx in configs.items():
        X_tr = X_train[:, idx]
        X_te = X_test[:, idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_train)

        # 循环多次推理以精准测量耗时
        t0 = time.time()
        for _ in range(5):
            _ = rf.predict(X_te)
        total_time = (time.time() - t0) / 5.0

        # 计算单一样本的预测延迟（微秒）
        latency_per_sample = (total_time / len(X_te)) * 1e6
        acc = accuracy_score(y_test, rf.predict(X_te))

        tradeoff_accs.append(acc)
        inference_delays.append(latency_per_sample)
        print(
            f"  - {name:18s} | Acc: {acc:.4%} | Latency per Sample: {latency_per_sample:.2f} μs"
        )

    # 绘制图 2：工程折中双轴图（准确率柱状图 + 延迟折线图）
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x_labels = list(configs.keys())
    x_pos = np.arange(len(x_labels))

    bars = ax1.bar(
        x_pos - 0.15,
        tradeoff_accs,
        width=0.3,
        color="#7A9BB5",
        label="Accuracy",
        edgecolor="white",
    )
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_title(
        "Engineering Trade-off: Accuracy vs Inference Latency", fontsize=13, pad=15
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x_pos + 0.15,
        inference_delays,
        marker="D",
        color="#D489A1",
        linewidth=2,
        markersize=8,
        label="Latency (μs)",
    )
    ax2.set_ylabel("Inference Latency per Sample (μs)", fontsize=11)
    ax2.set_ylim(0, max(inference_delays) * 1.3)

    for i, val in enumerate(inference_delays):
        ax2.text(
            i + 0.15,
            val + (max(inference_delays) * 0.02),
            f"{val:.2f} μs",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#B57A9B",
            fontweight="bold",
        )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    p2_path = os.path.join(OUTPUT_DIR, "challenge_tradeoff_latency.png")
    plt.tight_layout()
    plt.savefig(p2_path, dpi=150)
    plt.close()
    print(f"[Saved] Engineering Trade-off Plot -> {p2_path}")

def run_viz_only(data):
    """仅运行画图部分（加载缓存后调用）"""
    har = data['har']
    wis = data['wisdm']

    # HAR 时域特征图
    plot_violin_features(
        har['X_train'], har['y_train'], har['feat_names'], har['activities'],
        ['body_acc_x', 'body_acc_y', 'body_acc_z'],
        os.path.join(OUTPUT_DIR, 'har_violin.png'),
        'HAR - Feature Distributions by Activity',
    )
    plot_time_features(
        har['X_train'], har['y_train'], har['feat_names'], har['activities'],
        ['body_acc_x', 'body_acc_y', 'body_acc_z',
         'body_gyro_x', 'body_gyro_y', 'body_gyro_z'],
        os.path.join(OUTPUT_DIR, 'har_time_features.png'),
        'HAR - Time-Domain Features by Activity',
    )
    plot_radar_features(
        har['X_train'], har['y_train'], har['feat_names'], har['activities'],
        ['body_acc_x', 'body_acc_y', 'body_acc_z'],
        os.path.join(OUTPUT_DIR, 'har_radar.png'),
        'HAR - Feature Radar by Activity',
    )
    plot_top_feature_per_classifier(
        har['knn'], har['svm'], har['rf'], har['X_test_s'], har['y_test'], har['feat_names'],
        os.path.join(OUTPUT_DIR, 'har_top_feature_per_clf.png'),
        title_prefix='HAR',
    )
    # HAR 混淆矩阵
    from test import evaluate_har as _eh
    _eh(har['knn'], har['svm'], har['rf'], har['X_test_s'], har['y_test'],
        har['activities'], OUTPUT_DIR)
    # HAR 特征重要性
    plot_rf_feature_importance(har['rf'], har['feat_names'],
        os.path.join(OUTPUT_DIR, 'har_rf_importance.png'), n_top=15,
        title='HAR - Random Forest Feature Importance')
    for model, label in [(har['svm'], 'SVM'), (har['knn'], 'KNN')]:
        plot_permutation_importance(model, har['X_test_s'], har['y_test'], har['feat_names'],
            os.path.join(OUTPUT_DIR, f'har_{label.lower()}_importance.png'), n_top=15,
            title=f'HAR - {label} Permutation Feature Importance', clf_label=label)

    # WISDM 时域特征图
    plot_violin_features(
        wis['X_major'], wis['y_major'], wis['feat_names'], None, ['x', 'y', 'z'],
        os.path.join(OUTPUT_DIR, 'wisdm_violin.png'),
        'WISDM - Feature Distributions by Activity',
    )
    plot_time_features(
        wis['X_major'], wis['y_major'], wis['feat_names'], None, ['x', 'y', 'z'],
        os.path.join(OUTPUT_DIR, 'wisdm_time_features.png'),
        'WISDM - Time-Domain Features by Activity',
    )
    plot_radar_features(
        wis['X_major'], wis['y_major'], wis['feat_names'], None, ['x', 'y', 'z'],
        os.path.join(OUTPUT_DIR, 'wisdm_radar.png'),
        'WISDM - Feature Radar by Activity',
    )
    plot_top_feature_per_classifier(
        wis['knn'], wis['svm'], wis['rf'], wis['X_test_s'], wis['y_test'], wis['feat_names'],
        os.path.join(OUTPUT_DIR, 'wisdm_top_feature_per_clf.png'),
        title_prefix='WISDM',
    )
    # WISDM 混淆矩阵
    from test import evaluate_wisdm as _ew
    _ew(wis['knn'], wis['svm'], wis['rf'], wis['X_test_s'], wis['y_test'], OUTPUT_DIR)
    # WISDM 特征重要性
    plot_rf_feature_importance(wis['rf'], wis['feat_names'],
        os.path.join(OUTPUT_DIR, 'wisdm_rf_importance.png'), n_top=15,
        title='WISDM - Random Forest Feature Importance')
    for model, label in [(wis['svm'], 'SVM'), (wis['knn'], 'KNN')]:
        plot_permutation_importance(model, wis['X_test_s'], wis['y_test'], wis['feat_names'],
            os.path.join(OUTPUT_DIR, f'wisdm_{label.lower()}_importance.png'), n_top=15,
            title=f'WISDM - {label} Permutation Feature Importance', clf_label=label)

    # HAR 频谱可视化
    run_har_visualization(har['activities'])

    # 分类器对比
    plot_classifier_comparison(
        {'HAR': har['acc'], 'WISDM': wis['acc']},
        os.path.join(OUTPUT_DIR, 'classifier_comparison.png'),
        title='Classifier Accuracy: HAR vs WISDM',
    )

    print_summary()


# ==========================================
# 主入口 (全自动缓存版 - 已修复特征归一化 Bug)
# ==========================================
def main():
    import sys  # 确保顶部导入了 sys

    CACHE_FILE = os.path.join(OUTPUT_DIR,'cache', 'pipeline_data.pkl')  # 确保文件名与你代码中一致

    if os.path.exists(CACHE_FILE):
        print("\n" + "=" * 60)
        print("[⚡ 发现缓存] 直接读取基础特征数据，跳过漫长的预处理与基础训练！")
        print("=" * 60)

        data = load_pipeline_cache()

        X_train_har = data["har"]["X_train"]
        y_train_har = data["har"]["y_train"]
        X_test_s_har = data["har"]["X_test_s"]
        y_test_har = data["har"]["y_test"]
        feat_names = data["har"]["feat_names"]
        scaler_har = data["har"]["scaler"]
        activities = data["har"]["activities"]  # <--- 提取行为标签字典

        X_train_s_har = scaler_har.transform(X_train_har)

        # 运行之前的挑战一代码
        run_har_ablation(
            X_train_s_har, y_train_har, X_test_s_har, y_test_har, feat_names, OUTPUT_DIR
        )
        run_har_decision_fusion(
            X_train_s_har, y_train_har, X_test_s_har, y_test_har, feat_names, OUTPUT_DIR
        )


        run_challenge_enrichment_plots(
            X_train_s_har,
            y_train_har,
            X_test_s_har,
            y_test_har,
            feat_names,
            activities,
            OUTPUT_DIR,
        )

        run_challenge_tradeoff_and_tuning(
            X_train_s_har, y_train_har, X_test_s_har, y_test_har, feat_names, OUTPUT_DIR
        )

        print("\n[✔ 运行完毕] 所有图表已更新至 output/ 文件夹。")
        return

    # 初次运行分支
    print("\n[未发现缓存] 初次运行，执行完整预处理与训练流程...")

    (
        activities,
        har_acc,
        rf_model,
        feat_names,
        svm_model,
        knn_model,
        scaler_har,
        X_test_s_har,
        y_test_har,
        X_train_har,
        y_train_har,
    ) = run_har_pipeline()

    (
        wisdm_acc,
        X_major,
        y_major,
        wisdm_feat_names,
        svm_w,
        knn_w,
        rf_w,
        scaler_w,
        X_test_s_w,
        y_test_w,
    ) = run_wisdm_pipeline()

    run_har_visualization(activities)

    # 【核心修复】：初次运行时，同样要归一化训练集
    X_train_s_har = scaler_har.transform(X_train_har)

    ablation_res = run_har_ablation(
        X_train_s_har, y_train_har, X_test_s_har, y_test_har, feat_names, OUTPUT_DIR
    )
    fusion_res = run_har_decision_fusion(
        X_train_s_har, y_train_har, X_test_s_har, y_test_har, feat_names, OUTPUT_DIR
    )

    plot_classifier_comparison(
        {"HAR": har_acc, "WISDM": wisdm_acc},
        os.path.join(OUTPUT_DIR, "classifier_comparison.png"),
        title="Classifier Accuracy: HAR vs WISDM",
    )

    cache_data = {
        "har": {
            "X_train": X_train_har,
            "y_train": y_train_har,
            "feat_names": feat_names,
            "activities": activities,
            "X_test_s": X_test_s_har,
            "y_test": y_test_har,
            "scaler": scaler_har,
            "knn": knn_model,
            "svm": svm_model,
            "rf": rf_model,
            "acc": har_acc,
        },
        "wisdm": {
            "X_major": X_major,
            "y_major": y_major,
            "feat_names": wisdm_feat_names,
            "X_test_s": X_test_s_w,
            "y_test": y_test_w,
            "scaler": scaler_w,
            "knn": knn_w,
            "svm": svm_w,
            "rf": rf_w,
            "acc": wisdm_acc,
        },
        "challenge": {"ablation": ablation_res, "fusion": fusion_res},
    }
    save_pipeline_cache(cache_data)
    print_summary()


if __name__ == "__main__":
    main()
