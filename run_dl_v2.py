#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_dl_v2.py — 改进版深度学习流水线

改进点:
  - 加载 total_acc（含重力倾角信息），CNN 输入从 6 通道 → 9 通道
  - 训练过程可视化（loss/acc 曲线、混淆矩阵、类别 F1 对比）
  - 评测结果自动保存为文档到 output/v2/

用法: python run_dl_v2.py [--har] [--wisdm] [--epochs 100] [--model resnet]
"""

import numpy as np
import os, sys, warnings, zipfile
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- 全局 matplotlib 样式 ----
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from preprocessing import preprocess_wisdm, butter_lowpass_filter
from feature_extraction import extract_all_features, extract_time_features, extract_freq_features
from dl_models import (create_model, train_dl_model, evaluate_dl_model, get_device,
                       augment_signal)

# ============================================================
#  1. 自定义 HAR 数据加载 —— 包含 total_acc（重力倾角）
# ============================================================

HAR_FS = 50.0
HAR_WIN = 128
HAR_BANDS = [(0, 3), (3, 8), (8, 15)]

# v1: body_acc(3) + body_gyro(3) = 6 通道
# v2: body_acc(3) + body_gyro(3) + total_acc(3) = 9 通道
HAR_AXIS_NAMES_V2 = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
]

# 仅前 3 轴用于姿态角特征（用 total_acc 替代 body_acc）
HAR_ORIENTATION_AXES = ['total_acc_x', 'total_acc_y', 'total_acc_z']

# WISDM 参数
WISDM_FS = 20.0
WISDM_WIN = 40
WISDM_STEP = 20
WISDM_BANDS = [(0, 2), (2, 6), (6, 10)]
WISDM_AXIS_NAMES = ['x', 'y', 'z']
WISDM_MAJOR_ACTS = ['walking', 'jogging', 'sitting', 'standing', 'stairs']

WISDM_ACT_MAP = {
    'A': 'walking', 'B': 'jogging', 'C': 'stairs',
    'D': 'sitting', 'E': 'standing', 'F': 'typing',
    'G': 'teeth', 'H': 'soup', 'I': 'chips',
    'J': 'pasta', 'K': 'drinking', 'L': 'sandwich',
    'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
    'Q': 'writing', 'R': 'clapping', 'S': 'folding',
}


def load_har_data_v2(data_dir='./dataset'):
    """加载 UCI HAR 数据集（含 total_acc 重力信号）

    v1 只加载 body_acc + body_gyro（6 通道），丢失了重力倾角信息。
    v2 额外加载 total_acc（3 通道），共计 9 通道。

    Returns:
        train_inertial: dict {axis_name: ndarray (n_windows × 128)}
        test_inertial:  dict
        y_train, y_test: 1D int arrays
        activities: dict {int: name}
    """
    zip_path = os.path.join(data_dir, 'har', 'UCI HAR Dataset.zip')
    if not os.path.exists(zip_path):
        zip_path = os.path.join(os.path.dirname(data_dir), 'har_raw', 'UCI HAR Dataset.zip')

    activities = {}
    data = {}

    with zipfile.ZipFile(zip_path) as zf:
        for line in zf.read('UCI HAR Dataset/activity_labels.txt').decode().strip().split('\n'):
            idx, name = line.strip().split()
            activities[int(idx)] = name

        for subset in ['train', 'test']:
            y = np.loadtxt(zf.open(f'UCI HAR Dataset/{subset}/y_{subset}.txt'), dtype=int)

            inertial = {}
            # body_acc + body_gyro
            for st in ['body_acc', 'body_gyro']:
                for ax in ['x', 'y', 'z']:
                    key = f'{st}_{ax}'
                    inertial[key] = np.loadtxt(
                        zf.open(f'UCI HAR Dataset/{subset}/Inertial Signals/{key}_{subset}.txt')
                    )
            # ★ total_acc（含重力分量，用于区分坐/站/躺）
            for ax in ['x', 'y', 'z']:
                key = f'total_acc_{ax}'
                inertial[key] = np.loadtxt(
                    zf.open(f'UCI HAR Dataset/{subset}/Inertial Signals/total_acc_{ax}_{subset}.txt')
                )

            data[subset] = {'y': y, 'inertial': inertial}

    return (data['train']['inertial'], data['test']['inertial'],
            data['train']['y'], data['test']['y'], activities)


def prepare_har_raw_for_cnn_v2(inertial_dict):
    """HAR → CNN 格式 (N, 9, 128): body_acc(3) + body_gyro(3) + total_acc(3)"""
    channels = [inertial_dict[ax] for ax in HAR_AXIS_NAMES_V2]
    return np.stack(channels, axis=1).astype(np.float32)


def extract_har_features_v2(inertial_dict):
    """HAR 特征提取 v2: 用 total_acc 做姿态角特征，全部 9 轴做时/频域特征"""
    n = len(inertial_dict[HAR_AXIS_NAMES_V2[0]])
    feat_list, name_list = [], None
    for i in range(n):
        data = {ax: inertial_dict[ax][i] for ax in HAR_AXIS_NAMES_V2}
        vec, names = extract_all_features(data, HAR_AXIS_NAMES_V2, HAR_FS, HAR_BANDS,
                                          add_orientation=True)
        feat_list.append(vec)
        if name_list is None:
            name_list = names
    return np.array(feat_list), name_list


# ---- WISDM 工具 ----

def load_wisdm_raw(data_dir='./dataset'):
    """加载 WISDM 原始加速度数据"""
    zip_path = os.path.join(data_dir, 'wisdm', 'wisdm-dataset.zip')
    raw_signals, raw_labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        files = sorted([f for f in zf.namelist()
                        if f.startswith('wisdm-dataset/raw/phone/accel/') and f.endswith('.txt')])[:20]
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


def extract_wisdm_features_v2(segments):
    """WISDM 特征提取"""
    X_list, name_list = [], None
    for i in range(len(segments)):
        data_dict = {ax: segments[i, :, j] for j, ax in enumerate(WISDM_AXIS_NAMES)}
        vec, names = extract_all_features(data_dict, WISDM_AXIS_NAMES, WISDM_FS, WISDM_BANDS,
                                          add_orientation=True)
        X_list.append(vec)
        if name_list is None:
            name_list = names
    return np.array(X_list), name_list


def prepare_wisdm_raw_for_cnn_v2(segments):
    """WISDM → CNN 格式 (M, 3, 40)"""
    return np.transpose(segments, (0, 2, 1)).astype(np.float32)


# ============================================================
#  2. 可视化函数
# ============================================================

MORANDI = ['#B3C4D1', '#C4A8B8', '#A8C4B0', '#D1C4A8',
           '#C4B8D1', '#B8D1C4', '#D1B8A8', '#B8A8C4', '#A8B8C4', '#C4A8A8']


def plot_training_curves(history, model_name, output_dir):
    """绘制训练/验证 loss 和 accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], color=MORANDI[0], linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=MORANDI[1], linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(alpha=0.3)

    # Accuracy 曲线
    ax2.plot(epochs, history['val_acc'], color=MORANDI[2], linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Validation Accuracy')
    ax2.grid(alpha=0.3)
    # 标注最佳 epoch
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.annotate(f'Best: {best_acc:.4f} @ epoch {best_epoch}',
                 xy=(best_epoch, best_acc), xytext=(best_epoch + 2, best_acc - 0.02),
                 fontsize=8, color='red')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'training_curves_{model_name.lower().replace("-", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] {filepath}')


def plot_confusion_matrix_v2(y_true, y_pred, class_names, title, filepath):
    """绘制混淆矩阵（带归一化标注）"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_norm[i, j] > 0.55 else '#3D3D3D'
            text = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})' if cm[i, j] > 0 else '0'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, pad=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] {filepath}')


def plot_per_class_f1(results_dict, dataset_name, output_dir):
    """横向柱状图：各模型在每个类别上的 F1 分数"""
    models = list(results_dict.keys())
    if not models:
        return

    # 收集所有类别名
    all_classes = set()
    for m in models:
        all_classes.update(results_dict[m]['per_class_f1'].keys())
    all_classes = sorted(all_classes)

    n_models = len(models)
    n_classes = len(all_classes)
    bar_width = 0.8 / n_models
    y = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(10, max(5, n_classes * 0.8)))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for mi, m in enumerate(models):
        values = [results_dict[m]['per_class_f1'].get(c, 0) for c in all_classes]
        offset = (mi - n_models / 2 + 0.5) * bar_width
        bars = ax.barh(y + offset, values, bar_width, label=m, color=colors[mi], alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', va='center', fontsize=7, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(all_classes, fontsize=9)
    ax.set_xlabel('F1 Score')
    ax.set_title(f'{dataset_name} — Per-Class F1 Score', pad=12)
    ax.set_xlim(0, 1.15)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{dataset_name.lower()}_per_class_f1.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] {filepath}')


# ============================================================
#  3. 训练管道
# ============================================================

def train_and_eval_dl(model_type, X_train_mlp, y_train, X_test_mlp, y_test,
                      X_train_cnn, X_test_cnn, class_names, output_dir, epochs=100,
                      dataset_prefix='har'):
    """训练单个 DL 模型并返回完整评估结果

    Args:
        dataset_prefix: 文件名前缀 ('har' / 'wisdm')，防止两个数据集的图互相覆盖

    Returns:
        dict: {'acc': float, 'y_pred': array, 'y_true': array,
               'per_class_f1': dict, 'history': dict, 'class_names': list}
    """
    n_classes = len(class_names)
    device = get_device()

    # 确定输入形状
    if model_type == 'mlp':
        input_spec = X_train_mlp.shape[1]
    else:
        n_ch, n_ts = X_train_cnn.shape[1], X_train_cnn.shape[2]
        input_spec = (n_ch, n_ts)

    # 创建模型
    model = create_model(model_type, input_spec, n_classes,
                         hidden_dims=(256, 128, 64), dropout=0.3,
                         base_ch=64)

    # 训练
    train_X = X_train_mlp if model_type == 'mlp' else X_train_cnn
    model, history, le = train_dl_model(
        model, train_X, y_train, model_type=model_type,
        batch_size=64, epochs=epochs, lr=0.001,
        early_stopping_patience=20, verbose=True,
        use_focal=True, focal_gamma=2.0, use_augment=(model_type != 'mlp'),
    )

    # 绘制训练曲线
    plot_training_curves(history, f'{dataset_prefix}_DL-{model_type.upper()}', output_dir)

    # 评估
    test_X = X_test_mlp if model_type == 'mlp' else X_test_cnn

    # 标签处理
    if isinstance(y_test[0], str):
        le_eval = LabelEncoder()
        y_enc = le_eval.fit_transform(y_test)
    else:
        y_enc = np.asarray(y_test, dtype=np.int64)
        if y_enc.min() > 0:
            y_enc = y_enc - y_enc.min()

    from dl_models import predict_dl
    y_pred = predict_dl(model, test_X, model_type)

    acc = accuracy_score(y_enc, y_pred)

    # 每类 F1
    per_class_f1 = {}
    for ci, cname in enumerate(class_names):
        f1 = f1_score(y_enc == ci, y_pred == ci, zero_division=0)
        per_class_f1[cname] = f1

    # 混淆矩阵
    plot_confusion_matrix_v2(y_enc, y_pred, class_names,
                             f'{dataset_prefix.upper()} — {model_type.upper()} Confusion Matrix',
                             os.path.join(output_dir, f'{dataset_prefix}_cm_{model_type}.png'))

    # 分类报告
    print(f"\n{'=' * 60}")
    print(f"  DL-{model_type.upper()}  —  Classification Report")
    print(f"{'=' * 60}")
    label_map = {i: cname for i, cname in enumerate(class_names)}
    y_true_str = [label_map[l] for l in y_enc]
    y_pred_str = [label_map[p] for p in y_pred]
    report = classification_report(y_true_str, y_pred_str, zero_division=0)
    print(report)
    print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    return {
        'acc': acc,
        'y_pred': y_pred,
        'y_true': y_enc,
        'per_class_f1': per_class_f1,
        'history': history,
        'class_names': class_names,
        'classification_report': report,
    }


# ============================================================
#  4. 数据集流水线
# ============================================================

def run_har_v2(epochs=100, model_types=('mlp', 'cnn', 'cnnlstm', 'resnet')):
    ds_prefix = 'har'
    print("\n" + "=" * 70)
    print("  HAR v2 — total_acc 增强 (9 通道) + 重力倾角特征")
    print("=" * 70)

    # 加载（含 total_acc）
    train_inertial, test_inertial, y_train, y_test, activities = load_har_data_v2(DATA_DIR)
    n_classes = len(activities)
    class_names = [activities[i] for i in sorted(activities.keys())]
    print(f"Activities: {activities}")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    # CNN 数据（9 通道）
    X_cnn_train = prepare_har_raw_for_cnn_v2(train_inertial)
    X_cnn_test = prepare_har_raw_for_cnn_v2(test_inertial)
    print(f"CNN input: {X_cnn_train.shape[1]} channels × {X_cnn_train.shape[2]} timesteps")

    # 手工特征（含 total_acc 姿态角）
    X_train_feat, feat_names = extract_har_features_v2(train_inertial)
    X_test_feat, _ = extract_har_features_v2(test_inertial)
    print(f"Features: {X_train_feat.shape[1]} dims")

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train_feat)
    X_test_s = scaler.transform(X_test_feat)

    # 逐个模型训练
    results = {}
    for mt in model_types:
        print(f"\n{'─' * 50}")
        print(f"  Training DL-{mt.upper()} on HAR ...")
        print(f"{'─' * 50}")
        res = train_and_eval_dl(mt, X_train_s, y_train, X_test_s, y_test,
                                X_cnn_train, X_cnn_test, class_names,
                                OUTPUT_DIR, epochs, dataset_prefix=ds_prefix)
        results[f'DL-{mt.upper()}'] = res

    # 每类 F1 对比图
    plot_per_class_f1(results, 'HAR', OUTPUT_DIR)

    return results, class_names


def run_wisdm_v2(epochs=100, model_types=('mlp', 'cnn', 'cnnlstm', 'resnet')):
    ds_prefix = 'wisdm'
    print("\n" + "=" * 70)
    print("  WISDM v2")
    print("=" * 70)

    # 加载 + 预处理
    raw_signals, raw_labels = load_wisdm_raw(DATA_DIR)
    print(f"Raw samples: {len(raw_signals)}")

    segments, seg_labels = preprocess_wisdm(
        raw_signals, raw_labels, fs=WISDM_FS, win=WISDM_WIN, step=WISDM_STEP)
    print(f"Segments: {len(segments)}")

    # 特征提取
    X_all, feat_names = extract_wisdm_features_v2(segments)
    print(f"Features: {X_all.shape[1]} dims")

    # 筛选主要活动 + 分割
    major_mask = np.isin(seg_labels, WISDM_MAJOR_ACTS)
    X_major = X_all[major_mask]
    y_major = seg_labels[major_mask]
    seg_major = segments[major_mask]
    class_names = sorted(WISDM_MAJOR_ACTS)
    print(f"Major activities: {class_names}")

    X_train_f, X_test_f, y_train, y_test, seg_train, seg_test = train_test_split(
        X_major, y_major, seg_major,
        test_size=0.3, random_state=42, stratify=y_major,
    )
    print(f"Train: {len(X_train_f)} | Test: {len(X_test_f)}")

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    # CNN 数据
    X_cnn_train = prepare_wisdm_raw_for_cnn_v2(seg_train)
    X_cnn_test = prepare_wisdm_raw_for_cnn_v2(seg_test)
    print(f"CNN input: {X_cnn_train.shape[1]} channels × {X_cnn_train.shape[2]} timesteps")

    # 逐个模型训练
    results = {}
    for mt in model_types:
        print(f"\n{'─' * 50}")
        print(f"  Training DL-{mt.upper()} on WISDM ...")
        print(f"{'─' * 50}")
        res = train_and_eval_dl(mt, X_train_s, y_train, X_test_s, y_test,
                                X_cnn_train, X_cnn_test, class_names,
                                OUTPUT_DIR, epochs, dataset_prefix=ds_prefix)
        results[f'DL-{mt.upper()}'] = res

    # 每类 F1 对比图
    plot_per_class_f1(results, 'WISDM', OUTPUT_DIR)

    return results, class_names


# ============================================================
#  5. 结果文档自动保存
# ============================================================

def save_results_document(har_results, wisdm_results, output_dir, epochs,
                          terminal_summary=''):
    """将所有评测结果保存为 Markdown 文档 + 纯文本终端日志"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ---- 纯文本日志（终端输出） ----
    txt_path = os.path.join(output_dir, 'terminal_output.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"HAR Deep Learning Results (v2)\n")
        f.write(f"Generated: {now} | Epochs: {epochs} | Device: {get_device()}\n")
        f.write(terminal_summary)
    print(f"  [Saved] {txt_path}")

    # ---- Markdown 报告 ----
    md_path = os.path.join(output_dir, 'results_summary.md')
    lines = []
    lines.append(f"# HAR Deep Learning Results (v2 — total_acc enhanced)")
    lines.append(f"")
    lines.append(f"**Generated**: {now}")
    lines.append(f"**Epochs**: {epochs}")
    lines.append(f"**Device**: {get_device()}")
    lines.append(f"")
    lines.append(f"## Key Improvements over v1")
    lines.append(f"")
    lines.append(f"- CNN input: 6 channels → **9 channels** (added `total_acc_x/y/z` with gravity)")
    lines.append(f"- Orientation features computed from `total_acc` (gravity-inclusive) instead of `body_acc` (gravity-removed)")
    lines.append(f"- Focal Loss + class weighting for imbalanced / hard samples")
    lines.append(f"- Cosine annealing with warmup")
    lines.append(f"- Signal augmentation for CNN models")
    lines.append(f"")

    for ds_name, results in [('HAR', har_results), ('WISDM', wisdm_results)]:
        if not results:
            continue
        lines.append(f"## {ds_name} Dataset")
        lines.append(f"")

        # 准确率汇总表
        lines.append(f"| Model | Accuracy |")
        lines.append(f"|-------|----------|")
        for model_name, res in results.items():
            lines.append(f"| {model_name} | {res['acc']:.4f} ({res['acc'] * 100:.2f}%) |")
        lines.append(f"")

        # 最佳模型
        best_model = max(results, key=lambda k: results[k]['acc'])
        lines.append(f"**Best model**: {best_model} ({results[best_model]['acc']:.4f})")
        lines.append(f"")

        # 每类 F1
        class_names = list(results.values())[0]['class_names']
        lines.append(f"### Per-Class F1 Score")
        lines.append(f"")
        header = "| Activity | " + " | ".join(results.keys()) + " |"
        lines.append(header)
        sep = "|" + "|".join(["---"] * (len(results) + 1)) + "|"
        lines.append(sep)
        for cname in class_names:
            vals = " | ".join(f"{results[m]['per_class_f1'].get(cname, 0):.3f}" for m in results)
            lines.append(f"| {cname} | {vals} |")
        lines.append(f"")

        # 分类报告（最佳模型）
        lines.append(f"### Classification Report — {best_model}")
        lines.append(f"")
        lines.append(f"```")
        lines.append(results[best_model]['classification_report'].strip())
        lines.append(f"```")
        lines.append(f"")

    # 输出文件列表
    lines.append(f"## Generated Files")
    lines.append(f"")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        lines.append(f"- `{f}` ({size / 1024:.1f} KB)")
    lines.append(f"")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n{'=' * 70}")
    print(f"  Markdown report saved to: {md_path}")
    print(f"{'=' * 70}")

    # 同时输出到终端
    print('\n'.join(lines))


# ============================================================
#  6. 主入口
# ============================================================

def build_summary_lines(har_results, wisdm_results):
    """构建终端汇总文本，返回行列表（供打印和保存）"""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  FINAL SUMMARY")
    lines.append("=" * 70)
    for ds_name, results in [('HAR', har_results), ('WISDM', wisdm_results)]:
        if not results:
            lines.append(f"\n  {ds_name}: (no results)")
            continue
        lines.append(f"\n  {ds_name}:")
        if isinstance(results, dict):
            for model_name, res in results.items():
                lines.append(f"    {model_name:<18s}  Acc: {res['acc']:.4f} ({res['acc'] * 100:.2f}%)")
        else:
            lines.append(f"    (unexpected type: {type(results).__name__})")
    lines.append("")
    return lines


if __name__ == '__main__':
    # 参数解析
    has_ds_flag = '--har' in sys.argv or '--wisdm' in sys.argv or '--all' in sys.argv
    run_har = '--har' in sys.argv or '--all' in sys.argv or not has_ds_flag
    run_wisdm = '--wisdm' in sys.argv or '--all' in sys.argv or not has_ds_flag

    epochs = 100
    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])

    model_types = ('mlp', 'cnn', 'cnnlstm', 'resnet')
    for flag in ['--mlp', '--cnn', '--cnnlstm', '--resnet']:
        if flag in sys.argv:
            model_types = (flag.lstrip('-'),)
            break

    print(f"Configuration: models={model_types}, epochs={epochs}")
    print(f"Output directory: {OUTPUT_DIR}")

    har_results = {}
    wisdm_results = {}
    if run_har:
        try:
            ret = run_har_v2(epochs, model_types)
            har_results = ret[0]  # (results_dict, class_names)
        except Exception as e:
            print(f"\n[ERROR] HAR pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    if run_wisdm:
        try:
            ret = run_wisdm_v2(epochs, model_types)
            wisdm_results = ret[0]
        except Exception as e:
            print(f"\n[ERROR] WISDM pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # 终端汇总
    summary_lines = build_summary_lines(har_results, wisdm_results)
    print('\n'.join(summary_lines))

    # 保存结果文档
    if har_results or wisdm_results:
        save_results_document(har_results, wisdm_results, OUTPUT_DIR, epochs,
                              terminal_summary='\n'.join(summary_lines))

    print(f"\nAll output saved to: {OUTPUT_DIR}")
