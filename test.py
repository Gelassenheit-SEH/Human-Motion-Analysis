#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试/评估模块: 分类器评估、混淆矩阵、分类报告
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(clf, X_test, y_test):
    """评估单个分类器

    Returns:
        accuracy: float
        y_pred: ndarray
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred


def evaluate_both(knn, svm, X_test, y_test):
    """评估 KNN 和 SVM 两个分类器

    Returns:
        (knn_acc, knn_pred), (svm_acc, svm_pred)
    """
    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test)
    svm_acc, svm_pred = evaluate_model(svm, X_test, y_test)
    return (knn_acc, knn_pred), (svm_acc, svm_pred)


def print_results(name, y_test, y_pred, label_map=None, title_prefix=''):
    """打印准确率和分类报告

    Args:
        label_map: 如果是 dict (int→name)，用于替换数值标签
                   如果是 None，直接使用 y_test/y_pred 中的字符串
    """
    acc = accuracy_score(y_test, y_pred)
    print(f"  {name:12s}  Acc: {acc:.4f}")

    if label_map is not None:
        y_true_str = [label_map[y] for y in y_test]
        y_pred_str = [label_map[y] for y in y_pred]
    else:
        y_true_str = list(y_test)
        y_pred_str = list(y_pred)

    print(f"\nClassification Report ({title_prefix} {name}):")
    print(classification_report(y_true_str, y_pred_str, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, labels, title, filepath, cmap_name=None):
    """Morandi 渐变色混淆矩阵

    Args:
        labels: dict {int: name} 或 list of strings
        filepath: 完整保存路径
        cmap_name: 颜色方案名
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_true, y_pred)

    unique_labels = sorted(np.unique(y_true))

    if isinstance(labels, dict):
        label_names = [labels[l] for l in unique_labels]
    else:
        label_names = unique_labels

    # Morandi 渐变色系
    morandi_cmaps = {
        'Blues':     ['#F0EDEA', '#C4D1DC', '#8FA8C8', '#6A8AA8', '#4A6A88'],
        'Purples':   ['#F0EDEA', '#D1C8DC', '#B8A8C8', '#8A7AA8', '#605088'],
        'Oranges':   ['#F0EDEA', '#DCC8B8', '#C8A888', '#A8886A', '#886A50'],
        'Greens':    ['#F0EDEA', '#C4D1B8', '#A8C4A0', '#7AA880', '#5A8868'],
        'default':   ['#F5F0ED', '#D5C8C0', '#B5A098', '#8A7A75', '#60504A'],
    }
    cmap_colors = morandi_cmaps.get(cmap_name, morandi_cmaps['default'])
    n_bins = 100
    cmap_custom = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        'morandi', cmap_colors, N=n_bins
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=cmap_custom, interpolation='nearest')

    ax.set_xticks(range(len(unique_labels)))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(label_names, fontsize=10)

    # 标注：只显示数量
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else '#3D3D3D'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10, color=color,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=14, pad=15)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {filepath}')


def evaluate_har(knn, svm, rf, X_test, y_test, activities, output_dir='output'):
    """HAR 完整评估: 准确率 + 分类报告 + 混淆矩阵"""
    print("\n--- HAR Classification ---")

    # KNN
    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test)
    print(f"  {'KNN (k=5)':12s}  Acc: {knn_acc:.4f}")

    # SVM
    svm_acc, svm_pred = evaluate_model(svm, X_test, y_test)
    print(f"  {'SVM (RBF)':12s}  Acc: {svm_acc:.4f}")
    # RF
    rf_acc, rf_pred = evaluate_model(rf, X_test, y_test)
    print(f"  {'RF (100)':12s}  Acc: {rf_acc:.4f}")

    # 分类报告
    print(f"\nClassification Report (SVM):")
    print(classification_report(
        [activities[y] for y in y_test],
        [activities[y] for y in svm_pred],
        zero_division=0,
    ))
    print(f"\nClassification Report (KNN):")
    print(classification_report(
        [activities[y] for y in y_test],
        [activities[y] for y in knn_pred],
        zero_division=0,
    ))
    print(f"\nClassification Report (RF):")
    print(classification_report(
        [activities[y] for y in y_test],
        [activities[y] for y in rf_pred],
        zero_division=0,
    ))

    # 混淆矩阵
    plot_confusion_matrix(
        y_test, svm_pred, activities,
        'HAR - SVM Confusion Matrix',
        os.path.join(output_dir, 'har_cm.png'),
        cmap_name='Purples',
    )
    plot_confusion_matrix(
        y_test, knn_pred, activities,
        'HAR - KNN Confusion Matrix',
        os.path.join(output_dir, 'har_cm_knn.png'),
        cmap_name='Oranges',
    )
    plot_confusion_matrix(
        y_test, rf_pred, activities,
        'HAR - RF Confusion Matrix',
        os.path.join(output_dir, 'har_cm_rf.png'),
        cmap_name='Greens',
    )

    return (knn_acc, knn_pred), (svm_acc, svm_pred), (rf_acc, rf_pred)


def evaluate_wisdm(knn, svm, rf, X_test, y_test, output_dir='output'):
    """WISDM 完整评估: 准确率 + 分类报告 + 混淆矩阵"""
    print(f"\n--- WISDM Classification ---")

    # KNN
    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test)
    print(f"  {'KNN (k=5)':12s}  Acc: {knn_acc:.4f}")

    # SVM
    svm_acc, svm_pred = evaluate_model(svm, X_test, y_test)
    print(f"  {'SVM (RBF)':12s}  Acc: {svm_acc:.4f}")

    # RF
    rf_acc, rf_pred = evaluate_model(rf, X_test, y_test)
    print(f"  {'RF (100)':12s}  Acc: {rf_acc:.4f}")

    # 分类报告
    print(f"\nClassification Report (SVM):")
    print(classification_report(y_test, svm_pred, zero_division=0))
    print(f"\nClassification Report (KNN):")
    print(classification_report(y_test, knn_pred, zero_division=0))
    print(f"\nClassification Report (RF):")
    print(classification_report(y_test, rf_pred, zero_division=0))

    # 混淆矩阵
    labels_sorted = sorted(np.unique(y_test))
    plot_confusion_matrix(
        y_test, svm_pred, labels_sorted,
        'WISDM - SVM Confusion Matrix',
        os.path.join(output_dir, 'wisdm_cm.png'),
        cmap_name='Purples',
    )
    plot_confusion_matrix(
        y_test, knn_pred, labels_sorted,
        'WISDM - KNN Confusion Matrix',
        os.path.join(output_dir, 'wisdm_cm_knn.png'),
        cmap_name='Oranges',
    )
    plot_confusion_matrix(
        y_test, rf_pred, labels_sorted,
        'WISDM - RF Confusion Matrix',
        os.path.join(output_dir, 'wisdm_cm_rf.png'),
        cmap_name='Greens',
    )


    return (knn_acc, knn_pred), (svm_acc, svm_pred), (rf_acc, rf_pred)


if __name__ == '__main__':
    print("test.py — 请通过 main.py 运行完整评估流程")
