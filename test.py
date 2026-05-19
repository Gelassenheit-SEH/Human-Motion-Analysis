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


def plot_confusion_matrix(y_true, y_pred, labels, title, filepath, cmap='Blues'):
    """绘制并保存混淆矩阵

    Args:
        labels: dict {int: name} 或 list of strings
                如果为 dict, 自动按 key 排序并映射为 name
        filepath: 完整保存路径 (包括文件名)
    """
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(np.unique(y_true))

    if isinstance(labels, dict):
        label_names = [labels[l] for l in unique_labels]
    else:
        label_names = unique_labels

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names)

    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f'[Saved] {filepath}')


def evaluate_har(knn, svm, X_test, y_test, activities, output_dir='output'):
    """HAR 完整评估: 准确率 + 分类报告 + 混淆矩阵"""
    print("\n--- HAR Classification ---")

    # KNN
    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test)
    print(f"  {'KNN (k=5)':12s}  Acc: {knn_acc:.4f}")

    # SVM
    svm_acc, svm_pred = evaluate_model(svm, X_test, y_test)
    print(f"  {'SVM (RBF)':12s}  Acc: {svm_acc:.4f}")

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

    # 混淆矩阵
    plot_confusion_matrix(
        y_test, svm_pred, activities,
        'HAR - SVM Confusion Matrix',
        os.path.join(output_dir, 'har_cm.png'),
        cmap='Blues',
    )
    plot_confusion_matrix(
        y_test, knn_pred, activities,
        'HAR - KNN Confusion Matrix',
        os.path.join(output_dir, 'har_cm_knn.png'),
        cmap='Blues',
    )

    return (knn_acc, knn_pred), (svm_acc, svm_pred)


def evaluate_wisdm(knn, svm, X_test, y_test, output_dir='output'):
    """WISDM 完整评估: 准确率 + 分类报告 + 混淆矩阵"""
    print(f"\n--- WISDM Classification ---")

    # KNN
    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test)
    print(f"  {'KNN (k=5)':12s}  Acc: {knn_acc:.4f}")

    # SVM
    svm_acc, svm_pred = evaluate_model(svm, X_test, y_test)
    print(f"  {'SVM (RBF)':12s}  Acc: {svm_acc:.4f}")

    # 分类报告
    print(f"\nClassification Report (SVM):")
    print(classification_report(y_test, svm_pred, zero_division=0))
    print(f"\nClassification Report (KNN):")
    print(classification_report(y_test, knn_pred, zero_division=0))

    # 混淆矩阵
    labels_sorted = sorted(np.unique(y_test))
    plot_confusion_matrix(
        y_test, svm_pred, labels_sorted,
        'WISDM - SVM Confusion Matrix',
        os.path.join(output_dir, 'wisdm_cm.png'),
        cmap='Purples',
    )
    plot_confusion_matrix(
        y_test, knn_pred, labels_sorted,
        'WISDM - KNN Confusion Matrix',
        os.path.join(output_dir, 'wisdm_cm_knn.png'),
        cmap='Oranges',
    )

    return (knn_acc, knn_pred), (svm_acc, svm_pred)


if __name__ == '__main__':
    print("test.py — 请通过 main.py 运行完整评估流程")
