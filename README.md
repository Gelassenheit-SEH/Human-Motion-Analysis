# Human Motion Analysis — 运动数据分析与分类

利用智能手机惯性传感器（加速度计、陀螺仪）采集的运动数据，通过信号处理、特征工程、传统机器学习和深度学习方法进行人体活动识别（Human Activity Recognition, HAR）。

项目按照基础→进阶→挑战1→挑战2四个阶段逐步推进，每个阶段针对上一阶段暴露的瓶颈进行改进。

---

## 项目阶段概览

| 阶段 | 入口命令 | 核心内容 | 最高准确率 |
|------|----------|----------|-----------|
| **基础** | `python main.py --stage basic` | 低通滤波、重力分离、滑动窗口、时频域特征、KNN/SVM分类 | 87.14% (HAR) / 86.30% (WISDM) |
| **进阶** | `python main.py` | 增强特征(偏度/峰度/SMA/跨轴相关/合成幅值)、随机森林、特征重要性 | **90.30%** (HAR) / **92.20%** (WISDM) |
| **挑战1** | `python main.py --stage challenge1` | 多模态传感器融合(Acc+Gyro)、消融实验、Early/Late Fusion、超参数调优、推理延迟分析 | 90.30% (特征级融合) |
| **挑战2** | `python run_dl_v2.py` | 深度学习(MLP/CNN1D/CNN-LSTM/ResNet1D)、Focal Loss、Cosine退火、信号增强 | **94.13%** (HAR) / **94.60%** (WISDM) |

---

## 数据集

### UCI HAR (Human Activity Recognition)
- **来源**: UCI Machine Learning Repository, ID 240
- **传感器**: 加速度计 + 陀螺仪，6 轴信号 (body_acc x/y/z, body_gyro x/y/z)
- **采样率**: 50 Hz | **窗口**: 128 点 (2.56 s)，预分割 train/test
- **活动**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING（6 类）
- **规模**: 训练 7352 / 测试 2947 个窗口

### WISDM (Wireless Sensor Data Mining)
- **来源**: UCI Machine Learning Repository, ID 507
- **传感器**: 加速度计，3 轴 (x/y/z)
- **采样率**: ~20 Hz | **窗口**: 40 点 (2 s)，50% 重叠
- **活动**: walking, jogging, stairs, sitting, standing（18 类中取 5 类主要活动）
- **规模**: ~158 万原始采样点 → 78985 个窗口

---

## 技术路线

```
原始信号 → 低通滤波 + 重力分离 → 滑动窗口
         → 时域/频域/跨轴/合成幅值特征提取
         → MinMax归一化 → KNN/SVM/RF/MLP/CNN → 评估与可视化
```

---

## 阶段一：基础流水线

### 信号预处理 ([preprocessing.py](preprocessing.py))
- **HAR**: 数据集已预分割，无需额外预处理
- **WISDM**: 4 阶 Butterworth 低通滤波 (10 Hz) → 0.3 Hz 低通提取重力分量 → 原始信号减重力 → 滑动窗口 (40 点 / 20 点步长)

### 特征提取 ([feature_extraction.py](feature_extraction.py))
- **时域** (每轴 10 个): 均值、方差、过零率、RMS、峰值、峰峰值、波形因子、偏度、峰度、SMA
- **频域** (每轴 3 个): 频谱质心、总能量、3 个频带能量
  - 加汉宁窗 → FFT 单边谱
  - HAR 频带: [0-3), [3-8), [8-15) Hz
  - WISDM 频带: [0-2), [2-6), [6-10) Hz
- **跨轴特征**: Pearson 相关系数、合成幅值通道的时域+频域特征
- **姿态角特征**: Pitch/Roll 均值/标准差/范围、重力分量占比
- 特征维度: HAR 120 维, WISDM 63 维

### 分类器 ([train.py](train.py))
| 模型 | 配置 |
|------|------|
| KNN | k=5，欧氏距离 |
| SVM | RBF 核，C=1.0，class_weight='balanced' |
| **随机森林** | n_estimators=100，random_state=42 |

### 基础结果

| 数据集 | KNN | SVM | 随机森林 |
|--------|-----|-----|---------|
| UCI HAR | 82.46% | 87.14% | **90.30%** |
| WISDM | 88.99% | 86.30% | **92.20%** |

---

## 阶段二：进阶 — 特征增强与分类器优化

在基础特征之上增加了偏度、峰度、信号幅值面积（SMA）、跨轴 Pearson 相关系数和合成幅值通道，使特征维度从 HAR 72→120 / WISDM 36→63。

引入 Random Forest 替代单一 SVM，利用 100 棵决策树投票适应高维特征空间。同时启用 SVM 的 class_weight='balanced' 自动为少数类分配更高权重。

### 分阶段提升

| 数据集 | 原始 SVM | 增强特征 + SVM | 增强特征 + RF |
|--------|----------|---------------|--------------|
| UCI HAR | 84.32% | 87.48% | **90.30%** |
| WISDM | 85.03% | 86.30% | **92.20%** |

关键改善：UCI HAR 坐姿 F1 值 0.66→0.82，下楼梯 Recall 0.83→0.92。

---

## 阶段三：挑战1 — 多模态传感器融合与工程调优

仅适用于 UCI HAR（同时含加速度计和陀螺仪）。

### 融合策略

| 策略 | 实现方式 | 准确率 | 推理延迟 |
|------|----------|--------|---------|
| 仅加速度计 | 72 维 Acc 特征 → RF | 83.44% | 5.87 μs |
| 仅陀螺仪 | 48 维 Gyro 特征 → RF | 78.35% | 6.82 μs |
| 特征级融合 (Early Fusion) | 120 维 Acc+Gyro 拼接 → RF | **90.30%** | ~6.12 μs |
| 决策级融合 (Late Fusion) | Acc RF + Gyro RF → 1:1 软投票 | 88.97% | — |

**结论**: 加速度计是主要判别模态，陀螺仪是补充纠错模态。特征级融合优于决策级融合，且维度翻倍并未显著增加推理延迟。

### 超参数调优

| 树数量 | 准确率 | 训练时间 |
|--------|--------|---------|
| 10 | 87.14% | 0.475 s |
| 50 | 89.89% | 2.394 s |
| **100** | **90.30%** | 4.801 s |
| 150 | 90.84% | 6.085 s |
| 200 | 90.77% | 8.190 s |

综合精度与效率，最终选用 n_estimators=100。

---

## 阶段四：挑战2 — 深度学习探索

### 模型架构 ([dl_models.py](dl_models.py), [run_dl_v2.py](run_dl_v2.py))

| 模型 | 输入形式 | 设计目的 |
|------|----------|----------|
| **DL-MLP** | 增强手工特征 | 在传统特征空间中学习非线性组合 |
| **DL-CNN1D** | 原始窗口序列 | 提取局部时间波形和短时冲击模式 |
| **DL-CNN-LSTM** | 原始窗口序列 | CNN 提取局部 + LSTM 建模时间依赖 |
| **DL-ResNet1D** | 原始窗口序列 | 残差连接 + 多尺度卷积捕获不同粒度特征 |

### 训练策略
- **损失函数**: Focal Loss（聚焦难分样本）+ 类别权重
- **优化器**: AdamW + Cosine 学习率退火 + warmup
- **正则化**: Dropout、梯度裁剪、Early Stopping
- **数据增强**: 时间平移、幅值缩放、高斯噪声
- **输入通道**: HAR 使用 9 通道（body_acc×3 + body_gyro×3 + total_acc×3），WISDM 使用 3 通道

### 深度学习结果

| 数据集 | DL-MLP | DL-CNN | DL-CNN-LSTM | DL-ResNet | 最优模型 |
|--------|--------|--------|-------------|-----------|---------|
| UCI HAR | 92.23% | **94.13%** | 92.43% | 94.03% | DL-CNN |
| WISDM | **94.60%** | 92.25% | 91.86% | 91.98% | DL-MLP |

**关键发现**: CNN 善于从原始序列学习局部波形模式(UCI HAR)；而在短窗口少通道的数据上(WISDM)，增强手工特征 + MLP 更稳定。

---

## 环境与复现

### 环境配置

**Python 3.10.x**（推荐 3.10.20）

**核心依赖**: numpy, scipy, scikit-learn, matplotlib, pandas, torch≥2.0.0, seaborn

### 复现步骤

#### 方式 1: conda（推荐）
```bash
conda create -n motion_analysis python=3.10 -y
conda activate motion_analysis
pip install -r requirements.txt
python download_dataset.py
```

#### 方式 2: pip
```bash
python -m venv motion_env
# Windows: motion_env\Scripts\activate
# Linux/Mac: source motion_env/bin/activate
pip install -r requirements.txt
python download_dataset.py
```

### 运行

```bash
# 基础流水线（预处理 + 特征 + KNN/SVM/RF）
python main.py --stage basic

# 进阶（基础 + 增强特征可视化 + 挑战1融合实验）【默认】
python main.py

# 仅运行挑战1融合实验（需先运行过完整流水线生成缓存）
python main.py --stage challenge1

# 深度学习（挑战2）
python run_dl_v2.py                          # 全部模型，100 epochs
python run_dl_v2.py --har --cnn              # HAR + CNN only
python run_dl_v2.py --wisdm --epochs 50      # WISDM only, 50 epochs
python run_dl_v2.py --har --resnet           # HAR + ResNet only

# 仅重新生成可视化（需已有缓存）
python run_viz.py
```

> **注意**: 数据集下载约 350 MB。`main.py` 首次运行约 5-10 分钟（WISDM 预处理为主要耗时环节）。深度学习全量实验约 40-50 分钟，建议使用 CUDA 版 PyTorch。

---

## 结果汇总

### 传统机器学习

| 数据集 | KNN | SVM (RBF) | 随机森林 |
|--------|-----|-----------|---------|
| UCI HAR | 82.46% | 87.14% | **90.30%** |
| WISDM | 88.99% | 86.30% | **92.20%** |

### 传感器融合 (UCI HAR)

| 策略 | 准确率 |
|------|--------|
| Acc Only | 83.44% |
| Gyro Only | 78.35% |
| Early Fusion (Acc+Gyro) | **90.30%** |
| Late Fusion (Soft Voting) | 88.97% |

### 深度学习

| 数据集 | 最优模型 | 准确率 |
|--------|----------|--------|
| UCI HAR | DL-CNN | **94.13%** |
| WISDM | DL-MLP | **94.60%** |

---

## 输出文件

### 基础/进阶可视化 (`output/`)
| 文件 | 说明 |
|------|------|
| `har_cm.png` / `wisdm_cm.png` | 混淆矩阵 |
| `har_rf_importance.png` / `wisdm_rf_importance.png` | 随机森林特征重要性 |
| `har_time_features.png` / `wisdm_time_features.png` | 时域特征分组柱状图 |
| `har_violin.png` / `wisdm_violin.png` | 时域特征小提琴图 |
| `har_radar.png` / `wisdm_radar.png` | 特征雷达图 |
| `har_dft.png` / `har_stft.png` | 频谱分析 |
| `classifier_comparison.png` | 分类器准确率对比 |

### 挑战1 可视化 (`output/`)
| 文件 | 说明 |
|------|------|
| `ablation_sensor_fusion.png` | Acc/Gyro 消融对比 |
| `decision_fusion_comparison.png` | Early vs Late Fusion |
| `enrichment_per_class_acc.png` | 各类别准确率 |
| `enrichment_diff_cm.png` | 融合差值混淆矩阵 |
| `enrichment_feature_importance.png` | 融合模型特征重要性 |
| `challenge_tuning_curve.png` | 超参数调优曲线 |
| `challenge_tradeoff_latency.png` | 精度-延迟权衡 |

### 挑战2 结果 (`output/v2/`)
自动生成训练曲线、类别 F1 对比、混淆矩阵和 `results_summary.md`。

---

## 项目结构

```
├── main.py                   # 主流程编排（基础+进阶+挑战1）
├── preprocessing.py          # 信号预处理（滤波、重力分离、滑动窗口）
├── feature_extraction.py     # 特征提取（时域+频域+跨轴+姿态角）
├── train.py                  # 数据加载、特征批处理、传统ML训练、DL数据准备
├── test.py                   # 模型评估、混淆矩阵、分类报告
├── dl_models.py              # 深度学习模型定义（MLP/CNN1D/CNN-LSTM/ResNet1D）
├── run_dl.py                 # 深度学习训练入口 v1
├── run_dl_v2.py              # 深度学习训练入口 v2（增强版）
├── run_viz.py                # 仅运行可视化（加载缓存，跳过流水线）
├── download_dataset.py       # 数据集下载脚本
├── requirements.txt          # Python 依赖
├── environment.yml           # Conda 环境完整导出
├── CHANGELOG-features.md     # 特性变更日志
├── docs/
│   └── run_dl_v2.md          # 深度学习流水线技术文档
├── dataset/                  # 数据集存放目录（需下载）
└── output/                   # 输出图表与缓存
    └── cache/                # 流水线缓存（用于快速重绘）
```

---

## 参考资料

- [1] UCI Machine Learning Repository. Human Activity Recognition Using Smartphones, ID 240.
- [2] UCI Machine Learning Repository. WISDM Smartphone and Smartwatch Activity and Biometrics Dataset, ID 507.
- [3] 项目仓库：[Gelassenheit-SEH/Human-Motion-Analysis](https://github.com/Gelassenheit-SEH/Human-Motion-Analysis)
