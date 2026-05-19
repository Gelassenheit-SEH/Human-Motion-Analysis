# Motion State Analysis — 运动数据分析与分类

利用智能手机惯性传感器（加速度计、陀螺仪）采集的运动数据，通过信号处理和机器学习方法进行活动识别与分类。

---

## 数据集

### 1. UCI HAR (Human Activity Recognition)
- **来源**: UCI Machine Learning Repository, ID 240
- **传感器**: 手机内置加速度计 + 陀螺仪，6 轴信号 (body_acc x/y/z, body_gyro x/y/z)
- **采样率**: 50 Hz
- **窗口**: 128 点 (2.56 秒)，自带预分割 train/test
- **活动**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING（共 6 类）
- **训练/测试**: 7352 / 2947 个窗口

### 2. WISDM (Wireless Sensor Data Mining)
- **来源**: UCI Machine Learning Repository, ID 507
- **传感器**: 手机加速度计，3 轴信号 (x/y/z)
- **采样率**: ~20 Hz
- **窗口**: 40 点 (2 秒)，50% 重叠滑动窗口
- **活动**: walking, jogging, stairs, sitting, standing 等 18 类（分类取前 5 类主要活动）
- **数据量**: ~158 万原始采样点 → 78985 个窗口

---

## 处理流程

```
原始数据 → 预处理 → 时域特征提取 + 频域特征提取 → MinMax归一化 → KNN/SVM分类 → 评估
```

### 预处理 ([preprocessing.py](preprocessing.py))
- **HAR**: 数据集本身已做重力分离，无需额外预处理
- **WISDM**: 
  1. 4 阶 Butterworth 低通滤波 (10 Hz)
  2. 0.3 Hz 低通提取重力分量，原始信号减去重力
  3. 滑动窗口 (40 点 / 20 点步长，50% 重叠)

### 特征提取 ([feature_extraction.py](feature_extraction.py))
- **时域特征** (每轴 7 个): 均值、方差、过零率、RMS、峰值、峰峰值、波形因子
- **频域特征** (每轴 5 个): 频谱质心、总能量、3 个频带能量
  - 加汉宁窗 → FFT 单边谱
  - HAR 频带: [0-3), [3-8), [8-15) Hz
  - WISDM 频带: [0-2), [2-6), [6-10) Hz
- 总特征数: HAR 72 维, WISDM 36 维

### 分类器 ([train.py](train.py), [test.py](test.py))
- **归一化**: MinMaxScaler，映射到 [-1, 1]
- **KNN**: k=5，欧氏距离
- **SVM**: RBF 核

---

## 结果

| 数据集 | 分类器 | 准确率 |
|--------|--------|--------|
| HAR | KNN (k=5) | **77.74%** |
| HAR | SVM (RBF) | **84.32%** |
| WISDM | KNN (k=5) | **85.01%** |
| WISDM | SVM (RBF) | **85.03%** |

详细分类报告和混淆矩阵见运行输出及 [output/](output/) 目录。

### 输出文件
| 文件 | 说明 |
|------|------|
| `har_cm.png` / `har_cm_knn.png` | HAR 混淆矩阵 (SVM / KNN) |
| `wisdm_cm.png` / `wisdm_cm_knn.png` | WISDM 混淆矩阵 (SVM / KNN) |
| `har_dft.png` / `har_stft.png` | HAR 频谱分析图 |
| `wisdm_dft.png` / `wisdm_stft.png` | WISDM 频谱分析图 |

---

## 环境与复现

### Python 版本
**Python 3.10.20**（推荐 3.10.x）

### 依赖库

| 包 | 版本 |
|------|-------|
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| pandas | 2.3.3 |
| requests | 2.34.2 |
| urllib3 | 2.7.0 |

### 复现步骤

#### 方式 1: conda（推荐）
```bash
# 1. 创建并激活环境
conda create -n motion_analysis python=3.10 -y
conda activate motion_analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
python download_dataset.py

# 4. 运行完整分析流程
python main.py
```

#### 方式 2: pip（仅需 Python 3.10+）
```bash
# 1. 创建虚拟环境
python -m venv motion_env
# Windows:
motion_env\Scripts\activate
# Linux/Mac:
# source motion_env/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据集
python download_dataset.py

# 4. 运行
python main.py
```

> **注意**: 数据集下载约 350 MB（HAR 58 MB + WISDM 296 MB）。`main.py` 运行约需 5-10 分钟，WISDM 预处理（~158 万采样点的滤波 + 滑动窗口）为主要耗时环节。

---

## 项目结构
```
├── main.py                 # 主流程编排
├── preprocessing.py        # 预处理（滤波器、WISDM 预处理流水线）
├── feature_extraction.py   # 特征提取（时域 + 频域）
├── train.py                # 数据加载、特征提取编排、模型训练
├── test.py                 # 模型评估、混淆矩阵、分类报告
├── download_dataset.py     # 数据集下载脚本
├── requirements.txt        # Python 依赖
├── dataset/                # 数据集存放目录（需下载）
│   ├── har/
│   └── wisdm/
└── output/                 # 输出图表
```
