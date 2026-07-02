#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习模型: MLP / CNN1D / CNN-LSTM 用于 HAR 活动分类

- MLP:      输入为手工特征向量，适用于 enhanced feature pipeline
- CNN1D:    输入为原始窗口信号 (channels × timesteps)，自动学习局部模式
- CNN-LSTM: CNN 提取局部特征 + LSTM 建模时间依赖，端到端学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
import copy
import math

warnings.filterwarnings('ignore')

# ======================== 设备管理 ========================

def get_device():
    """自动选择可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# ======================== 模型定义 ========================

class MLPClassifier(nn.Module):
    """多层感知机，用于手工特征向量分类

    Args:
        input_dim: 输入特征维度
        num_classes: 分类类别数
        hidden_dims: 各隐藏层神经元数，默认 [256, 128, 64]
        dropout: Dropout 比例
    """
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """一维卷积神经网络，用于原始时序信号分类

    通过堆叠 Conv1d → BN → ReLU → MaxPool 层逐步提取
    多尺度时序模式，最后经全局平均池化后分类。

    Args:
        n_channels:   输入通道数（信号轴数，如 3 或 6）
        n_timesteps:  时间步数（窗口长度）
        num_classes:  分类类别数
        conv_filters: 各卷积层滤波器数，默认 [64, 128, 256]
        kernel_size:  卷积核大小
        dropout:      Dropout 比例
    """
    def __init__(self, n_channels, n_timesteps, num_classes,
                 conv_filters=(64, 128, 256), kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        prev_ch = n_channels
        for i, cf in enumerate(conv_filters):
            layers.append(nn.Conv1d(prev_ch, cf, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(cf))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            prev_ch = cf
        self.conv = nn.Sequential(*layers)

        # 计算卷积输出展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_timesteps)
            conv_out = self.conv(dummy)
            self.flatten_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNNLSTM(nn.Module):
    """CNN-LSTM 混合模型，用于原始时序信号分类

    CNN 阶段提取局部时序特征并降采样，然后 LSTM 在压缩后的
    时间轴上建模长程依赖关系，兼顾局部模式和全局上下文。

    Args:
        n_channels:   输入通道数
        n_timesteps:  时间步数
        num_classes:  分类类别数
        conv_filters: CNN 滤波器数列表
        lstm_hidden:  LSTM 隐藏层维度
        lstm_layers:  LSTM 层数
        kernel_size:  卷积核大小
        dropout:      Dropout 比例
    """
    def __init__(self, n_channels, n_timesteps, num_classes,
                 conv_filters=(64, 128, 256), lstm_hidden=128, lstm_layers=2,
                 kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        prev_ch = n_channels
        for cf in conv_filters:
            layers.append(nn.Conv1d(prev_ch, cf, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(cf))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            prev_ch = cf
        self.conv = nn.Sequential(*layers)
        self.conv_out_channels = prev_ch

        self.lstm = nn.LSTM(
            input_size=prev_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, channels, timesteps)
        x = self.conv(x)                        # → (batch, conv_out_channels, reduced_timesteps)
        x = x.permute(0, 2, 1)                  # → (batch, reduced_timesteps, conv_out_channels)
        lstm_out, _ = self.lstm(x)              # → (batch, reduced_timesteps, lstm_hidden*2)
        x = lstm_out[:, -1, :]                  # 取最后时间步 → (batch, lstm_hidden*2)
        return self.fc(x)


# ======================== ResNet1D（多尺度残差卷积） ========================

class ResBlock1D(nn.Module):
    """一维残差块：双卷积 + 跳跃连接，集成多尺度卷积核"""

    def __init__(self, in_ch, out_ch, kernel_sizes=(3, 5, 7), stride=1, dropout=0.2):
        super().__init__()
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

        # 多尺度卷积分支（并行不同 kernel size，结果相加）
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, ks, stride=stride, padding=ks // 2, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            for ks in kernel_sizes
        ])

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, max(kernel_sizes), stride=1,
                      padding=max(kernel_sizes) // 2, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.relu = nn.ReLU()

        # 1×1 投影跳跃连接（维度不匹配时）
        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)

        # 多尺度分支求和
        out = sum(conv(x) for conv in self.convs)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return self.dropout(out)


class ResNet1D(nn.Module):
    """多尺度残差卷积网络，专门优化步态/楼梯等频率敏感活动

    Args:
        n_channels:   输入通道数
        n_timesteps:  时间步数
        num_classes:  分类数
        base_ch:      基础通道数，逐层翻倍 [base_ch, base_ch*2, base_ch*4]
        dropout:      Dropout 比例
    """

    def __init__(self, n_channels, n_timesteps, num_classes,
                 base_ch=64, dropout=0.3):
        super().__init__()
        # 初始卷积（不降采样，保留细粒度信息）
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # 三个残差阶段，通道数逐级翻倍
        ch = [base_ch, base_ch * 2, base_ch * 4]
        self.layer1 = self._make_layer(ch[0], ch[0], num_blocks=2, dropout=dropout)
        self.layer2 = self._make_layer(ch[0], ch[1], num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(ch[1], ch[2], num_blocks=2, stride=2, dropout=dropout)

        # 全局自适应池化 → FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ch[2], 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1, dropout=0.2):
        layers = []
        layers.append(ResBlock1D(in_ch, out_ch, stride=stride, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ======================== Focal Loss ========================

class FocalLoss(nn.Module):
    """Focal Loss: 自动降低易分样本的 loss 权重，聚焦难分样本（如楼梯混淆）

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: 类别权重，None 则自动从训练集计算
        gamma: 聚焦参数，越大越关注难分样本（推荐 2.0）
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ======================== 信号增强 ========================

def augment_signal(X, noise_std=0.02, time_shift_max=5, scale_range=(0.9, 1.1)):
    """对时序信号做随机增强（仅训练时使用）

    Args:
        X: (N, channels, timesteps) 或 (N, n_features)
        noise_std:      高斯噪声标准差（相对于信号 std）
        time_shift_max: 最大时间偏移（仅对 CNN 格式生效）
        scale_range:    幅值随机缩放范围

    Returns:
        X_aug: 增强后的数据
    """
    X_aug = X.copy() if isinstance(X, np.ndarray) else X.clone()

    # 仅对 CNN 格式 (batch, channels, timesteps) 做时间偏移
    if X_aug.ndim == 3:
        shift = np.random.randint(-time_shift_max, time_shift_max + 1)
        if shift != 0:
            X_aug = np.roll(X_aug, shift, axis=-1)

    # 幅值缩放
    scale = np.random.uniform(*scale_range)
    X_aug = X_aug * scale

    # 高斯噪声（相对于每个样本的 std）
    if noise_std > 0:
        sample_std = np.std(X_aug, axis=-1, keepdims=True) + 1e-8
        noise = np.random.randn(*X_aug.shape).astype(np.float32) * noise_std * sample_std
        X_aug = X_aug + noise

    return X_aug


# ======================== 模型工厂 ========================

def create_model(model_type, input_dim_or_shape, num_classes, **kwargs):
    """创建指定类型的深度学习模型

    Args:
        model_type: 'mlp', 'cnn', 或 'cnnlstm'
        input_dim_or_shape:
            - 对于 'mlp': int，特征维度
            - 对于 'cnn' / 'cnnlstm': (n_channels, n_timesteps) 元组
        num_classes: 类别数
        **kwargs: 传递给具体模型构造函数的参数

    Returns:
        nn.Module
    """
    if model_type == 'mlp':
        return MLPClassifier(
            input_dim=input_dim_or_shape,
            num_classes=num_classes,
            hidden_dims=kwargs.get('hidden_dims', (256, 128, 64)),
            dropout=kwargs.get('dropout', 0.3),
        )
    elif model_type == 'cnn':
        n_channels, n_timesteps = input_dim_or_shape
        return CNN1D(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            conv_filters=kwargs.get('conv_filters', (64, 128, 256)),
            kernel_size=kwargs.get('kernel_size', 5),
            dropout=kwargs.get('dropout', 0.3),
        )
    elif model_type == 'cnnlstm':
        n_channels, n_timesteps = input_dim_or_shape
        return CNNLSTM(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            conv_filters=kwargs.get('conv_filters', (64, 128, 256)),
            lstm_hidden=kwargs.get('lstm_hidden', 128),
            lstm_layers=kwargs.get('lstm_layers', 2),
            kernel_size=kwargs.get('kernel_size', 5),
            dropout=kwargs.get('dropout', 0.3),
        )
    elif model_type == 'resnet':
        n_channels, n_timesteps = input_dim_or_shape
        return ResNet1D(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            base_ch=kwargs.get('base_ch', 64),
            dropout=kwargs.get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ======================== 训练与评估 ========================

def train_dl_model(model, X, y, model_type='mlp',
                   batch_size=64, epochs=100, lr=0.001,
                   weight_decay=1e-4, early_stopping_patience=20,
                   val_size=0.15, verbose=True, seed=42,
                   use_focal=True, focal_gamma=2.0,
                   use_augment=True, warmup_epochs=5):
    """训练深度学习模型（增强版：Focal Loss + 数据增强 + Cosine 退火 + 早停）

    Args:
        model, X, y, model_type, batch_size: 同旧版
        epochs:                  最大训练轮次
        lr:                      初始学习率
        weight_decay:            L2 正则化系数
        early_stopping_patience: 早停耐心值
        val_size:                验证集比例
        verbose:                 是否打印训练信息
        seed:                    随机种子
        use_focal:               是否使用 Focal Loss（聚焦难分样本）
        focal_gamma:             Focal Loss gamma 参数
        use_augment:             是否对 CNN 输入做信号增强
        warmup_epochs:           LR warmup 轮数

    Returns:
        model, history, le
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    model = model.to(device)

    # 标签编码
    le = None
    if isinstance(y[0], str):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    else:
        y_enc = np.asarray(y, dtype=np.int64)
        if y_enc.min() > 0:
            y_enc = y_enc - y_enc.min()

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=val_size, random_state=seed, stratify=y_enc
    )

    # ---- 类别权重 (用于 FocalLoss / CrossEntropy) ----
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # ---- 损失函数 ----
    if use_focal:
        criterion = FocalLoss(alpha=class_weights_t, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # ---- 优化器 + Cosine 退火调度 ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(X_train) // batch_size)

    def cosine_schedule(step):
        if step < warmup_epochs * max(1, len(X_train) // batch_size):
            return (step + 1) / max(1, warmup_epochs * max(1, len(X_train) // batch_size))
        progress = (step - warmup_epochs * max(1, len(X_train) // batch_size)) / max(1, total_steps - warmup_epochs * max(1, len(X_train) // batch_size))
        return 0.5 * (1 + math.cos(math.pi * progress))

    # ---- 训练循环 ----
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    global_step = 0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss_sum = 0.0
        # 每个 epoch 重新打乱 + 增强（仅对 CNN 格式）
        indices = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]

            # 信号增强（仅训练时，仅 CNN 格式）
            if use_augment and model_type in ('cnn', 'cnnlstm', 'resnet'):
                batch_X = augment_signal(batch_X)

            batch_X_t = torch.tensor(batch_X, dtype=torch.float32).to(device)
            batch_y_t = torch.tensor(batch_y, dtype=torch.long).to(device)

            # LR schedule
            lr_scale = cosine_schedule(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale
            global_step += 1

            optimizer.zero_grad()
            outputs = model(batch_X_t)
            loss = criterion(outputs, batch_y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(idx)

        avg_train_loss = train_loss_sum / len(X_train)
        history['train_loss'].append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_labels = [], []
        # 验证损失用标准 CE（不用 Focal，便于比较）
        val_criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        val_ds = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = val_criterion(outputs, batch_y)
                val_loss_sum += loss.item() * batch_X.size(0)
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss_sum / len(X_val)
        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:3d}/{epochs} | "
                  f"lr: {current_lr:.2e} | "
                  f"train_loss: {avg_train_loss:.4f} | "
                  f"val_loss: {avg_val_loss:.4f} | "
                  f"val_acc: {val_acc:.4f}")

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

    # 加载最佳权重
    model.load_state_dict(best_model_state)
    return model, history, le


@torch.no_grad()
def predict_dl(model, X, model_type='mlp', batch_size=128):
    """使用训练好的 DL 模型进行预测

    Args:
        model: 训练好的 nn.Module
        X:     输入数据 (N, ...)
        model_type: 'mlp' / 'cnn' / 'cnnlstm'
        batch_size: 推理批大小

    Returns:
        y_pred: 预测的类别索引 (int array)
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    for (batch_X,) in loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())

    return np.concatenate(all_preds)


def evaluate_dl_model(model, X_test, y_test, model_type='mlp',
                      label_map=None, batch_size=128):
    """评估深度学习模型：准确率 + 分类报告

    Args:
        model:     训练好的 nn.Module
        X_test:    测试数据
        y_test:    测试标签（与训练时格式一致，支持 str 或 int）
        model_type: 'mlp' / 'cnn' / 'cnnlstm'
        label_map:  dict {int: name}，用于将整数标签映射为活动名称
        batch_size: 推理批大小

    Returns:
        acc:    准确率
        y_pred: 预测标签（整数索引）
    """
    # 处理标签
    if isinstance(y_test[0], str):
        le = LabelEncoder()
        y_enc = le.fit_transform(y_test)
    else:
        y_enc = np.asarray(y_test, dtype=np.int64)
        if y_enc.min() > 0:
            y_enc = y_enc - y_enc.min()

    y_pred = predict_dl(model, X_test, model_type, batch_size)
    acc = accuracy_score(y_enc, y_pred)

    print(f"  {'DL-' + model_type.upper():16s} Acc: {acc:.4f}")

    # 分类报告
    if label_map is not None:
        rev_map = {v: k for k, v in label_map.items()}
        # label_map 是 {int: name}, y_pred 是 0-based index
        unique_acts = sorted(label_map.keys())
        target_names = [label_map[k] for k in unique_acts]
        y_true_str = [label_map[unique_acts[l]] for l in y_enc]
        y_pred_str = [label_map[unique_acts[p]] for p in y_pred]
    else:
        target_names = None
        y_true_str = y_enc
        y_pred_str = y_pred

    print(f"\nClassification Report (DL-{model_type.upper()}):")
    print(classification_report(y_true_str, y_pred_str,
                                target_names=target_names, zero_division=0))

    return acc, y_pred
