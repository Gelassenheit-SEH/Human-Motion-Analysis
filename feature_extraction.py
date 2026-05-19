#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块
  - extract_time_features: 时域特征 (均值、方差、过零率、RMS、峰值、峰峰值、波形因子)
  - extract_freq_features: 频域特征 (汉宁窗→FFT→频谱质心+频带能量)
  - extract_all_features: 多轴信号统一提取时域+频域特征
"""

import numpy as np
from scipy import fft


def extract_time_features(signal_data):
    """提取时域特征: 均值、方差、过零率、RMS、峰值、峰峰值、波形因子"""
    mean_val = np.mean(signal_data)
    var_val = np.var(signal_data)
    zero_crossing = np.sum(np.diff(np.signbit(signal_data)))
    rms_val = np.sqrt(np.mean(signal_data ** 2))
    peak_val = np.max(np.abs(signal_data))
    peak_to_peak = np.max(signal_data) - np.min(signal_data)
    mean_abs = np.mean(np.abs(signal_data))
    waveform_factor = rms_val / (mean_abs + 1e-10)

    return {
        'mean': mean_val,
        'var': var_val,
        'zero_crossing': zero_crossing,
        'rms': rms_val,
        'peak': peak_val,
        'peak_to_peak': peak_to_peak,
        'waveform_factor': waveform_factor,
    }


def extract_freq_features(signal_data, fs, bands):
    """提取频域特征: 加汉宁窗 → FFT → 频谱质心 + 频带能量

    Args:
        signal_data: 1D 输入信号
        fs: 采样率 (Hz)
        bands: 频带列表 [(low, high), ...]

    Returns:
        dict: 包含 spectral_centroid, total_energy, band_{low}-{high} 等特征
    """
    n = len(signal_data)
    window = np.hanning(n)
    windowed = signal_data * window
    spectrum = fft.fft(windowed)[:n // 2]  # 单边谱
    freqs = fft.fftfreq(n, d=1 / fs)[:n // 2]
    magnitude = np.abs(spectrum)

    # 频谱质心
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

    # 频带能量
    band_energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = np.sum(magnitude[mask] ** 2)
        band_energies.append(energy)

    features = {
        'spectral_centroid': spectral_centroid,
        'total_energy': np.sum(magnitude ** 2),
    }
    for idx, (low, high) in enumerate(bands):
        features[f'band_{low}-{high}'] = band_energies[idx]

    return features


def extract_all_features(data, axis_names, fs, freq_bands):
    """对多轴信号统一提取时域+频域特征

    Args:
        data: dict, {axis_name: 1D_array}
        axis_names: list of axis names (决定提取顺序)
        fs: 采样率
        freq_bands: 频带列表

    Returns:
        feature_vector: 1D numpy array
        feature_names: list of strings
    """
    feature_vec = []
    feature_names = []

    for axis in axis_names:
        sig = data[axis]

        # 时域特征
        tf = extract_time_features(sig)
        for key, val in tf.items():
            feature_vec.append(val)
            feature_names.append(f'{axis}_{key}')

        # 频域特征
        ff = extract_freq_features(sig, fs, freq_bands)
        for key, val in ff.items():
            feature_vec.append(val)
            feature_names.append(f'{axis}_{key}')

    return np.array(feature_vec), feature_names
