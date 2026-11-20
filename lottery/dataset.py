"""滑动窗口时序数据集构建。

建模语义: 前 window 期号码 -> 预测下一期。所有切分均按时间顺序,
不做随机打乱, 避免时序泄漏。

可选统计特征 (--stats): 每个时间步追加此前 STATS_WINDOW 期内
各球出现频率, 为模型提供平滑的频率先验输入。
"""

from dataclasses import dataclass

import numpy as np

from lottery.config import TRAIN_RATIO

STATS_WINDOW = 50  # 频率特征回看期数


@dataclass
class Dataset:
    X: np.ndarray            # (n, window, F) 归一化输入
    targets: dict            # {列名: (n,) int 类别索引(号码-1)}
    X_next: np.ndarray       # (1, window, F) 最近 window 期(含最新一期), 预测下一期
    split: int               # 时间顺序切分点: X[:split] 训练, X[split:] 验证

    def train_val(self):
        y_tr = {k: v[:self.split] for k, v in self.targets.items()}
        y_va = {k: v[self.split:] for k, v in self.targets.items()}
        return self.X[:self.split], self.X[self.split:], y_tr, y_va


def _normalize(arr, cfg):
    """号码归一化到 [0,1]: 红 (num-1)/(red_max-1)，蓝同理。原地修改。"""
    r = cfg.red_count
    arr[..., :r] = (arr[..., :r] - 1) / (cfg.red_max - 1)
    arr[..., r:] = (arr[..., r:] - 1) / (cfg.blue_max - 1)


def _slide(a, window):
    """(n, F) -> (n-window, window, F) 滑动窗口。"""
    return np.stack([a[i:i + window] for i in range(len(a) - window)])


def _rolling_freq(vals, cfg, k=STATS_WINDOW):
    """每期之前 k 期内各球出现频率 (n, red_max+blue_max)。

    第 i 行只使用 [i-k, i-1] 的数据, 对第 i 期而言是已知信息, 无泄漏。
    """
    n = len(vals)
    out = np.zeros((n, cfg.red_max + cfg.blue_max), dtype=np.float32)
    idx_lo = np.maximum(np.arange(n) - k, 0)
    # i=0 无历史, 用 max(...,1) 避免除零(该行退化为第 0 期自身, 仅作占位)
    denom = np.maximum(np.arange(n) - idx_lo, 1).astype(np.float32)
    for col in range(cfg.n_balls):
        red = col < cfg.red_count
        mx = cfg.red_max if red else cfg.blue_max
        base = 0 if red else cfg.red_max
        onehot = np.zeros((n, mx), dtype=np.float32)
        onehot[np.arange(n), vals[:, col].astype(np.int32) - 1] = 1.0
        csum = np.vstack([np.zeros((1, mx), np.float32), np.cumsum(onehot, axis=0)])
        out[:, base:base + mx] = (csum[1:] - csum[idx_lo]) / denom[:, None]
    return out


def build_dataset(df_asc, cfg, window, stats=False) -> Dataset:
    """按时间升序 DataFrame 构建滑动窗口样本。stats=True 时追加频率特征。"""
    vals = df_asc[cfg.ball_cols].values.astype(np.float32)

    X = _slide(vals, window)
    nxt = vals[window:]
    # 预测输入必须包含最新一期, 不能复用最后一个训练样本(其窗口止于最新前一期)
    X_next = vals[-window:][np.newaxis, ...].copy()

    _normalize(X, cfg)
    _normalize(X_next, cfg)

    if stats:
        F = _rolling_freq(vals, cfg)          # (n, red_max+blue_max), 已在 [0,1]
        X = np.concatenate([X, _slide(F, window)], axis=-1)
        X_next = np.concatenate([X_next, F[-window:][np.newaxis]], axis=-1)

    targets = {}
    for i, c in enumerate(cfg.reds):
        targets[c] = nxt[:, i].astype(np.int32) - 1
    for j, c in enumerate(cfg.blues):
        targets[c] = nxt[:, cfg.red_count + j].astype(np.int32) - 1

    split = int(len(X) * TRAIN_RATIO)
    if split < 10 or len(X) - split < 5:
        raise RuntimeError(f'样本过少(n={len(X)})，请减小窗口期数')
    return Dataset(X=X, targets=targets, X_next=X_next, split=split)


def window_input(vals, cfg, window, stats=False):
    """将 window 期原始号码 (window, F) 转为 (1, window, F) 归一化输入。

    vals 为时间升序的完整历史, 频率特征基于其此前 STATS_WINDOW 期计算。
    """
    w = vals[-window:].astype(np.float32).copy()
    _normalize(w, cfg)
    if stats:
        F = _rolling_freq(vals, cfg)
        w = np.concatenate([w, F[-window:]], axis=-1)
    return w[np.newaxis, ...]
