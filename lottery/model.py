"""模型构建与保存。"""

import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_model(cfg, window, units=64, dropout=0.2, lr=1e-3, in_dim=None):
    """逐球位 softmax 分类头模型: 输出天然落在合法号码范围。

    in_dim 默认号码维; 追加统计特征时由调用方传入实际特征维。
    """
    inp = keras.Input(shape=(window, in_dim or cfg.n_balls))
    x = layers.LSTM(units)(inp)  # 默认 tanh，relu 易梯度爆炸/NaN
    x = layers.Dropout(dropout)(x)
    outs = [layers.Dense(cfg.red_max, activation='softmax', name=c)(x) for c in cfg.reds]
    outs += [layers.Dense(cfg.blue_max, activation='softmax', name=c)(x) for c in cfg.blues]
    model = keras.Model(inp, outs, name=cfg.tag)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics={c: ['accuracy'] for c in cfg.ball_cols})
    return model


def save_model(model, cfg):
    """保存模型为 .keras 格式。"""
    model.save(cfg.model)
    return cfg.model


def load_model(cfg):
    """加载已保存的模型(仅供展示概率用)。"""
    return keras.models.load_model(cfg.model)


def early_stopping(patience=15):
    return keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass
