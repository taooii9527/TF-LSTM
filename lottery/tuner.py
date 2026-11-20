"""超参数随机搜索调优。

以时间后 20% 的验证集 val_loss 为目标做随机搜索。
评估一次 ≈ 一次训练(30-60s), 随机搜索在少量 trial 下性价比优于网格搜索。
注意: 由于彩票无可学习信号, 调优收敛到的主要是"数值上更平滑"的配置,
不会带来真实预测能力提升。
"""

import json
import logging
import random

from lottery.dataset import build_dataset
from lottery.runner import train_model

SEARCH_SPACE = {
    'window': [5, 8, 10, 15, 20],
    'units': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3, 0.5],
    'lr': [1e-4, 3e-4, 1e-3, 3e-3],
    'batch': [32, 64, 128],
}


def sample_params(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def random_search(df_asc, cfg, trials, epochs, seed):
    """返回按 val_loss 升序的 (params, val_loss) 列表并落盘 JSON。"""
    rng = random.Random(seed)
    tried = []
    for i in range(trials):
        params = sample_params(rng)
        ds = build_dataset(df_asc, cfg, params['window'])
        if ds.split < 10 or len(ds.X) - ds.split < 5:
            logging.warning('trial %d 参数 %s 样本不足, 跳过', i + 1, params)
            continue
        try:
            _, val_loss = train_model(
                cfg, df_asc, params['window'], epochs, params['batch'],
                units=params['units'], dropout=params['dropout'], lr=params['lr'],
                verbose=0, patience=10)
        except Exception as e:  # 训练发散(如 lr 过大)不中断搜索
            logging.warning('trial %d 参数 %s 训练失败: %s', i + 1, params, e)
            continue
        tried.append({**params, 'val_loss': val_loss})
        logging.info('trial %d/%d: %s -> val_loss=%.4f', i + 1, trials, params, val_loss)

    tried.sort(key=lambda d: d['val_loss'])
    if tried:
        logging.info('--------- 调优结果 (top %d) ---------', min(5, len(tried)))
        for d in tried[:5]:
            logging.info('val_loss=%.4f  %s', d['val_loss'],
                         {k: v for k, v in d.items() if k != 'val_loss'})
        path = f"tune_{cfg.name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tried, f, ensure_ascii=False, indent=2)
        logging.info('全部 trial 已写入 %s', path)
    return tried
