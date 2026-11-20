"""walk-forward 滚动回测。

对最近 steps 期逐期评估: 每期只用该期之前的数据训练(refit_every 控制重训频率),
预测当期号码并与实际开奖比对, 统计命中数并与随机选号期望基线对比。
彩票为独立随机事件, 回测的意义在于验证建模/采样链路无泄漏、无系统性偏差,
而不是证明模型有预测能力。
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

import lottery.model as M
from lottery.dataset import build_dataset, window_input
from lottery.sampler import pattern_valid, sample_set


@dataclass
class StepResult:
    issue: int
    pred_reds: list
    pred_blues: list
    actual_reds: list
    actual_blues: list
    red_hits: int
    blue_hits: int


def _fit_prefix(df_asc, cfg, t, window, epochs, batch, units, dropout, lr, patience=10,
                stats=False):
    """只用第 t 期之前的数据训练(内部再按 80/20 时间切分出验证集)。"""
    ds = build_dataset(df_asc.iloc[:t], cfg, window, stats=stats)
    X_tr, X_va, y_tr, y_va = ds.train_val()
    model = M.build_model(cfg, window, units=units, dropout=dropout, lr=lr,
                          in_dim=ds.X.shape[-1])
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch, verbose=0,
              callbacks=[M.early_stopping(patience)])
    return model


def walk_forward(df_asc, cfg, steps, refit_every, window, epochs, batch,
                 seed, temperature=1.0, units=64, dropout=0.2, lr=1e-3, stats=False):
    """对最后 steps 期做滚动预测回测，返回 StepResult 列表。
    temperature 与 predict 保持一致(默认1.0): 模型偏好无信息量,
    过低温度的过度集中会人为压低命中率。
    """
    vals = df_asc[cfg.ball_cols].values
    n = len(vals)
    if steps >= n - window:
        raise RuntimeError(f'steps({steps}) 过大, 可回测期数最多 {n - window - 1}')
    start = n - steps
    model, results = None, []

    for t in range(start, n):
        if model is None or (t - start) % refit_every == 0:
            t0 = time.time()
            model = _fit_prefix(df_asc, cfg, t, window, epochs, batch,
                                units, dropout, lr, stats=stats)
            logging.info('refit @ 第%d期 (%.1fs)', df_asc['No'].iloc[t], time.time() - t0)

        X_t = window_input(vals[:t], cfg, window, stats=stats)
        head_probs = dict(zip(cfg.ball_cols, model.predict(X_t, verbose=0)))
        rng = np.random.default_rng(seed + t)
        pred_reds, pred_blues = None, None
        for _ in range(50):  # 集合级合理性约束, 与 predict 同标准
            pred_reds, pred_blues, _ = sample_set(cfg, head_probs, temperature, rng)
            if pattern_valid(cfg, pred_reds):
                break

        actual = vals[t].astype(int)
        actual_reds = sorted(actual[:cfg.red_count].tolist())
        actual_blues = sorted(actual[cfg.red_count:].tolist())
        results.append(StepResult(
            issue=int(df_asc['No'].iloc[t]),
            pred_reds=pred_reds, pred_blues=pred_blues,
            actual_reds=actual_reds, actual_blues=actual_blues,
            red_hits=len(set(pred_reds) & set(actual_reds)),
            blue_hits=len(set(pred_blues) & set(actual_blues)),
        ))
        logging.info('回测进度 %d/%d (期%d: 红%d 蓝%d)',
                     len(results), steps, results[-1].issue,
                     results[-1].red_hits, results[-1].blue_hits)
    return results


def summarize(cfg, results):
    """打印命中率汇总并与随机基线对比，返回摘要 dict。"""
    red_hits = [r.red_hits for r in results]
    blue_hits = [r.blue_hits for r in results]
    steps = len(results)
    mean_red, mean_blue = float(np.mean(red_hits)), float(np.mean(blue_hits))
    exp_red, exp_blue = cfg.expected_red_hits, cfg.expected_blue_hits

    logging.info('--------- 回测汇总 (%d 期) ---------', steps)
    logging.info('红球命中: 均值 %.3f (随机基线 %.3f)  分布 %s',
                 mean_red, exp_red, dict(sorted(Counter(red_hits).items())))
    logging.info('蓝球命中: 均值 %.3f (随机基线 %.3f)  分布 %s',
                 mean_blue, exp_blue, dict(sorted(Counter(blue_hits).items())))
    best = max(results, key=lambda r: r.red_hits * 10 + r.blue_hits)
    logging.info('最佳一期: %s期 命中 红%d 蓝%d | 预测 %s %s | 实际 %s %s',
                 best.issue, best.red_hits, best.blue_hits,
                 best.pred_reds, best.pred_blues, best.actual_reds, best.actual_blues)
    # 差异的标准误差, 粗略判断是否显著偏离随机
    se_red = float(np.std(red_hits) / np.sqrt(steps)) if steps > 1 else 0.0
    logging.info('红球偏差: %+.3f ± %.3f (|偏差|<2SE 内视为随机波动)',
                 mean_red - exp_red, se_red)
    return {'steps': steps, 'mean_red_hits': mean_red, 'expected_red_hits': exp_red,
            'mean_blue_hits': mean_blue, 'expected_blue_hits': exp_blue,
            'red_deviation': mean_red - exp_red, 'red_deviation_se': se_red}


def write_csv(cfg, results):
    """逐期明细落盘 backtest_{name}.csv。"""
    import csv
    path = f"backtest_{cfg.name}.csv"
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['issue', 'pred_reds', 'pred_blues', 'actual_reds',
                    'actual_blues', 'red_hits', 'blue_hits'])
        for r in results:
            w.writerow([r.issue,
                        ' '.join(map(str, r.pred_reds)), ' '.join(map(str, r.pred_blues)),
                        ' '.join(map(str, r.actual_reds)), ' '.join(map(str, r.actual_blues)),
                        r.red_hits, r.blue_hits])
    logging.info('逐期明细已写入 %s', path)
