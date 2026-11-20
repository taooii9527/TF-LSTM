"""训练与预测主流程。"""

import logging
import time

import numpy as np

import lottery.model as M
from lottery.dataset import build_dataset, window_input
from lottery.fetcher import fetch_draws
from lottery.sampler import diverse_candidates


def train_model(cfg, df_asc, window, epochs, batch, units=64, dropout=0.2, lr=1e-3,
                verbose=2, patience=15, stats=False):
    """在时间升序数据上训练，返回 (model, best val_loss)。验证集为时间后 20%。"""
    ds = build_dataset(df_asc, cfg, window, stats=stats)
    X_tr, X_va, y_tr, y_va = ds.train_val()
    model = M.build_model(cfg, window, units=units, dropout=dropout, lr=lr,
                          in_dim=ds.X.shape[-1])
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=epochs, batch_size=batch, verbose=verbose,
                     callbacks=[M.early_stopping(patience)])
    val_loss = float(min(hist.history['val_loss']))
    logging.info('训练完成: %.1fs, best val_loss=%.4f', time.time() - t0, val_loss)
    return model, val_loss


def run_predict(cfg, rounds, window, epochs, batch, temperature, min_hamming, seed,
                stats=False):
    """预测主流程: 抓取 -> 落盘 csv -> 训练 -> 存模型 -> 采样 N 组 -> 写记录。"""
    M.set_seed(seed)

    # 1. 数据抓取(最新在前) -> 落盘 csv(兼容旧格式) -> 转为时间升序
    df, next_no = fetch_draws(cfg)
    df.to_csv(cfg.csv, index=False)
    latest = df.iloc[0]
    balls_str = ' '.join(f'{latest[c]:02d}' for c in cfg.ball_cols)
    logging.info('最新一期: %s期 开奖日期[%s] 销售额[%s] 号码: %s',
                 latest['No'], latest['date'], latest['sale'], balls_str)
    df_asc = df.iloc[::-1].reset_index(drop=True)

    # 2. 训练一次模型
    model, _ = train_model(cfg, df_asc, window, epochs, batch, stats=stats)
    path = M.save_model(model, cfg)
    logging.info('模型已保存: %s', path)

    # 3. 用最近 window 期(含最新一期)预测下一期
    vals = df_asc[cfg.ball_cols].values
    X_next = window_input(vals, cfg, window, stats=stats)
    head_probs = dict(zip(cfg.ball_cols, model.predict(X_next, verbose=0)))

    # 4. 温度采样生成互不趋同的多组预测
    results = diverse_candidates(cfg, head_probs, rounds, temperature,
                                 min_hamming, seed)
    for i, (reds_out, blues_out, _) in enumerate(results, 1):
        logging.info('预测组 %d: red %s blue %s', i, reds_out, blues_out)

    # 5. 写入 predict.rec
    with open(cfg.record, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ----------{cfg.tag}------------\n")
        f.write(f"LAST SALE [{latest['date']}][{latest['No']}][{latest['sale']}] --- {balls_str}\n")
        for i, (reds_out, blues_out, score) in enumerate(results, 1):
            red_str = ' '.join(f'{v:02d}' for v in reds_out)
            blue_str = ' '.join(f'{v:02d}' for v in blues_out)
            f.write(f'[{next_no}] #{i} red {red_str} blue {blue_str} [SCORE] {score:.4f}\n')
    logging.info('结果已写入 %s, 下一期期号: %d', cfg.record, next_no)
