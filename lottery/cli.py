"""命令行入口: predict / backtest / tune 三个子命令。"""

import argparse
import logging

from lottery import GAMES
from lottery.backtest import summarize, walk_forward, write_csv
from lottery.fetcher import fetch_draws
from lottery.runner import run_predict
from lottery.tuner import random_search


def to_asc(cfg):
    """抓取数据并转为时间升序。"""
    df, _ = fetch_draws(cfg)
    return df.iloc[::-1].reset_index(drop=True)


def cmd_predict(cfg, args):
    run_predict(cfg, rounds=args.rounds, window=args.window, epochs=args.epochs,
                batch=args.batch, temperature=args.temperature,
                min_hamming=args.min_hamming, seed=args.seed, stats=args.stats)


def cmd_backtest(cfg, args):
    import lottery.model as M
    M.set_seed(args.seed)
    df_asc = to_asc(cfg)
    results = walk_forward(df_asc, cfg, steps=args.steps, refit_every=args.refit_every,
                           window=args.window, epochs=args.epochs, batch=args.batch,
                           seed=args.seed, units=args.units, dropout=args.dropout,
                           lr=args.lr, stats=args.stats)
    summarize(cfg, results)
    write_csv(cfg, results)


def cmd_tune(cfg, args):
    import lottery.model as M
    M.set_seed(args.seed)
    df_asc = to_asc(cfg)
    random_search(df_asc, cfg, trials=args.trials, epochs=args.epochs, seed=args.seed)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='auto.py', description='双色球/大乐透 LSTM 预测工具包')
    parser.add_argument('command', nargs='?', default='predict',
                        choices=['predict', 'backtest', 'tune'],
                        help='predict 预测下一期 | backtest 滚动回测 | tune 随机搜索调参')
    parser.add_argument('--name', choices=list(GAMES), default='ssq', help='彩种')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度, 越小越接近argmax(默认1.0, predict/backtest 通用)')

    g_data = parser.add_argument_group('数据/模型')
    g_data.add_argument('--window', type=int, default=10, help='滑动窗口期数(默认10)')
    g_data.add_argument('--epochs', type=int, default=100, help='最大训练轮数(默认100)')
    g_data.add_argument('--batch', type=int, default=32, help='batch size(默认32)')
    g_data.add_argument('--units', type=int, default=64, help='LSTM 单元数(默认64)')
    g_data.add_argument('--dropout', type=float, default=0.2, help='dropout(默认0.2)')
    g_data.add_argument('--lr', type=float, default=1e-3, help='学习率(默认1e-3)')
    g_data.add_argument('--stats', action='store_true',
                        help='追加各球滚动频率统计特征(默认关闭)')

    g_pred = parser.add_argument_group('predict')
    g_pred.add_argument('--rounds', type=int, default=3, help='预测组数(默认3)')
    g_pred.add_argument('--min-hamming', type=int, default=2, dest='min_hamming',
                        help='预测组间最小红球差异, 0关闭(默认2)')

    g_bt = parser.add_argument_group('backtest')
    g_bt.add_argument('--steps', type=int, default=30, help='回测期数(默认30)')
    g_bt.add_argument('--refit-every', type=int, default=10, dest='refit_every',
                      help='每 N 期重训一次模型(默认10)')

    g_tune = parser.add_argument_group('tune')
    g_tune.add_argument('--trials', type=int, default=10, help='随机搜索 trial 数(默认10)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    cfg = GAMES[args.name]
    try:
        {'predict': cmd_predict, 'backtest': cmd_backtest, 'tune': cmd_tune}[args.command](cfg, args)
    except RuntimeError as e:
        # 已知失败(网络/解析/参数)给出清晰信息; 未知异常保留完整堆栈
        logging.error('%s', e)
        raise SystemExit(1)
