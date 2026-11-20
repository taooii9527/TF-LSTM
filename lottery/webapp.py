"""本地 Web 仪表盘: 数据总览 / 流程 / 回测 / 调优。

启动: python -m lottery.webapp  (浏览器打开 http://127.0.0.1:8765)

端点:
  GET  /            仪表盘页面 (web/)
  GET  /api/overview   数据总览(只读缓存 csv, 不访问网络)
  GET  /api/artifacts  回测/调优/最近预测产物(读落盘文件)
  GET  /api/status     当前任务状态 + 日志流
  POST /api/run        启动任务 predict|backtest|tune|fetch

设计约定: 任务执行完全复用 cli.cmd_* 与命令行同一代码路径,
Web 层只做编排(单任务互斥 + 日志采集 + 阶段推进), 不复制业务逻辑。
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 须在导入 tensorflow 前设置

import argparse
import json
import logging
import re
import threading
import time
from collections import Counter, deque

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import lottery.model as M
from lottery import GAMES
from lottery.cli import cmd_backtest, cmd_predict, cmd_tune
from lottery.dataset import window_input
from lottery.sampler import pattern_valid, sample_set

app = Flask(__name__)

# ---------------------------------------------------------------- 日志采集
LOG = deque(maxlen=600)
STAGES = ['数据抓取', '模型训练', '推理/评估', '结果落盘']


class DequeHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        LOG.append({'t': time.strftime('%H:%M:%S'), 'lvl': record.levelname, 'msg': msg})
        _set_stage(msg)


def _set_stage(msg):
    """按日志关键词推进任务阶段。"""
    s = 0
    if '抓取历史数据' in msg or '最新一期' in msg:
        s = 1
    elif '训练完成' in msg or 'refit @' in msg or 'trial ' in msg or ' EarlyStopping' in msg:
        s = 2
    elif '预测组' in msg or '回测汇总' in msg or '调优结果' in msg \
            or '最佳一期' in msg or '回测进度' in msg:
        s = 3
    elif '已写入' in msg or '已保存' in msg:
        s = 4
    if s:
        _state['stage'] = max(_state['stage'], s)


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', handlers=[DequeHandler()])

# ---------------------------------------------------------------- 任务编排
_state = {'running': False, 'kind': None, 'name': None, 'started': None,
          'stage': 0, 'error': None}
_task_lock = threading.Lock()


def _worker(kind, name, params):
    try:
        cfg = GAMES[name]
        if kind == 'fetch':
            from lottery.fetcher import fetch_draws
            df, _ = fetch_draws(cfg)
            df.to_csv(cfg.csv, index=False)
            logging.info('数据已更新: %s (%d 期) -> %s', name, len(df), cfg.csv)
        else:
            ns = argparse.Namespace(
                command=kind, name=name, seed=params['seed'],
                temperature=params['temperature'], window=params['window'],
                epochs=params['epochs'], batch=params['batch'],
                units=params.get('units', 64), dropout=params.get('dropout', 0.2),
                lr=params.get('lr', 1e-3), rounds=params.get('rounds', 3),
                min_hamming=params.get('min_hamming', 2),
                steps=params.get('steps', 30), refit_every=params.get('refit_every', 10),
                trials=params.get('trials', 10))
            {'predict': cmd_predict, 'backtest': cmd_backtest,
             'tune': cmd_tune}[kind](cfg, ns)
        _state['error'] = None
    except Exception as e:  # 线程内异常必须捕获, 否则状态卡死
        logging.error('%s: %s', type(e).__name__, e)
        _state['error'] = f'{type(e).__name__}: {e}'
    finally:
        _state['running'] = False
        _state['started'] = None


@app.post('/api/run')
def api_run():
    body = request.get_json(force=True, silent=True) or {}
    kind, name = body.get('kind'), body.get('name', 'ssq')
    if kind not in ('predict', 'backtest', 'tune', 'fetch'):
        return jsonify(ok=False, message='未知任务类型'), 400
    if name not in GAMES:
        return jsonify(ok=False, message='未知彩种'), 400
    if _state['running']:
        return jsonify(ok=False, message='已有任务在运行, 请等待完成'), 409

    params = {k: body.get(k, d) for k, d in
              dict(seed=42, temperature=1.0, window=10, epochs=100, batch=32).items()}
    for k in ('rounds', 'steps', 'refit_every', 'trials'):
        if body.get(k) is not None:
            params[k] = int(body[k])
    for k in ('epochs', 'window', 'batch', 'seed'):
        params[k] = int(params[k])

    LOG.clear()
    _state.update(running=True, kind=kind, name=name, started=time.time(),
                  stage=0, error=None)
    threading.Thread(target=_worker, args=(kind, name, params), daemon=True).start()
    return jsonify(ok=True)


@app.get('/api/status')
def api_status():
    return jsonify(running=_state['running'], kind=_state['kind'], name=_state['name'],
                   elapsed=(time.time() - _state['started']) if _state['started'] else 0,
                   stage=_state['stage'], stages=STAGES, error=_state['error'],
                   logs=list(LOG))


# ---------------------------------------------------------------- 数据总览
@app.get('/api/overview')
def api_overview():
    cfg = _cfg_or_none()
    if cfg is None:
        return _bad_game()
    if not os.path.exists(cfg.csv):
        return jsonify(ok=False, message='本地无缓存数据, 请先在「流程」页抓取数据')
    df = pd.read_csv(cfg.csv)  # 最新在前
    asc = df.iloc[::-1].reset_index(drop=True)
    n = len(df)

    sums = asc[cfg.reds].sum(axis=1)

    # 随机性体检: 单球频率在均匀随机下必然趋平(无信息), 改看组合结构
    # 奇偶/大小分布 vs 理论超几何期望, 以及连号率
    from math import comb
    red_vals = asc[cfg.reds].values
    r = cfg.red_count
    if cfg.name == 'ssq':
        size_cut, n_odd = 17, 16            # 1-33: 17 小 / 16 大, 16 奇 / 17 偶
    else:
        size_cut, n_odd = 18, 18            # 1-35: 18 小 / 17 大, 18 奇 / 17 偶
    odd_counts = (red_vals % 2).sum(axis=1)
    big_counts = (red_vals > size_cut).sum(axis=1)
    theo_odd = [comb(n_odd, k) * comb(cfg.red_max - n_odd, r - k) / comb(cfg.red_max, r)
                for k in range(r + 1)]
    theo_big = [comb(cfg.red_max - size_cut, k) * comb(size_cut, r - k) / comb(cfg.red_max, r)
                for k in range(r + 1)]
    has_consec = (np.diff(np.sort(red_vals, axis=1), axis=1) == 1).any(axis=1)

    return jsonify(ok=True, name=cfg.name, tag=cfg.tag, draws=n,
                   latest_no=int(df['No'].iloc[0]), latest_date=str(df['date'].iloc[0]),
                   randomness={
                       'parity': {'counts': [int((odd_counts == k).sum()) for k in range(r + 1)],
                                  'theo': [round(p * n, 1) for p in theo_odd]},
                       'size': {'counts': [int((big_counts == k).sum()) for k in range(r + 1)],
                                'theo': [round(p * n, 1) for p in theo_big]},
                       'consec_rate': round(float(has_consec.mean()), 3)},
                   sum_trend={'issues': [str(x) for x in asc['No'].tail(300)],
                              'sums': [int(x) for x in sums.tail(300)],
                              'roll20': [round(float(x), 1) for x in sums.rolling(20).mean().tail(300)]},
                   recent=[{'no': str(r['No']), 'date': str(r['date']),
                            'reds': [int(r[c]) for c in cfg.reds],
                            'blues': [int(r[c]) for c in cfg.blues]}
                           for _, r in df.head(50).iterrows()],
                   red_mean_sum=round(float(sums.mean()), 1))


# ---------------------------------------------------------------- 期望账本
from lottery.prizes import _prize_dlt, _prize_ssq, PRIZE_FN


@app.get('/api/ledger')
def api_ledger():
    """期望账本: 回测期内每期买 1 注(2元)的投入 vs 实际中奖。"""
    cfg = _cfg_or_none()
    if cfg is None:
        return _bad_game()
    path = f'backtest_{cfg.name}.csv'
    if not os.path.exists(path):
        return jsonify(ok=False, message='暂无回测明细, 请先运行 backtest')
    prize_fn = _prize_ssq if cfg.name == 'ssq' else _prize_dlt
    import csv as _csv
    with open(path, encoding='utf-8') as f:
        rows = list(_csv.DictReader(f))
    total, wins, best = 0, 0, None
    for row in rows:
        pr = set(map(int, row['pred_reds'].split()))
        pb = set(map(int, row['pred_blues'].split()))
        ar = set(map(int, row['actual_reds'].split()))
        ab = set(map(int, row['actual_blues'].split()))
        r, b = len(pr & ar), len(pb & ab)
        p = prize_fn(r, b)
        total += p
        if p:
            wins += 1
            if best is None or p > best['prize']:
                best = {'issue': row['issue'], 'red_hits': r, 'blue_hits': b, 'prize': p}
    steps = len(rows)
    invest = steps * 2
    return jsonify(ok=True, steps=steps, invest=invest, prize=total,
                   net=total - invest, ret_rate=round(total / invest, 3),
                   wins=wins, best=best)


# ---------------------------------------------------------------- 预测战绩
@app.get('/api/forecast-record')
def api_forecast_record():
    """predict.rec 历史预测 vs 实际开奖对照。"""
    cfg = _cfg_or_none()
    if cfg is None:
        return _bad_game()
    import re
    if not os.path.exists(cfg.record) or not os.path.exists(cfg.csv):
        return jsonify(ok=False, message='暂无预测记录')
    df = pd.read_csv(cfg.csv)
    drawn = {int(r['No']): ([int(r[c]) for c in cfg.reds], [int(r[c]) for c in cfg.blues])
             for _, r in df.iterrows()}
    issues = {}
    order = []
    with open(cfg.record, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'\[(\d+)\] #(\d+) red ([\d ]+) blue ([\d ]+)', line.strip())
            if not m:
                continue
            issue = int(m.group(1))
            if issue not in issues:
                issues[issue] = {'issue': issue, 'groups': [], 'drawn': False, 'hits': []}
                order.append(issue)
            reds = list(map(int, m.group(3).split()))
            blues = list(map(int, m.group(4).split()))
            if issue in drawn:
                ar, ab = drawn[issue]
                issues[issue]['drawn'] = True
                issues[issue]['hits'].append(
                    {'red': len(set(reds) & set(ar)), 'blue': len(set(blues) & set(ab))})
            issues[issue]['groups'].append({'reds': reds, 'blues': blues})
    recent = [issues[i] for i in order[-12:]][::-1]
    return jsonify(ok=True, tag=cfg.tag, record=recent)


# ---------------------------------------------------------------- 运行时健康
def _health_game(cfg):
    """单彩种深度自检: 静默退化在此显式化。"""
    import csv as _csv
    items = []

    def item(name, passed, detail=''):
        items.append({'name': name, 'passed': bool(passed), 'detail': str(detail)})

    if os.path.exists(cfg.csv):
        df = pd.read_csv(cfg.csv)
        latest = str(df['date'].iloc[0])
        days = (pd.Timestamp.now() - pd.Timestamp(latest)).days
        item('数据完整性', len(df) >= 500, f'{len(df)} 期')
        item('数据新鲜度', days <= 5, f'最新开奖 {latest}' if days <= 5
             else f'最新开奖 {latest}, 已 {days} 天未更新')
    else:
        item('数据文件', False, f'{cfg.csv} 缺失, 先运行 fetch')

    lp = _last_predict(cfg)
    n_groups = len(lp['groups']) if lp else 0
    item('预测产物', n_groups >= 1,
         f'{n_groups} 组可解析' if n_groups else 'predict.rec 无可解析预测组')

    ok_model = os.path.exists(cfg.model) and os.path.getsize(cfg.model) > 100_000
    item('模型文件', ok_model,
         f'{os.path.getsize(cfg.model) // 1024} KB' if ok_model else f'{cfg.model} 缺失')

    bt_path = f'backtest_{cfg.name}.csv'
    if os.path.exists(bt_path):
        with open(bt_path, encoding='utf-8') as f:
            rows = list(_csv.DictReader(f))
        base_p = cfg.red_count * cfg.red_count / cfg.red_max
        hits = [len(set(map(int, r['pred_reds'].split())) &
                    set(map(int, r['actual_reds'].split()))) for r in rows]
        mean = sum(hits) / len(hits)
        se = float(np.std(hits, ddof=1)) / len(rows) ** 0.5 if len(rows) > 1 else 0.0
        dev = mean - base_p
        item('回测统计哨兵', abs(dev) <= 3 * se,
             f'偏差 {dev:+.3f} (±{3 * se:.3f})' if abs(dev) <= 3 * se
             else f'红球偏差 {dev:+.3f} 超出 3SE — 疑似泄漏/参数伪象')
    else:
        item('回测统计哨兵', True, '暂无回测明细(跳过)')

    return {'game': cfg.name, 'ok': all(i['passed'] for i in items), 'items': items}


@app.get('/api/health')
def api_health():
    checks = [_health_game(c) for c in GAMES.values()]
    return jsonify(ok=all(c['ok'] for c in checks), checks=checks)


@app.get('/api/analysis')
def api_analysis():
    """下期分析: 热温冷/遗漏(近30期统计) + 模型概率聚合(形态参考/推荐指数)。
    遵循彩票门户惯例: 展示号码集与形态, 而非逐球位原始概率。
    """
    cfg = _cfg_or_none()
    if cfg is None:
        return _bad_game()
    if not os.path.exists(cfg.csv):
        return jsonify(ok=False, message='本地无缓存数据, 请先抓取')
    df = pd.read_csv(cfg.csv)
    asc = df.iloc[::-1].reset_index(drop=True)
    vals = asc[cfg.ball_cols].values

    def heat_stats(maxv, cols):
        recent = asc[cols].tail(30).values
        all_mask = asc[cols].values
        out = []
        for b in range(1, maxv + 1):
            freq = int((recent == b).any(axis=1).sum())
            hit_rows = np.nonzero((all_mask == b).any(axis=1))[0]
            omi = int(len(asc) - 1 - hit_rows[-1]) if len(hit_rows) else int(len(asc))
            level = 'hot' if freq >= 7 else ('warm' if freq >= 4 else 'cold')
            out.append({'ball': b, 'freq': freq, 'omi': omi, 'level': level})
        return out

    analysis = {
        'ok': True, 'tag': cfg.tag, 'next_no': int(df['No'].iloc[0]) + 1,
        'red_heat': heat_stats(cfg.red_max, cfg.reds),
        'blue_heat': heat_stats(cfg.blue_max, cfg.blues),
        'model': None,
    }

    if os.path.exists(cfg.model):
        model = M.load_model(cfg)
        # 依据已存模型的输入维自适应: 是否用统计特征训练的
        stats = model.input_shape[-1] != cfg.n_balls
        X_next = window_input(vals, cfg, 10, stats=stats)
        # 保持 predict 原始输出 (1, N) 形状: sampler 内部取 head[0]
        head_probs = dict(zip(cfg.ball_cols, model.predict(X_next, verbose=0)))
        # 蒙特卡洛: 从模型概率采样 500 组合法号码, 聚合形态分布
        rng = np.random.default_rng(42)
        n_sample, red_freq = 500, np.zeros(cfg.red_max + 1)
        sums, odds, bigs = [], [], []
        red_mid = (cfg.red_max + 1) / 2
        while len(sums) < n_sample and len(sums) < 5000:
            r, _, _ = sample_set(cfg, head_probs, 1.0, rng)
            if not pattern_valid(cfg, r):
                continue
            for v in r:
                red_freq[v] += 1
            sums.append(sum(r))
            odds.append(sum(1 for v in r if v % 2))
            bigs.append(sum(1 for v in r if v >= red_mid))
        n_ok = len(sums)
        lo, hi = np.percentile(sums, [25, 75])
        blue_p = np.mean([np.asarray(head_probs[c]).ravel() for c in cfg.blues], axis=0)
        analysis['model'] = {
            'red_top': sorted(
                ({'ball': i, 'score': round(float(red_freq[i] / n_ok), 3)}
                 for i in range(1, cfg.red_max + 1)),
                key=lambda d: -d['score'])[:8],
            'blue_top': sorted(
                ({'ball': i + 1, 'p': round(float(p), 3)}
                 for i, p in enumerate(blue_p)),
                key=lambda d: -d['p'])[:3],
            'sum_range': [round(float(lo)), round(float(hi))],
            'odd_ratio': Counter(f'{o}:{cfg.red_count - o}'
                                 for o in odds).most_common(1)[0][0],
            'big_ratio': Counter(f'{g}:{cfg.red_count - g}'
                                 for g in bigs).most_common(1)[0][0],
            'samples': n_ok,
        }
    return jsonify(analysis)


# ---------------------------------------------------------------- 产物读取
_PRED_RE = re.compile(r'\[(\d+)\] #(\d+) red ([\d ]+) blue ([\d ]+) \[SCORE\] (-?[\d.]+)')


def _cfg_or_none():
    name = request.args.get('name', 'ssq')
    return GAMES[name] if name in GAMES else None


def _bad_game():
    return jsonify(ok=False, message=f"未知彩种: {request.args.get('name')}"), 400


def _last_predict(cfg):
    if not os.path.exists(cfg.record):
        return None
    blocks, cur = [], None
    with open(cfg.record, encoding='utf-8') as f:
        for line in f:
            if '------------' in line and cfg.tag in line:
                cur = {'time': line.split(' ')[0], 'last_sale': '', 'groups': []}
                blocks.append(cur)
            elif cur is not None and 'LAST SALE' in line:
                m = re.match(r'LAST SALE \[(.*?)\]\[(.*?)\]', line)
                if m:
                    cur['last_sale'] = {'date': m.group(1), 'no': m.group(2)}
            elif cur is not None:
                m = _PRED_RE.match(line.strip())
                if m:
                    cur['groups'].append({'no': m.group(1), 'idx': int(m.group(2)),
                                          'reds': [int(v) for v in m.group(3).split()],
                                          'blues': [int(v) for v in m.group(4).split()],
                                          'score': float(m.group(5))})
    return blocks[-1] if blocks else None


@app.get('/api/artifacts')
def api_artifacts():
    cfg = _cfg_or_none()
    if cfg is None:
        return _bad_game()
    out = {'ok': True, 'name': cfg.name, 'tag': cfg.tag,
           'predict': _last_predict(cfg), 'backtest': None, 'tune': None}

    bt_path = f'backtest_{cfg.name}.csv'
    if os.path.exists(bt_path):
        bt = pd.read_csv(bt_path)
        reds, blues = bt['red_hits'].tolist(), bt['blue_hits'].tolist()
        steps = len(bt)
        se = float(np.std(reds) / np.sqrt(steps)) if steps > 1 else 0.0
        best_i = int((bt['red_hits'] * 10 + bt['blue_hits']).idxmax())
        b = bt.iloc[best_i]
        out['backtest'] = {
            'steps': steps,
            'mean_red': round(float(np.mean(reds)), 3),
            'mean_blue': round(float(np.mean(blues)), 3),
            'exp_red': cfg.expected_red_hits, 'exp_blue': cfg.expected_blue_hits,
            'deviation': round(float(np.mean(reds)) - cfg.expected_red_hits, 3),
            'se': round(se, 3),
            'best': {'issue': str(b['issue']), 'red_hits': int(b['red_hits']),
                     'blue_hits': int(b['blue_hits']),
                     'pred_reds': [int(v) for v in str(b['pred_reds']).split()],
                     'pred_blues': [int(v) for v in str(b['pred_blues']).split()],
                     'actual_reds': [int(v) for v in str(b['actual_reds']).split()],
                     'actual_blues': [int(v) for v in str(b['actual_blues']).split()]},
            'red_dist': dict(Counter(reds)), 'blue_dist': dict(Counter(blues)),
            'rows': [{'issue': str(r['issue']), 'red_hits': int(r['red_hits']),
                      'blue_hits': int(r['blue_hits']),
                      'pred_reds': [int(v) for v in str(r['pred_reds']).split()],
                      'actual_reds': [int(v) for v in str(r['actual_reds']).split()]}
                     for _, r in bt.iterrows()]}

    tune_path = f'tune_{cfg.name}.json'
    if os.path.exists(tune_path):
        with open(tune_path, encoding='utf-8') as f:
            out['tune'] = json.load(f)
    return jsonify(out)


# ---------------------------------------------------------------- 静态页面
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')


@app.get('/')
def index():
    return send_from_directory(_WEB_DIR, 'index.html')


@app.get('/<path:p>')
def static_files(p):
    return send_from_directory(_WEB_DIR, p)


if __name__ == '__main__':
    print('仪表盘: http://127.0.0.1:8765')
    app.run(host='127.0.0.1', port=8765, threaded=True)
