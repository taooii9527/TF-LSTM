"""开奖时刻表驱动的自动预测/校对守护进程。

运行:
  python -m lottery.scheduler                     # 前台常驻
  Start-Process python -ArgumentList '-m','lottery.scheduler' -WindowStyle Hidden

循环(每 60s), 双色球 周二/四/日 21:15、大乐透 周一/三/六 21:25:
  预测窗  开奖前 45min 内 且 本期未预测 -> auto.py predict -> predict.rec
  校对窗  开奖后 40min~3h 内, 每 10min 拉一次数据; 有新开奖即校对
          predict.rec 中全部未核对条目, 命中与奖金落盘 verify.csv
幂等: scheduler_state.json 记录各彩种 last_predicted/last_verified 期号。
判定逻辑为纯函数(should_predict/judge), 由 tests/test_scheduler.py 覆盖。
"""

import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lottery.config import GAMES          # noqa: E402
from lottery.prizes import PRIZE_FN       # noqa: E402

DRAW_DAYS = {'ssq': {1, 3, 6}, 'dlt': {0, 2, 5}}   # weekday: Mon=0
DRAW_TIME = {'ssq': (21, 15), 'dlt': (21, 25)}     # 官方开奖时刻
PREDICT_AHEAD = 45                                  # 开奖前 45min 出预测
VERIFY_AFTER = 40                                   # 开奖后 40min 起可校对
VERIFY_SPAN = 6                                      # 校对窗口 6h(覆盖数据源延迟)
FETCH_INTERVAL = 600                                # 校对窗内拉数据间隔 10min

STATE_PATH = ROOT / 'scheduler_state.json'
VERIFY_PATH = ROOT / 'verify.csv'
PRED_RE = re.compile(r'\[(\d+)\] #(\d+) red ([\d ]+) blue ([\d ]+)')

log = logging.getLogger('scheduler')


# ---------------------------------------------------------------- 纯函数(可测)
def draw_moment(cfg, now):
    """当天开奖时刻; 非开奖日返回 None。"""
    if now.weekday() not in DRAW_DAYS[cfg.name]:
        return None
    h, m = DRAW_TIME[cfg.name]
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


def should_predict(cfg, now, latest_issue, done_issue):
    """返回预测目标期号; 不该预测返回 None。"""
    dm = draw_moment(cfg, now)
    if dm is None or not (dm - timedelta(minutes=PREDICT_AHEAD) <= now < dm):
        return None
    target = latest_issue + 1
    return None if done_issue == target else target


def judge(cfg, reds, blues, drawn_reds, drawn_blues):
    """单注校对: 命中数与对应奖级奖金。"""
    r = len(set(reds) & set(drawn_reds))
    b = len(set(blues) & set(drawn_blues))
    return {'red_hits': r, 'blue_hits': b, 'prize': PRIZE_FN[cfg.name](r, b)}


def parse_rec(cfg, all_blocks=False):
    """解析 predict.rec 中本彩种的预测条目。

    默认返回最后一个块(每期以开奖前最后一次预测为准);
    all_blocks=True 返回全部块(校对用, 确保错过窗口的旧期也能补校)。
    """
    path = ROOT / cfg.record
    if not path.exists():
        return []
    blocks, cur, in_block = [], None, False
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if '------------' in s:  # 块头行, 如 "----------SSQ------------"
            if cur is not None:
                blocks.append(cur)
                cur = None
            in_block = cfg.tag in s
            continue
        if not in_block:
            continue
        m = PRED_RE.match(s)
        if m:
            if cur is None:
                cur = []
            cur.append({'issue': int(m.group(1)), 'idx': int(m.group(2)),
                        'reds': [int(v) for v in m.group(3).split()],
                        'blues': [int(v) for v in m.group(4).split()]})
    if cur is not None:
        blocks.append(cur)
    if all_blocks:
        seen, merged = set(), []
        for blk in blocks:            # 同 (issue, idx) 保留最后一次出现
            for e in blk:
                key = (e['issue'], e['idx'])
                if key not in seen:
                    seen.add(key)
                    merged.append(e)
        return merged
    return blocks[-1] if blocks else []


# ---------------------------------------------------------------- 状态
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    return {'predicted': {n: 0 for n in GAMES},
            'verified': {n: 0 for n in GAMES}, 'last_fetch': {}}


def save_state(st):
    STATE_PATH.write_text(json.dumps(st), encoding='utf-8')


# ---------------------------------------------------------------- 动作
def local_latest_issue(cfg):
    import pandas as pd
    csv = ROOT / cfg.csv
    if not csv.exists():
        return 0
    return int(pd.read_csv(csv)['No'].iloc[0])


def do_predict(cfg, target):
    log.info('[%s] 预测第 %s 期...', cfg.tag, target)
    r = subprocess.run([sys.executable, 'auto.py', '--name', cfg.name, '--seed', '42'],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    ok = r.returncode == 0
    log.info('[%s] 预测%s', cfg.tag, '完成' if ok else f'失败: {(r.stderr or "")[-200:]}')
    return ok


def fetch_draws_df(cfg):
    from lottery.fetcher import fetch_draws
    df, _ = fetch_draws(cfg)
    return df  # 最新在前


def do_verify(cfg, st):
    """校对 rec 中所有已开奖未核对的条目, 追加 verify.csv。"""
    import pandas as pd
    df = fetch_draws_df(cfg)
    drawn = {int(r['No']): ([int(r[c]) for c in cfg.reds],
                            [int(r[c]) for c in cfg.blues])
             for _, r in df.iterrows()}
    done = set()
    if VERIFY_PATH.exists():
        v = pd.read_csv(VERIFY_PATH)
        done = set(zip(v['game'], v['issue'], v['idx']))

    rows = []
    for e in parse_rec(cfg, all_blocks=True):  # 全部历史块, 漏校的旧期补校
        if e['issue'] not in drawn or (cfg.name, e['issue'], e['idx']) in done:
            continue
        dr, db = drawn[e['issue']]
        j = judge(cfg, e['reds'], e['blues'], dr, db)
        rows.append({'game': cfg.name, 'issue': e['issue'], 'idx': e['idx'],
                     'reds': ' '.join(map(str, e['reds'])),
                     'blues': ' '.join(map(str, e['blues'])),
                     **j, 'actual': ' '.join(map(str, dr + db)),
                     'checked_at': datetime.now().isoformat(timespec='seconds')})
    if rows:
        header = not VERIFY_PATH.exists()
        pd.DataFrame(rows).to_csv(VERIFY_PATH, mode='a',
                                  header=header, index=False, encoding='utf-8')
        for r in rows:
            log.info('[%s] 校对 %s期 #%d: 红%d 蓝%d 奖%d元',
                     cfg.tag, r['issue'], r['idx'], r['red_hits'], r['blue_hits'], r['prize'])
    latest = int(df['No'].iloc[0])
    if latest > st['verified'][cfg.name]:
        st['verified'][cfg.name] = latest
    return len(rows)


# ---------------------------------------------------------------- 主循环
def tick(cfg, st, now=None):
    """单彩种单轮检查; 返回本轮动作说明。"""
    now = now or datetime.now()
    dm = draw_moment(cfg, now)
    if dm is None:
        return '非开奖日'

    # 预测窗
    target = should_predict(cfg, now, local_latest_issue(cfg), st['predicted'][cfg.name])
    if target:
        if do_predict(cfg, target):
            st['predicted'][cfg.name] = target
            save_state(st)
        return f'已预测 {target} 期'

    # 校对窗
    lo, hi = dm + timedelta(minutes=VERIFY_AFTER), dm + timedelta(hours=VERIFY_SPAN)
    if lo <= now <= hi:
        last = st['last_fetch'].get(cfg.name, 0)
        if time.time() - last >= FETCH_INTERVAL:
            st['last_fetch'][cfg.name] = time.time()
            save_state(st)
            try:
                n = do_verify(cfg, st)
                save_state(st)
                return f'校对完成, 新增 {n} 条' if n else '暂无新开奖'
            except Exception as e:  # 数据源未更新属常态, 下一轮重试
                log.warning('[%s] 校对暂缓: %s', cfg.tag, e)
                return f'校对暂缓: {e}'
    return '等待'


def run_round(st):
    """全部彩种各检查一轮(本地循环与 GitHub Actions --once 共用)。"""
    for cfg in GAMES.values():
        try:
            log.info('[%s] %s', cfg.tag, tick(cfg, st))
        except Exception as e:
            log.error('[%s] 本轮异常: %s', cfg.tag, e)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--once', action='store_true',
                    help='运行一轮检查后退出(GitHub Actions 定时驱动)')
    args = ap.parse_args()
    logging.basicConfig(filename=str(ROOT / 'scheduler.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    log.info('调度启动%s: %s', '(once)' if args.once else '', list(GAMES))
    st = load_state()
    run_round(st)
    if not args.once:
        while True:
            time.sleep(60)
            run_round(st)


if __name__ == '__main__':
    main()
