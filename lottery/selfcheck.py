"""一键自检入口: python -m lottery.selfcheck [--fast]

三层检查(标准库实现, 不依赖 pytest):
  L1 语法    compileall 全模块
  L3 端到端  1-epoch 全链路训练预测 + 产物 round-trip 自证
  L3 API契约 flask test_client 遍历全部端点, 断言 schema 与关键不变量

设计原则: 写进去的必须读得回来(round-trip); 空结果必须显式失败。
"""

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def report(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'[{tag}] {name}' + (f' — {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def check_syntax():
    print('== L1 语法检查 ==')
    ok = compileall.compile_dir(str(ROOT / 'lottery'), quiet=1, force=True)
    ok &= compileall.compile_file(str(ROOT / 'auto.py'), quiet=1, force=True)
    report('compileall 全模块', bool(ok))


def check_e2e(name):
    """1-epoch 全链路: 训练→预测→落盘→读回, round-trip 自证。"""
    print(f'== L3 端到端冒烟 ({name}) ==')
    r = subprocess.run(
        [sys.executable, 'auto.py', '--name', name, '--epochs', '1',
         '--rounds', '2', '--seed', '42'],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if not report('predict 全链路退出码', r.returncode == 0,
                  (r.stderr or '')[-200:] if r.returncode else ''):
        return
    from lottery import webapp as W
    from lottery.config import GAMES
    cfg = GAMES[name]
    lp = W._last_predict(cfg)
    if not report('predict.rec round-trip', bool(lp and lp['groups']),
                  '最后一块无可解析预测组'):
        return
    groups = lp['groups']
    report('预测组数', len(groups) == 2, f'{len(groups)} 组')
    ok = all(len(set(g['reds'])) == cfg.red_count and
             all(1 <= v <= cfg.red_max for v in g['reds']) and
             all(1 <= v <= cfg.blue_max for v in g['blues'])
             for g in groups)
    report('号码合法性与唯一性', ok)


def check_api_contracts():
    print('== L3 API 契约 ==')
    from lottery import webapp as W
    client = W.app.test_client()

    def get(path):
        r = client.get(path)
        assert r.status_code == 200, f'{path} -> {r.status_code}'
        return r.get_json()

    for name in ('ssq', 'dlt'):
        d = get(f'/api/overview?name={name}')
        report(f'overview[{name}]', d.get('ok') and d.get('draws', 0) > 500 and
               all(k in d.get('randomness', {}) for k in ('parity', 'size', 'consec_rate')),
               f"draws={d.get('draws')}")

        d = get(f'/api/analysis?name={name}')
        m = d.get('model') or {}
        report(f'analysis[{name}]', d.get('ok') and
               len(d.get('red_heat', [])) > 0 and
               {'red_top', 'blue_top', 'sum_range', 'odd_ratio', 'big_ratio'} <= set(m),
               f"red_heat={len(d.get('red_heat', []))}")

        d = get(f'/api/forecast-record?name={name}')
        report(f'forecast-record[{name}]', 'ok' in d and 'record' in d,
               f"{len(d.get('record', []))} 条历史")

        d = get(f'/api/artifacts?name={name}')
        lp = (d.get('predict') or {}).get('groups', [])
        report(f'artifacts[{name}] 预测非空', d.get('ok') and len(lp) >= 1,
               f'{len(lp)} 组' if lp else 'groups 为空')

        d = get(f'/api/ledger?name={name}')
        ok = d.get('ok') and d['invest'] == d['steps'] * 2 and \
            abs(d['ret_rate'] - d['prize'] / d['invest']) < 1e-9
        report(f'ledger[{name}] 账目自洽', ok,
               f"steps={d.get('steps')} ret={d.get('ret_rate')}")

    d = get('/api/health')
    report('health', 'checks' in d and len(d['checks']) >= 2, f"ok={d.get('ok')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='跳过训练冒烟')
    args = ap.parse_args()

    check_syntax()
    if not args.fast:
        for name in ('ssq', 'dlt'):
            check_e2e(name)
    check_api_contracts()

    print('=' * 40)
    if FAILS:
        print(f'自检失败 {len(FAILS)} 项: {", ".join(FAILS)}')
        sys.exit(1)
    print('全部自检通过')


if __name__ == '__main__':
    main()
