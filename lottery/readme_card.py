"""GitHub 首页(README)展示渲染: 最新预测 SVG 卡片 + 兑奖战绩表格。

python -m lottery.readme_card
读 predict.rec(各彩种最后块)与 verify.csv, 替换 README.md 中
<!-- LOTTERY:START --> ~ <!-- LOTTERY:END --> 标记区块, 并生成 cards/*.svg。
由 GitHub Actions 在每轮调度后调用, commit 回仓库实现首页自动更新。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lottery.config import GAMES           # noqa: E402
from lottery.scheduler import parse_rec    # noqa: E402

RED, BLUE = '#d5281f', '#1f6fd5'
BASELINE = {'ssq': '红 1.09 · 蓝 0.06', 'dlt': '红 0.71 · 蓝 0.33'}
START, END = '<!-- LOTTERY:START -->', '<!-- LOTTERY:END -->'
LABELS = {'ssq': '双色球', 'dlt': '大乐透'}


def svg_card(cfg, entry):
    """最新预测的号码球卡片 (红区上, 蓝/后区下)。"""
    balls = [(n, RED) for n in entry['reds']] + [(n, BLUE) for n in entry['blues']]
    w = 18 + len(entry['reds']) * 34 + 10 + len(entry['blues']) * 34
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="64">']
    x = 18
    for n, color in balls:
        parts.append(f'<circle cx="{x}" cy="{20 if color == RED else 44}" r="14" '
                     f'fill="{color}"/>'
                     f'<text x="{x}" y="{24 if color == RED else 48}" text-anchor="middle" '
                     f'font-size="12" fill="#fff" font-family="monospace">{n:02d}</text>')
        x += 34 if color == RED else 30
    parts.append('</svg>')
    out = ROOT / 'cards'
    out.mkdir(exist_ok=True)
    path = out / f'{cfg.name}.svg'
    path.write_text(''.join(parts), encoding='utf-8')
    return path


def fmt_draw(row):
    """verify.csv 一行的预测与开奖号码展示(全角｜, 避免撕裂表格)。"""
    return (row['reds'] + ' ｜ ' + row['blues'],
            ' '.join(row['actual'].split()[:len(row['reds'].split())]) + ' ｜ ' +
            ' '.join(row['actual'].split()[len(row['reds'].split()):]))


def render():
    lines = ['## 🎯 最新预测 (每开奖日自动出号)', '',
             '| 彩种 | 期号 | 预测号码 |', '|---|---|---|']
    for cfg in GAMES.values():
        entries = parse_rec(cfg)
        if not entries:
            continue
        best = max(entries, key=lambda e: 0)  # 最后块首组
        svg_card(cfg, best)
        lines.append(f'| {LABELS[cfg.name]} | {best["issue"]} | '
                     f'![](cards/{cfg.name}.svg) |')

    import pandas as pd
    vpath = ROOT / 'verify.csv'
    lines += ['', '## 📋 兑奖战绩 (开奖后自动校对)', '',
              '| 期号 | 彩种 | 预测 | 开奖 | 命中 | 奖金 |', '|---|---|---|---|---|---|']
    n = 0
    if vpath.exists():
        v = pd.read_csv(vpath).tail(15)
        for _, r in v[::-1].iterrows():
            p, a = fmt_draw(r)
            lines.append(f'| {r["issue"]} | {r["game"]} | `{p}` | `{a}` '
                         f'| 红{r["red_hits"]} 蓝{r["blue_hits"]} | {r["prize"]}元 |')
        all_v = pd.read_csv(vpath)
        n = len(all_v)

    if n:
        lines += ['', f'**汇总** (随机基线: 双色球 {BASELINE["ssq"]} · 大乐透 {BASELINE["dlt"]}):']
        for gname, sub in all_v.groupby('game'):
            lines.append(f'- {LABELS.get(gname, gname)}: 校对 {len(sub)} 注 · '
                         f'投入 {2 * len(sub)} 元 · 中回 {int(sub["prize"].sum())} 元 · '
                         f'红/注 {sub["red_hits"].mean():.2f} 蓝/注 {sub["blue_hits"].mean():.2f}')
    else:
        lines += ['', '_尚无已开奖校对记录, 首个开奖日后自动更新_']

    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8')
    if START not in text or END not in text:
        print('README.md 缺少标记区块, 跳过渲染')
        return
    i, j = text.index(START), text.index(END)
    readme.write_text(text[:i + len(START)] + '\n' + '\n'.join(lines) + '\n\n' + text[j:],
                      encoding='utf-8')
    print(f'README 已更新: {n} 条战绩')


if __name__ == '__main__':
    render()
