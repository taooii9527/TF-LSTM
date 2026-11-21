"""GitHub 首页(README)展示渲染: 分彩种 tab 看板 + 每期一行战绩。

python -m lottery.readme_card
读 predict.rec(各彩种最后块)与 verify.csv, 替换 README.md 中
<!-- LOTTERY:START --> ~ <!-- LOTTERY:END --> 标记区块, 并生成 cards/*.svg。
双色球/大乐透各一个 <details> 折叠 tab; 战绩每期一行(首组预测+合计奖金)。
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
LABELS = {'ssq': '双色球', 'dlt': '大乐透'}
START, END = '<!-- LOTTERY:START -->', '<!-- LOTTERY:END -->'


def svg_card(cfg, entry):
    """最新预测的号码球卡片: 红蓝单行排开, 球距统一(34px)保证对齐。"""
    balls = [(n, RED) for n in entry['reds']] + [(n, BLUE) for n in entry['blues']]
    w = 18 + len(entry['reds']) * 34 + 6 + len(entry['blues']) * 34
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="32">']
    x = 18
    for i, (n, color) in enumerate(balls):
        if i == len(entry['reds']):
            x += 6  # 红蓝区间隔
        parts.append(f'<circle cx="{x}" cy="16" r="14" fill="{color}"/>'
                     f'<text x="{x}" y="20" text-anchor="middle" font-size="12" '
                     f'fill="#fff" font-family="monospace">{n:02d}</text>')
        x += 34
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
    import pandas as pd
    vpath = ROOT / 'verify.csv'
    all_v = pd.read_csv(vpath) if vpath.exists() else None

    blocks = ['## 🎯 自动预测看板 (每开奖日自动出号 / 开奖后自动校对)', '']
    for cfg in GAMES.values():
        entries = parse_rec(cfg)
        if entries:
            best = entries[0]
            svg_card(cfg, best)
            head = f'最新预测 **{best["issue"]} 期**: ![](cards/{cfg.name}.svg)'
        else:
            head = '_暂无预测, 首个开奖日调度后自动生成_'
        lines = [head, '', '| 期号 | 预测 → 开奖 | 命中 | 奖金 |', '|---|---|---|---|']
        if all_v is not None and len(all_v):
            sub = all_v[all_v['game'] == cfg.name]
            if len(sub):
                issues = list(dict.fromkeys(sub['issue']))[-15:][::-1]
                for issue in issues:            # 每期一行(首组预测, 合计奖金)
                    g = sub[sub['issue'] == issue]
                    r1 = g.iloc[0]
                    p, a = fmt_draw(r1)
                    lines.append(f'| {issue} | `{p}` → `{a}` '
                                 f'| 红{r1["red_hits"]} 蓝{r1["blue_hits"]} '
                                 f'| {int(g["prize"].sum())}元 |')
                lines += ['', f'**汇总**: 校对 {len(issues)} 期 · 投入 {2 * len(sub)} 元 · '
                          f'中回 {int(sub["prize"].sum())} 元 · '
                          f'红/注 {sub["red_hits"].mean():.2f} 蓝/注 {sub["blue_hits"].mean():.2f} '
                          f'(基线 {BASELINE[cfg.name]})']
        else:
            lines += ['', '_尚无已开奖校对记录, 首个开奖日后自动更新_']
        open_tag = '<details open>' if cfg.name == 'ssq' else '<details>'
        blocks += [open_tag, f'<summary><b>{LABELS[cfg.name]} ({cfg.name})</b></summary>', '']
        blocks += lines
        blocks += ['', '</details>', '']
    lines = blocks

    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8')
    if START not in text or END not in text:
        print('README.md 缺少标记区块, 跳过渲染')
        return
    i, j = text.index(START), text.index(END)
    readme.write_text(text[:i + len(START)] + '\n' + '\n'.join(lines) + '\n\n' + text[j:],
                      encoding='utf-8')
    print('README 已更新: 分彩种 tab 看板')


if __name__ == '__main__':
    render()
