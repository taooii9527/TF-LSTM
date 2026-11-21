"""GitHub 首页(README)展示渲染: 左右两栏看板 + 预测/开奖双列表 + 命中金边标记。

python -m lottery.readme_card
读 predict.rec(各彩种最后块)与 verify.csv, 替换 README.md 中
<!-- LOTTERY:START --> ~ <!-- LOTTERY:END --> 标记区块, 并生成 cards/*.svg。
布局: 预测列表(待开奖)与开奖列表(每期一行)左右两栏并排, 左双色球/右大乐透;
预测卡上与开奖号相同的球加金色描边标记。
由 GitHub Actions 在每轮调度后调用, commit 回仓库实现首页自动更新。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lottery.config import GAMES           # noqa: E402
from lottery.scheduler import parse_rec    # noqa: E402

RED, BLUE, GOLD = '#d5281f', '#1f6fd5', '#f0a30a'
LABELS = {'ssq': '🔴 双色球', 'dlt': '🔵 大乐透'}
START, END = '<!-- LOTTERY:START -->', '<!-- LOTTERY:END -->'


def svg_card(cfg, entry, path=None, hits=()):
    """号码球卡片: 红蓝单行等距排开; hits 中的球加金色描边(命中标记)。"""
    balls = [(int(n), RED) for n in entry['reds']] + [(int(n), BLUE) for n in entry['blues']]
    w = 18 + len(entry['reds']) * 34 + 6 + len(entry['blues']) * 34
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="32">']
    x = 18
    for i, (n, color) in enumerate(balls):
        if i == len(entry['reds']):
            x += 6  # 红蓝区间隔
        ring = 'stroke="#f0a30a" stroke-width="3.5"' if n in hits else ''
        parts.append(f'<circle cx="{x}" cy="16" r="13" fill="{color}" {ring}/>'
                     f'<text x="{x}" y="20" text-anchor="middle" font-size="12" '
                     f'fill="#fff" font-family="monospace">{n:02d}</text>')
        x += 34
    parts.append('</svg>')
    out = ROOT / 'cards'
    out.mkdir(exist_ok=True)
    path = out / (path or f'{cfg.name}.svg')
    path.write_text(''.join(parts), encoding='utf-8')
    return path


def hits_of(pred_reds, pred_blues, actual):
    """预测与开奖的号码交集(整型集合)。"""
    nums = [int(v) for v in actual.split()]
    nr = len(pred_reds)
    return set(nums[:nr]) & {int(v) for v in pred_reds}, \
        set(nums[nr:]) & {int(v) for v in pred_blues}


def render():
    import pandas as pd
    vpath = ROOT / 'verify.csv'
    all_v = pd.read_csv(vpath) if vpath.exists() else pd.DataFrame()
    names = list(GAMES.keys())

    # 每彩种: 待开奖预测 + 已校对期(倒序, 最近 10)
    info = {}
    for name in names:
        cfg = GAMES[name]
        entries = parse_rec(cfg)
        pred_issue = pred_entry = None
        if entries:
            cand = entries[0]
            done = set(all_v[all_v['game'] == name]['issue']) if len(all_v) else set()
            if cand['issue'] not in done:
                pred_issue, pred_entry = cand['issue'], cand
        if len(all_v):
            sub = all_v[all_v['game'] == name]
            issues = list(dict.fromkeys(sub['issue']))[-10:][::-1]
        else:
            sub, issues = pd.DataFrame(), []
        info[name] = (cfg, pred_issue, pred_entry, issues, sub)

    # 1) 预测列表 (待开奖, 左右两栏)
    def _pred_cell(name):
        cfg, pi, pe, _, _ = info[name]
        if pi is None:
            return '—', '—'
        svg_card(cfg, pe)
        return f'**{pi}**', f'![](cards/{name}.svg)'

    l1, l2 = _pred_cell('ssq'), _pred_cell('dlt')
    lines = ['### 🎯 预测列表 (待开奖)', '',
             '| 双色球 | 预测号码 | 大乐透 | 预测号码 |', '|---|---|---|---|',
             f'| {l1[0]} | {l1[1]} | {l2[0]} | {l2[1]} |']

    # 2) 开奖列表 (每期一行, 左右两栏, 预测卡金边标记命中)
    lines += ['', '### ✅ 开奖列表 (每期一行, 金边 = 预测命中的球)', '',
              '| 双色球 | 预测 | 开奖 | 命中·奖金 | 大乐透 | 预测 | 开奖 | 命中·奖金 |',
              '|---|---|---|---|---|---|---|---|']

    def _issue_cells(name, issue):
        cfg, _, _, issues, sub = info[name]
        if issue not in issues:
            return '—', '—', '—', '—'
        g = sub[sub['issue'] == issue]
        r1 = g.iloc[0]
        nr = len(r1['reds'].split())
        hr, hb = hits_of(r1['reds'].split(), r1['blues'].split(), r1['actual'])
        pe = {'reds': r1['reds'].split(), 'blues': r1['blues'].split()}
        ae = {'reds': r1['actual'].split()[:nr],
              'blues': r1['actual'].split()[nr:]}
        svg_card(cfg, pe, path=f'{name}_{issue}.svg', hits=hr | hb)
        svg_card(cfg, ae, path=f'{name}_{issue}_draw.svg')
        return f'**{issue}**', f'![](cards/{name}_{issue}.svg)', \
            f'![](cards/{name}_{issue}_draw.svg)', \
            f'红{r1["red_hits"]} 蓝{r1["blue_hits"]} · {int(g["prize"].sum())}元'

    rows = max(len(info['ssq'][3]), len(info['dlt'][3])) or 0
    s_issues, d_issues = info['ssq'][3], info['dlt'][3]
    for i in range(max(rows, 1)):
        si = s_issues[i] if i < len(s_issues) else None
        di = d_issues[i] if i < len(d_issues) else None
        s = _issue_cells('ssq', si) if si else ('—',) * 4
        d = _issue_cells('dlt', di) if di else ('—',) * 4
        lines.append('| ' + ' | '.join(s + d) + ' |')

    # 尾部合计
    def _sum(name):
        _, _, _, _, sub = info[name]
        return f'中回 {int(sub["prize"].sum())} 元 / {len(sub)} 注' if len(sub) else '—'

    lines.append(f'| **合计** | | | {_sum("ssq")} | **合计** | | | {_sum("dlt")} |')

    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8')
    if START not in text or END not in text:
        print('README.md 缺少标记区块, 跳过渲染')
        return
    i, j = text.index(START), text.index(END)
    readme.write_text(text[:i + len(START)] + '\n' + '\n'.join(lines) + '\n\n' + text[j:],
                      encoding='utf-8')
    print('README 已更新: 左右两栏 + 双列表 + 命中金边')


if __name__ == '__main__':
    render()
