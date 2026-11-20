"""L2 不变量测试: 测数学性质而非代码路径。"""

import numpy as np

from lottery.dataset import build_dataset, window_input, _normalize
from lottery.sampler import diverse_candidates, pattern_valid, sample_set
from lottery.prizes import _prize_dlt, _prize_ssq


# ---------------- config 一致性 ----------------
def test_config_consistency(cfg):
    assert cfg.red_count == len(cfg.reds) and cfg.red_count in (5, 6)
    assert len(cfg.blues) == cfg.blue_count
    assert len(cfg.ball_cols) == cfg.red_count + cfg.blue_count
    assert cfg.red_max >= cfg.red_count and cfg.blue_max >= cfg.blue_count


# ---------------- sampler 合法性 ----------------
def _uniform_probs(cfg):
    return {c: np.full((1, m), 1.0 / m)
            for c, m in zip(cfg.ball_cols,
                            [cfg.red_max] * cfg.red_count + [cfg.blue_max] * cfg.blue_count)}


def test_sample_set_valid(cfg, rng):
    probs = _uniform_probs(cfg)
    for _ in range(20):
        reds, blues, _ = sample_set(cfg, probs, 1.0, rng)
        assert len(set(reds)) == cfg.red_count
        assert all(1 <= v <= cfg.red_max for v in reds)
        assert all(1 <= v <= cfg.blue_max for v in blues)
        assert pattern_valid(cfg, reds)


def test_diverse_candidates_hamming(cfg):
    results = diverse_candidates(cfg, _uniform_probs(cfg), rounds=3, temperature=1.0,
                                 min_hamming=2, seed=42)
    reds_list = [r for r, _, _ in results]
    assert len(reds_list) == 3
    for i in range(len(reds_list)):
        for j in range(i + 1, len(reds_list)):
            common = len(set(reds_list[i]) & set(reds_list[j]))
            assert cfg.red_count - common >= 2, '组间汉明距离不足'
    blues_list = [tuple(sorted(b)) for _, b, _ in results]
    assert len(set(blues_list)) == len(blues_list), '组间蓝球(后区)重复, 浪费覆盖'


# ---------------- dataset 对齐与无泄漏 ----------------
def test_dataset_no_nan(cfg, df_asc):
    ds = build_dataset(df_asc, cfg, window=10)
    assert np.isfinite(ds.X).all() and np.isfinite(ds.X_next).all()
    assert 10 <= ds.split < len(ds.X)


def test_dataset_alignment(cfg, df_asc):
    """X[i] 的最后一个时间步必须恰好是第 i+window-1 期(时间升序)。"""
    window = 10
    ds = build_dataset(df_asc, cfg, window=window)
    vals = df_asc[cfg.ball_cols].values.astype(np.float32)
    w = vals[window - 1:window].copy()      # 第 0 个窗口的末行 = 第 window-1 期
    _normalize(w, cfg)
    assert np.allclose(ds.X[0, -1], w[0], atol=1e-6)


def test_dataset_next_is_latest(cfg, df_asc):
    """X_next = 最新 window 期原样归一化(预测输入含最新一期, 不复用训练窗口)。"""
    ds = build_dataset(df_asc, cfg, window=10)
    w = df_asc[cfg.ball_cols].values.astype(np.float32)[-10:].copy()
    _normalize(w, cfg)
    assert np.allclose(ds.X_next[0], w, atol=1e-6)


def test_dataset_target_alignment(cfg, df_asc):
    """targets[i] 必须是 X[i] 末行(第 i+window-1 期)的下一期开奖。"""
    ds = build_dataset(df_asc, cfg, window=10)
    vals = df_asc[cfg.ball_cols].values
    i = ds.split  # 任取一个训练样本
    assert list(map(int, vals[i + 10])) == \
        [int(ds.targets[c][i]) + 1 for c in cfg.ball_cols]


def test_window_input_stats_shape(cfg, df_asc):
    vals = df_asc[cfg.ball_cols].values
    out = window_input(vals, cfg, window=10, stats=True)
    n_feat = len(cfg.ball_cols) + cfg.red_max + cfg.blue_max
    assert out.shape == (1, 10, n_feat)


# ---------------- 奖级表完整性 ----------------
def test_prize_ssq_complete():
    combos = {(r, b): _prize_ssq(r, b)
              for r in range(7) for b in range(2)}
    assert combos[(6, 1)] == 5_000_000 and combos[(6, 0)] == 150_000
    assert combos[(0, 1)] == 5 and combos[(0, 0)] == 0
    assert combos[(2, 0)] == 0 and combos[(1, 0)] == 0
    winners = [k for k, v in combos.items() if v > 0]
    assert len(winners) == 10  # 双色球共 10 个中奖组合(含蓝球独中)


def test_prize_dlt_complete():
    combos = {(f, g): _prize_dlt(f, g)
              for f in range(6) for g in range(3)}
    assert combos[(5, 2)] == 5_000_000
    assert combos[(0, 2)] == 5  # 0+2 亦为九等奖(后区全中)
    assert combos[(4, 0)] == 100 and combos[(2, 1)] == 5
    winners = [k for k, v in combos.items() if v > 0]
    assert len(winners) == 13  # 大乐透固定+浮动共 13 个中奖组合


# ---------------- 历史 draw 全部通过形态校验(高比例) ----------------
def test_pattern_valid_on_history(cfg, df_asc):
    vals = df_asc[cfg.reds].values
    passed = sum(pattern_valid(cfg, list(map(int, row))) for row in vals)
    assert passed / len(vals) >= 0.98, '历史开奖大量未过形态校验, 校验过严'


# ---------------- rec 解析 round-trip(生产事故回归) ----------------
def test_pred_re_negative_score(tmp_path, monkeypatch):
    from lottery import webapp as W
    from lottery.config import GAMES
    (tmp_path / 'predict.rec').write_text(
        '2026-10-16 12:00:00 ----------SSQ------------\n'
        'LAST SALE [2026-10-15][26118]\n'
        '[26119] #1 red 03 08 16 20 23 31 blue 09 [SCORE] -17.5382\n'
        '[26119] #2 red 05 11 17 21 28 32 blue 14 [SCORE] -18.0021\n')
    monkeypatch.chdir(tmp_path)
    lp = W._last_predict(GAMES['ssq'])
    assert lp and len(lp['groups']) == 2
    assert lp['groups'][0]['score'] == -17.5382
    assert lp['groups'][1]['reds'] == [5, 11, 17, 21, 28, 32]


def test_last_predict_real_rec():
    """真实产物 round-trip: predict.rec 里写进去的必须全部读得回来。"""
    import pytest
    from pathlib import Path
    from lottery import webapp as W
    from lottery.config import GAMES
    cfg = GAMES['ssq']
    if not Path(cfg.record).exists():
        pytest.skip('predict.rec 不存在')
    lp = W._last_predict(cfg)
    assert lp is not None and len(lp['groups']) >= 1, \
        'predict.rec 存在但解析不到预测组 — 写入/解析格式漂移!'
