"""调度判定纯函数测试: 开奖时刻表、预测窗、校对判定。"""

from datetime import datetime

from lottery.config import GAMES
from lottery.scheduler import draw_moment, judge, parse_rec, should_predict

SSQ, DLT = GAMES['ssq'], GAMES['dlt']
TUE = datetime(2026, 10, 20)   # 周二, 双色球开奖日
MON = datetime(2026, 10, 19)   # 周一, 大乐透开奖日
WED = datetime(2026, 10, 21)   # 周三, 两者都开奖


def test_draw_moment_schedule():
    m = draw_moment(SSQ, TUE.replace(hour=10))
    assert (m.hour, m.minute) == (21, 15)
    assert draw_moment(SSQ, MON) is None
    m = draw_moment(DLT, MON.replace(hour=10))
    assert (m.hour, m.minute) == (21, 25)
    assert draw_moment(DLT, TUE) is None


def test_should_predict_window():
    latest = 26118
    assert should_predict(SSQ, TUE.replace(hour=20, minute=29), latest, 0) is None
    assert should_predict(SSQ, TUE.replace(hour=20, minute=44), latest, 0) == 26119
    assert should_predict(SSQ, TUE.replace(hour=21, minute=0), latest, 0) == 26119
    assert should_predict(SSQ, TUE.replace(hour=21, minute=15), latest, 0) is None  # 已开奖
    assert should_predict(SSQ, TUE.replace(hour=21, minute=0), latest, 26119) is None  # 已预测
    assert should_predict(SSQ, WED.replace(hour=21, minute=0), latest, 0) is None    # 非开奖日
    assert should_predict(DLT, MON.replace(hour=21, minute=0), 26100, 0) == 26101


def test_judge_prizes():
    assert judge(SSQ, [1, 2, 3, 10, 20, 30], [5],
                 [1, 2, 3, 11, 22, 33], [5]) == {'red_hits': 3, 'blue_hits': 1, 'prize': 10}
    assert judge(SSQ, [1, 2, 3, 4, 5, 6], [1],
                 [1, 2, 3, 4, 5, 6], [1])['prize'] == 5_000_000
    assert judge(DLT, [1, 2, 3, 20, 30], [5, 9],
                 [1, 2, 3, 11, 22], [5, 9]) == {'red_hits': 3, 'blue_hits': 2, 'prize': 200}
    assert judge(DLT, [1, 2, 3, 4, 5], [1, 2],
                 [6, 7, 8, 9, 10], [3, 4])['prize'] == 0


def test_parse_rec_real():
    """真实产物 round-trip: rec 存在则必须解析出条目。"""
    import pytest
    from pathlib import Path
    if not Path(SSQ.record).exists():
        pytest.skip('predict.rec 不存在')
    entries = parse_rec(SSQ)
    assert entries and all(
        len(e['reds']) == SSQ.red_count and len(e['blues']) == SSQ.blue_count
        for e in entries)
