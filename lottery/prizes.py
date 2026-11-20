"""奖级规则表(供期望账本与自动校对复用, 避免依赖 webapp/tensorflow)。"""


def _prize_ssq(r, b):
    """双色球固定奖级; 一/二等奖为浮动, 按常见近似值。"""
    if r == 6 and b == 1:
        return 5_000_000
    if r == 6:
        return 150_000
    if r == 5 and b == 1:
        return 3_000
    if r == 5 or (r == 4 and b == 1):
        return 200
    if (r == 4 and b == 0) or (r == 3 and b == 1):
        return 10
    return 5 if b == 1 else 0


def _prize_dlt(f, g):
    """大乐透固定奖级; 一/二等奖为浮动, 按常见近似值。"""
    if f == 5 and g == 2:
        return 5_000_000
    if f == 5 and g == 1:
        return 80_000
    if f == 5 and g == 0:
        return 10_000
    if f == 4 and g == 2:
        return 3_000
    if f == 4 and g == 1:
        return 300
    if f == 3 and g == 2:
        return 200
    if f == 4 and g == 0:
        return 100
    if f == 3 and g == 1:
        return 15
    if f == 2 and g == 2:
        return 10
    if (f == 3 and g == 0) or (f == 1 and g == 2) or (f == 2 and g == 1) \
            or (f == 0 and g == 2):
        return 5
    return 0


PRIZE_FN = {'ssq': _prize_ssq, 'dlt': _prize_dlt}
