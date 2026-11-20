"""pytest 公共夹具: 项目路径与彩种参数化。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from lottery.config import GAMES


@pytest.fixture(params=list(GAMES.values()), ids=list(GAMES))
def cfg(request):
    return request.param


@pytest.fixture()
def df_asc(cfg):
    """时间升序开奖数据(读缓存, 不联网); 无缓存则跳过。"""
    import pandas as pd
    path = ROOT / cfg.csv
    if not path.exists():
        pytest.skip(f'{cfg.csv} 不存在')
    df = pd.read_csv(path)
    return df.iloc[::-1].reset_index(drop=True)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)
