"""彩种配置与全局常量。"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """单个彩种的全部差异点集中于此。"""
    name: str          # CLI 标识: ssq / dlt
    tag: str           # 日志标签: SSQ / DLT
    url: str           # 500.com 历史数据接口
    csv: str           # 数据落盘路径
    model: str         # 模型保存路径(后缀运行时适配 tf 版本)
    record: str        # 预测记录文件
    red_count: int     # 红球个数
    red_max: int       # 红球最大值
    blue_count: int    # 蓝球个数
    blue_max: int      # 蓝球最大值

    @property
    def n_balls(self) -> int:
        return self.red_count + self.blue_count

    @property
    def reds(self) -> list:
        return [f'red{i}' for i in range(1, self.red_count + 1)]

    @property
    def blues(self) -> list:
        if self.blue_count == 1:  # ssq 蓝球列名保持 'blue' 以兼容旧 csv
            return ['blue']
        return [f'blue{i}' for i in range(1, self.blue_count + 1)]

    @property
    def ball_cols(self) -> list:
        return self.reds + self.blues

    @property
    def expected_red_hits(self) -> float:
        """随机选号的期望红球命中数。"""
        return self.red_count * self.red_count / self.red_max

    @property
    def expected_blue_hits(self) -> float:
        """随机选号的期望蓝球命中数。"""
        return self.blue_count * self.blue_count / self.blue_max


GAMES = {
    'ssq': GameConfig(
        name='ssq', tag='SSQ',
        url='https://datachart.500.com/ssq/history/newinc/history.php?limit=5000&sort=0',
        csv='ssq.csv', model='ssq_model.keras', record='predict.rec',
        red_count=6, red_max=33, blue_count=1, blue_max=16,
    ),
    'dlt': GameConfig(
        name='dlt', tag='DLT',
        url='https://datachart.500.com/dlt/history/newinc/history.php?limit=5000&sort=0',
        csv='dlt.csv', model='dlt_model.keras', record='predict.rec',
        red_count=5, red_max=35, blue_count=2, blue_max=12,
    ),
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
BALL_RE = re.compile(r'^\d{2}$')
TRAIN_RATIO = 0.8      # 时间顺序切分: 前 80% 训练 / 后 20% 验证
