"""lottery - 双色球/大乐透 预测工具包。

模块结构:
    config   彩种配置与常量
    fetcher  历史开奖数据抓取
    dataset  滑动窗口数据集构建
    model    模型构建与保存
    sampler  号码采样与多样性约束
    runner   预测主流程
    backtest walk-forward 回测
    tuner    随机搜索调参
    cli      命令行入口
"""

from lottery.config import GAMES, GameConfig

__all__ = ['GAMES', 'GameConfig']
