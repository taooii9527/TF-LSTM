"""双色球/大乐透预测工具 - 薄入口。

用法:
    python auto.py                       # 等价于 predict --name ssq
    python auto.py predict --name dlt
    python auto.py backtest --name ssq --steps 30 --refit-every 10
    python auto.py tune --name ssq --trials 10
"""

from lottery.cli import main

if __name__ == '__main__':
    main()
