# TF-LSTM 彩票预测研究工作台

LSTM 逐球位建模 + 温度采样的双色球/大乐透预测研究项目, 含自动回测、
随机基线对比与开奖时刻表驱动的自动预测/校对闭环。

> **声明**: 开奖为均匀随机事件, 本项目所有评估(回测/账本/统计哨兵)均以随机基线
> 为参照, 结果仅供研究娱乐, 不构成任何购彩建议。

## 自动预测看板

<!-- LOTTERY:START -->
### 🎯 预测列表 (待开奖)

| 双色球 | 预测号码 | 大乐透 | 预测号码 |
|---|---|---|---|
| **26121** | ![](cards/ssq.svg) | — | — |

### ✅ 开奖列表 (每期一行, 金边 = 预测命中的球)

| 双色球 | 预测 | 开奖 | 命中·奖金 | 大乐透 | 预测 | 开奖 | 命中·奖金 |
|---|---|---|---|---|---|---|---|
| — | — | — | — | **26120** | ![](cards/dlt_26120.svg) | ![](cards/dlt_26120_draw.svg) | 红1 蓝0 · 0元 |
| **合计** | | | — | **合计** | | | 中回 0 元 / 3 注 |

<!-- LOTTERY:END -->

## 功能

- **预测**: 滑动窗口时序建模, 逐球位 softmax + 温度采样, 组间汉明距离/蓝球互异保证覆盖
- **闭环**: 开奖时刻表驱动(双色球 二/四/日 21:15, 大乐透 一/三/六 21:25),
  开奖前 45min 自动预测, 开奖后自动校对落盘 `verify.csv`
- **验证**: 38 项不变量测试 + API 契约测试 + 运行时健康哨兵(`/api/health`)
- **评估**: walk-forward 回测、期望账本(真实奖级)、超参随机搜索
- **看板**: Flask Web 界面(本地) + GitHub README 自动看板(云端)

## 使用

```bash
pip install -r requirements.txt
python auto.py predict --name ssq          # 预测
python auto.py backtest --steps 30        # 回测
python -m lottery.scheduler               # 本地守护(自动预测/校对)
python -m lottery.webapp                  # Web 界面 http://127.0.0.1:8765
```

GitHub Actions 每开奖日自动预测一组并在上方更新战绩; 详见
`.github/workflows/schedule.yml`。
