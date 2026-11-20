"""历史开奖数据抓取。"""

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from lottery.config import BALL_RE, HEADERS


def fetch_draws(cfg):
    """抓取历史开奖，返回 (按期号倒序 DataFrame, 下一期期号)。"""
    resp = None
    last_err = None
    for attempt in range(1, 4):
        try:
            logging.info('抓取历史数据 (第 %d/3 次)', attempt)
            resp = requests.get(cfg.url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = 'utf-8'
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * attempt)
    if resp is None:
        raise RuntimeError(f'抓取历史数据失败(网络/风控): {last_err}')

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', id='tablelist')
    if table is None:
        raise RuntimeError('页面未找到 table#tablelist，站点结构可能已变化')

    records = []
    for row in table.find_all('tr'):
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        # 有效数据行: 期号为纯数字且所有球号均为两位数字符串
        if len(cols) < cfg.n_balls + 1 or not cols[0].isdigit():
            continue
        balls = cols[1:1 + cfg.n_balls]
        if not all(BALL_RE.match(b) for b in balls):
            continue
        records.append([int(cols[0])] + [int(b) for b in balls]
                       + [cols[-2], cols[-1]])  # [-2]=销售额, [-1]=开奖日期

    if not records:
        raise RuntimeError('未解析到任何开奖记录')
    if len(records) <= 60:
        raise RuntimeError(f'记录过少({len(records)}期)，数据异常')

    df = pd.DataFrame(records, columns=['No'] + cfg.ball_cols + ['sale', 'date'])
    next_no = int(df['No'].iloc[0]) + 1  # 首行为最新一期
    return df, next_no
