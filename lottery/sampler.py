"""号码采样与多样性约束。"""

import logging

import numpy as np


def sample_set(cfg, head_probs, temperature, rng):
    """按各头 softmax 做温度加权无放回采样，保证号码合法且红球不重复。

    temperature->0 退化为近似 argmax，->inf 趋近均匀采样。
    返回 (reds升序, blues升序, 联合对数概率得分)。
    """
    def sample(head, max_val, banned):
        p = head[0].astype(np.float64) ** (1.0 / temperature)
        for b in banned:
            p[b - 1] = 0.0
        p_sum = p.sum()
        if p_sum <= 0:  # 理论不可达: softmax>0 且 max_val > banned 数
            avail = [i for i in range(max_val) if (i + 1) not in banned]
            idx = rng.choice(avail)
            return idx + 1, -27.6  # log(1e-12)
        p /= p_sum
        idx = rng.choice(len(p), p=p)
        return idx + 1, float(np.log(max(p[idx], 1e-12)))

    banned_reds, reds_out, score = set(), [], 0.0
    for c in cfg.reds:
        num, lp = sample(head_probs[c], cfg.red_max, banned_reds)
        banned_reds.add(num)
        reds_out.append(num)
        score += lp

    banned_blues, blues_out = set(), []
    for c in cfg.blues:
        num, lp = sample(head_probs[c], cfg.blue_max, banned_blues)
        banned_blues.add(num)
        blues_out.append(num)
        score += lp

    return sorted(reds_out), sorted(blues_out), float(score)


def diverse_enough(reds_out, prev_sets, min_hamming, blues_out=(), prev_blue_sets=()):
    """红球与已生成集合保持最小汉明距离; 蓝球(后区)组间互异。

    蓝球互异不改变期望命中, 但最大化"至少一注中蓝"(六等奖)的覆盖概率。
    """
    if min_hamming > 0:
        for prev in prev_sets:
            common = len(set(reds_out) & set(prev))
            if len(reds_out) - common < min_hamming:
                return False
    if prev_blue_sets and tuple(sorted(blues_out)) in prev_blue_sets:
        return False
    return True


def pattern_valid(cfg, reds):
    """集合级合理性(借鉴 powerpredict 的 range coverage / pattern matching):
    和值在均值±3σ内、至少覆盖 2 个三等分区。
    σ 按无放回抽样解析式计算。阈值取 3σ 以保证 ≥99% 真实均匀开奖可通过
    (2σ 会误拒 ~4.6% 合法组合, 全奇全偶规则同理已移除)。
    """
    n, mx = cfg.red_count, cfg.red_max
    mean = n * (mx + 1) / 2
    sd = (n * (mx + 1) * (mx - n) / 12) ** 0.5
    if not (mean - 3 * sd <= sum(reds) <= mean + 3 * sd):
        return False
    zones = {min((v - 1) * 3 // mx, 2) for v in reds}
    return len(zones) >= 2


def diverse_candidates(cfg, head_probs, rounds, temperature, min_hamming, seed):
    """生成 rounds 组互不趋同的预测，按联合概率得分降序返回。"""
    results, prev_sets, prev_blues = [], [], []
    for e in range(rounds):
        t = temperature
        for attempt in range(50):
            rng = np.random.default_rng(seed + e * 1000 + attempt)
            cand = sample_set(cfg, head_probs, t, rng)
            if diverse_enough(cand[0], prev_sets, min_hamming,
                              cand[1], prev_blues) \
                    and pattern_valid(cfg, cand[0]):
                break
            t += 0.1  # 采不出合法/发散组合时升温重试
        else:
            logging.warning('预测组 %d 经 50 次重试仍未满足汉明距离约束, 使用最后候选', e + 1)
        prev_sets.append(cand[0])
        prev_blues.append(tuple(sorted(cand[1])))
        results.append(cand)
    results.sort(key=lambda r: -r[2])
    return results
