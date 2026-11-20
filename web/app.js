/* 前端逻辑: 只做取数/渲染/轮询, 业务均在后端 */
'use strict';
let game = 'ssq';
let currentTab = 'overview';
let charts = {};
let overviewData = null;
let artifactsData = null;
let pollTimer = null;

const $ = s => document.querySelector(s);
const PARAM_DEFS = {
  predict: [['rounds', '预测组数', 3], ['window', '窗口期数', 10], ['epochs', '训练轮数', 100], ['temperature', '采样温度', 1.0]],
  backtest: [['steps', '回测期数', 30], ['refit_every', '重训间隔', 10], ['window', '窗口期数', 10], ['epochs', '训练轮数', 100]],
  tune: [['trials', 'trial 数', 10], ['epochs', '训练轮数', 100]],
  fetch: [],
};

/* ---------- 工具 ---------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function ballRow(nums, cls) {
  const d = el('span', 'balls');
  nums.forEach(v => d.appendChild(el('span', `nball ${cls}`, String(v).padStart(2, '0'))));
  return d;
}
function kpi(v, l, cls) {
  const k = el('div', `kpi ${cls || ''}`);
  k.appendChild(el('div', 'v', v));
  k.appendChild(el('div', 'l', l));
  return k;
}
function chart(id) {
  if (window.__echartsFail || !window.echarts) return null;
  if (!charts[id]) charts[id] = echarts.init(document.getElementById(id));
  return charts[id];
}
const AXIS = { axisLabel: { color: '#8a93a3' }, splitLine: { lineStyle: { color: '#eef1f6' } } };
const RED_SERIES = ['#d5281f', '#d5281f', '#e6a23c', '#d9dee8', '#8b94a7', '#5f94f0', '#3fb96f'];

/* ---------- 数据总览 ---------- */
async function renderOverview() {
  const [ov, ok] = await fetchOverview();
  const empty = $('#ov-empty'), body = $('#ov-body');
  empty.classList.toggle('hidden', ok);
  body.classList.toggle('hidden', !ok);
  if (!ok) return;

  const k = $('#ov-kpis');
  k.replaceChildren(
    kpi(ov.draws, '历史期数'),
    kpi(ov.latest_no, `最新期号 (${ov.latest_date})`),
    kpi(ov.red_mean_sum, '红球和值均值'),
    kpi(`${(ov.randomness.consec_rate * 100).toFixed(0)}%`, '含连号期占比'));

  // 随机性体检: 组合结构分布 vs 理论超几何期望(单球频率无信息, 不再展示)
  const distOption = (d, color) => ({
    grid: { left: 40, right: 12, top: 24, bottom: 26 },
    xAxis: { type: 'category', data: d.counts.map((_, i) => i), ...AXIS },
    yAxis: { type: 'value', ...AXIS },
    series: [
      { type: 'bar', data: d.counts, itemStyle: { color } },
      { type: 'line', data: d.theo, symbol: 'none',
        lineStyle: { color: '#e6a23c', type: 'dashed' } }] });
  chart('chart-parity')?.setOption(distOption(ov.randomness.parity, '#d5281f'));
  chart('chart-size')?.setOption(distOption(ov.randomness.size, '#2f6fe4'));
  chart('chart-sum')?.setOption({
    grid: { left: 50, right: 16, top: 24, bottom: 26 },
    xAxis: { type: 'category', data: ov.sum_trend.issues, ...AXIS },
    yAxis: { type: 'value', ...AXIS },
    series: [
      { type: 'line', data: ov.sum_trend.sums, symbol: 'none',
        lineStyle: { color: '#d5281f', width: 1 } },
      { type: 'line', data: ov.sum_trend.roll20, symbol: 'none',
        lineStyle: { color: '#e6a23c', width: 2 } }],
  });

  const t = $('#ov-recent');
  t.replaceChildren(
    (() => { const r = el('tr'); ['期号', '日期', '红球', '蓝球'].forEach(h => r.appendChild(el('th', '', h))); return r; })(),
    ...ov.recent.map(row => {
      const tr = el('tr');
      tr.appendChild(el('td', 'mono', row.no));
      tr.appendChild(el('td', 'dim', row.date));
      const td1 = el('td'); td1.appendChild(ballRow(row.reds, 'r')); tr.appendChild(td1);
      const td2 = el('td'); td2.appendChild(ballRow(row.blues, 'b')); tr.appendChild(td2);
      return tr;
    }));
}

async function fetchOverview() {
  const r = await fetch(`/api/overview?name=${game}`);
  const d = await r.json();
  overviewData = d;
  return [d, !!d.ok];
}

/* ---------- 流程图: 按任务类型渲染节点链, 节点随日志实时点亮 ---------- */
const FLOW_KINDS = {
  predict: { label: 'predict — 预测下一期', nodes: [
    { s: 1, t: '数据抓取', d: '500.com 最新在前 → csv' },
    { s: 2, t: '构建数据集', d: '滑动窗口 10 期 · 80/20 时序切分' },
    { s: 2, t: '训练模型', d: 'LSTM 逐球位 softmax · EarlyStopping' },
    { s: 3, t: '推理', d: '最近 10 期 → 下期概率' },
    { s: 3, t: '采样出号', d: '温度采样 · 汉明距离≥2 · 合法性校验' },
    { s: 4, t: '落盘', d: '模型 .keras · predict.rec' },
  ] },
  backtest: { label: 'backtest — 滚动回测', nodes: [
    { s: 1, t: '数据抓取', d: '转为时间升序' },
    { s: 2, t: '构建前缀', d: '每期仅用该期之前数据, 无泄漏' },
    { s: 2, t: '滚动重训', d: '前缀上重新训练模型', loop: '每 refit 期循环' },
    { s: 3, t: '推理比对', d: '采样出号 vs 实际开奖' },
    { s: 3, t: '基线对比', d: '随机期望 · 偏差±SE 显著性' },
    { s: 4, t: '落盘', d: 'backtest_彩种.csv 逐期明细' },
  ] },
  tune: { label: 'tune — 随机搜索调参', nodes: [
    { s: 1, t: '数据抓取', d: '转为时间升序' },
    { s: 2, t: '采样参数', d: 'window/units/dropout/lr/batch 随机组合' },
    { s: 2, t: 'trial 训练', d: '目标 val_loss · patience=10', loop: '× trial 数循环' },
    { s: 3, t: '排序', d: '按 val_loss 升序取 top' },
    { s: 4, t: '落盘', d: 'tune_彩种.json' },
  ] },
  fetch: { label: 'fetch — 刷新数据', nodes: [
    { s: 1, t: '数据抓取', d: '500.com 最新在前' },
    { s: 4, t: '落盘 csv', d: '覆盖本地缓存' },
  ] },
};

let lastKind = 'predict';

function flowProgress(kind, logs) {
  const last = re => {
    for (let i = logs.length - 1; i >= 0; i--) if (re.test(logs[i].msg)) return logs[i].msg;
    return null;
  };
  const parts = [];
  if (kind === 'predict') {
    const tr = last(/训练完成/); if (tr) parts.push(tr);
    const pr = last(/预测组 \d+/); if (pr) parts.push(pr);
  } else if (kind === 'backtest') {
    const rf = last(/refit @/); if (rf) parts.push(rf);
    const bp = last(/回测进度/); if (bp) parts.push(bp);
  } else if (kind === 'tune') {
    const t = last(/trial \d+\/\d+/); if (t) parts.push(t);
  }
  return parts;
}

function renderFlow(s) {
  if (s.kind) lastKind = s.kind;
  const def = FLOW_KINDS[s.kind || lastKind];
  const st = s.stage || 0;
  $('#flow-task-label').textContent =
    s.running ? `${def.label} · 运行中` : def.label;
  const box = $('#flow-diagram');
  box.replaceChildren();
  def.nodes.forEach((n, i) => {
    if (i) box.appendChild(el('span', 'farrow', '➔'));
    const cls = st > n.s ? 'done' : (s.running && st === n.s ? 'active' : 'idle');
    const node = el('div', `fnode ${cls}`);
    node.appendChild(el('span', 'fi', String(i + 1)));
    const tx = el('div', 'ftx');
    tx.appendChild(el('div', 'ft', n.t));
    tx.appendChild(el('div', 'fd', n.d));
    if (n.loop) tx.appendChild(el('span', 'floop', n.loop));
    node.appendChild(tx);
    box.appendChild(node);
  });
  const prog = $('#flow-progress');
  if (s.error) {
    prog.replaceChildren(el('span', 'err', `任务失败: ${s.error}`));
  } else if (!s.running && st === 0) {
    prog.textContent = '选择任务并点击「运行任务」, 节点将随执行实时点亮';
  } else {
    const parts = flowProgress(s.kind || lastKind, s.logs || []);
    prog.textContent = parts.join('   ·   ') || (s.running ? '运行中…' : '已完成');
  }
}

function renderLog(logs) {
  const c = $('#log-console');
  c.replaceChildren(...logs.map(l => {
    const line = el('span', l.lvl === 'ERROR' ? 'err' : '');
    line.textContent = `[${l.t}] ${l.msg}\n`;
    return line;
  }));
  c.scrollTop = c.scrollHeight;
}

function renderTaskParams() {
  const kind = $('#f-task').value;
  const box = $('#f-params');
  box.replaceChildren(...PARAM_DEFS[kind].map(([key, label, def]) => {
    const wrap = el('label', '', label);
    const inp = el('input');
    inp.id = `p-${key}`; inp.value = def; inp.type = 'number'; inp.step = 'any';
    wrap.appendChild(inp);
    return wrap;
  }));
  $('#run-hint').textContent = kind === 'fetch' ? '仅刷新本地数据缓存, 不训练'
    : kind === 'tune' ? '每 trial 约一次完整训练 (30-60s)'
    : kind === 'backtest' ? '滚动回测, 每 N 期重训一次, 耗时数分钟' : '训练一次并采样多组预测';
}

function renderPredResult() {
  const box = $('#pred-result');
  const p = artifactsData?.predict;
  box.replaceChildren();
  if (!p) { box.appendChild(el('p', 'dim', '暂无预测记录')); return; }
  const head = el('p', 'dim small',
    `生成于 ${p.time} · 目标期 ${p.groups[0]?.no || '?'} · 基于开奖 ${p.last_sale?.no || '?'}`);
  box.appendChild(head);
  p.groups.forEach(g => {
    const row = el('div', 'grp');
    row.appendChild(el('span', 'tag', `#${g.idx}`));
    row.appendChild(ballRow(g.reds, 'r'));
    row.appendChild(ballRow(g.blues, 'b'));
    row.appendChild(el('span', 'dim small mono', `score ${g.score.toFixed(4)}`));
    box.appendChild(row);
  });
}

/* ---------- 回测 ---------- */
function renderBacktest() {
  const bt = artifactsData?.backtest;
  const empty = $('#bt-empty'), body = $('#bt-body');
  empty.classList.toggle('hidden', !!bt);
  body.classList.toggle('hidden', !bt);
  if (!bt) return;
  const sig = Math.abs(bt.deviation) > 2 * bt.se;
  $('#bt-kpis').replaceChildren(
    kpi(bt.mean_red, `红球命中均值 (基线 ${bt.exp_red})`),
    kpi(bt.mean_blue, `蓝球命中均值 (基线 ${bt.exp_blue.toFixed(3)})`),
    kpi(`${bt.deviation >= 0 ? '+' : ''}${bt.deviation} ± ${bt.se}`,
      `红球偏差 ${sig ? '· 超 2SE 显著' : '· 2SE 内随机波动'}`, sig ? 'warn' : 'ok'),
    kpi(`${bt.best.issue}期`, `最佳一期 红${bt.best.red_hits} 蓝${bt.best.blue_hits}`));

  loadLedger();

  chart('chart-bt')?.setOption({
    grid: { left: 40, right: 12, top: 24, bottom: 40 },
    xAxis: { type: 'category', data: bt.rows.map(r => r.issue), ...AXIS,
      axisLabel: { color: '#8b94a7', rotate: 45 } },
    yAxis: { type: 'value', maxCeil: 6, ...AXIS },
    series: [
      { type: 'bar', data: bt.rows.map(r => r.red_hits), itemStyle: { color: '#d5281f' } },
      { type: 'line', data: bt.rows.map(() => bt.exp_red), symbol: 'none',
        lineStyle: { color: '#e6a23c', type: 'dashed' } }],
  });

  const distOption = (dist, color) => {
    const keys = Object.keys(dist).map(Number).sort((a, b) => a - b);
    return { grid: { left: 40, right: 12, top: 24, bottom: 26 },
      xAxis: { type: 'category', data: keys, ...AXIS },
      yAxis: { type: 'value', ...AXIS },
      series: [{ type: 'bar', data: keys.map(k => dist[k]), itemStyle: { color } }] };
  };
  chart('chart-bt-red')?.setOption(distOption(bt.red_dist, '#d5281f'));
  chart('chart-bt-blue')?.setOption(distOption(bt.blue_dist, '#2f6fe4'));

  const t = $('#bt-table');
  t.replaceChildren((() => {
    const r = el('tr');
    ['期号', '预测红球', '实际红球', '红中', '蓝中'].forEach(h => r.appendChild(el('th', '', h)));
    return r;
  })(), ...bt.rows.map(row => {
    const tr = el('tr');
    tr.appendChild(el('td', 'mono', row.issue));
    const td1 = el('td'); td1.appendChild(ballRow(row.pred_reds, 'r')); tr.appendChild(td1);
    const td2 = el('td');
    const hits = new Set(row.actual_reds);
    row.pred_reds.forEach(v => td2.appendChild(el('span', `nball r${hits.has(v) ? ' hit' : ''}`,
      String(v).padStart(2, '0'))));
    td2.firstChild.style.marginRight = '6px';
    [...td2.children].forEach((c, i) => { if (i) c.style.marginLeft = '-20px'; });
    tr.appendChild(td2);
    tr.appendChild(el('td', 'mono', String(row.red_hits)));
    tr.appendChild(el('td', 'mono', String(row.blue_hits)));
    return tr;
  }));
}

/* ---------- 期望账本 + 预测战绩 ---------- */
async function loadLedger() {
  const d = await (await fetch(`/api/ledger?name=${game}`)).json();
  const chips = $('#ledger-chips');
  chips.replaceChildren();
  if (!d.ok) {
    $('#ledger-note').textContent = ` — ${d.message}`;
    return;
  }
  $('#ledger-note').textContent = ` — 回测 ${d.steps} 期`;
  const money = v => v >= 10000 ? `${(v / 10000).toFixed(v % 10000 ? 1 : 0)}万` : `${v}元`;
  [['投入', `${money(d.invest)}`], ['实际中奖', `${money(d.prize)}`],
   ['净收益', `${d.net >= 0 ? '+' : ''}${money(Math.abs(d.net))}`],
   ['返奖率', `${(d.ret_rate * 100).toFixed(1)}%`],
   ['中奖期数', `${d.wins}/${d.steps}`],
   ['最佳一期', d.best ? `${d.best.issue}期 中${money(d.best.prize)}` : '—']]
    .forEach(([l, v]) => {
      const chip = el('span', 'shape-chip');
      chip.appendChild(el('span', 'sc-l', l));
      chip.appendChild(el('span', 'sc-v', v));
      chips.appendChild(chip);
    });
}

async function loadForecastRecord() {
  const d = await (await fetch(`/api/forecast-record?name=${game}`)).json();
  const empty = $('#rec-empty'), t = $('#rec-table');
  empty.classList.toggle('hidden', !!d.ok);
  t.replaceChildren();
  if (!d.ok) return;
  t.appendChild((() => {
    const r = el('tr');
    ['期号', '状态', '预测号码', '命中'].forEach(h => r.appendChild(el('th', '', h)));
    return r;
  })());
  d.record.forEach(rec => {
    rec.groups.forEach((g, i) => {
      const tr = el('tr');
      tr.appendChild(el('td', 'mono', i === 0 ? String(rec.issue) : ''));
      tr.appendChild(el('td', rec.drawn ? 'dim' : '',
        rec.drawn ? '已开奖' : '待开奖'));
      const td = el('td');
      const balls = el('span', 'balls');
      g.reds.forEach(v => balls.appendChild(el('span', 'nball r', String(v).padStart(2, '0'))));
      g.blues.forEach(v => balls.appendChild(el('span', 'nball b', String(v).padStart(2, '0'))));
      td.appendChild(balls);
      tr.appendChild(td);
      const h = rec.drawn ? rec.hits[i] : null;
      tr.appendChild(el('td', 'mono', h ? `红${h.red} 蓝${h.blue}` : '—'));
      t.appendChild(tr);
    });
  });
}

/* ---------- 调优 ---------- */
function renderTune() {
  const tn = artifactsData?.tune;
  const empty = $('#tn-empty'), body = $('#tn-body');
  empty.classList.toggle('hidden', !!tn);
  body.classList.toggle('hidden', !tn);
  if (!tn || !tn.length) return;
  const best = tn[0];
  const keys = ['window', 'units', 'dropout', 'lr', 'batch'];
  const bc = $('#tn-best');
  bc.replaceChildren();
  bc.appendChild(el('h3', '', `最优组合 · val_loss ${best.val_loss.toFixed(4)}`));
  const wrap = el('div', 'shape-row');
  keys.forEach(k => {
    const chip = el('span', 'shape-chip');
    chip.appendChild(el('span', 'sc-l', k));
    chip.appendChild(el('span', 'sc-v', String(best[k])));
    wrap.appendChild(chip);
  });
  bc.appendChild(wrap);
  bc.appendChild(el('p', 'dim small',
    '注意: 彩票无可学习信号, 调优收敛到的是数值上更平滑的配置, 不代表预测力提升'));

  chart('chart-tune')?.setOption({
    grid: { left: 60, right: 12, top: 24, bottom: 30 },
    xAxis: { type: 'category', data: tn.map((_, i) => `#${i + 1}`), ...AXIS },
    yAxis: { type: 'value', scale: true, ...AXIS },
    series: [{ type: 'bar', data: tn.map(d => d.val_loss),
      itemStyle: { color: '#5f94f0' } }],
  });

  const t = $('#tn-table');
  t.replaceChildren((() => {
    const r = el('tr');
    ['排名', 'val_loss', 'window', 'units', 'dropout', 'lr', 'batch'].forEach(h => r.appendChild(el('th', '', h)));
    return r;
  })(), ...tn.map((d, i) => {
    const tr = el('tr');
    [String(i + 1), d.val_loss.toFixed(4), d.window, d.units, d.dropout, d.lr, d.batch]
      .forEach((v, j) => tr.appendChild(el('td', j <= 1 ? 'mono' : '', String(v))));
    return tr;
  }));
}

/* ---------- 下期分析: 热温冷/遗漏 + 模型形态聚合 ---------- */
let analysisData = null;

async function loadAnalysis() {
  analysisData = await (await fetch(`/api/analysis?name=${game}`)).json();
  renderAnalysis();
}

function heatRow(items, hotCls) {
  const row = el('div', 'heat-row');
  items.forEach(it => {
    const item = el('span', 'heat-item');
    const lv = it.level === 'hot' ? hotCls : (it.level === 'warm' ? 'w' : 'c');
    item.appendChild(el('span', `nball ${lv}`, String(it.ball).padStart(2, '0')));
    item.appendChild(el('span', 'omi', String(it.omi)));
    row.appendChild(item);
  });
  return row;
}

function renderAnalysis() {
  const empty = $('#ana-empty'), heatBox = $('#ana-heat');
  const ok = !!analysisData?.ok;
  empty.classList.toggle('hidden', ok);
  heatBox.classList.toggle('hidden', !ok);
  if (!ok) { $('#ana-empty-msg').textContent = analysisData?.message || '暂无数据'; return; }
  $('#ana-next-no').textContent = analysisData.next_no;
  $('#ana-red-max').textContent = analysisData.red_heat.length;
  $('#heat-red').replaceChildren(heatRow(analysisData.red_heat, 'r'));
  $('#heat-blue').replaceChildren(heatRow(analysisData.blue_heat, 'b'));

  const m = analysisData.model;
  const chips = $('#shape-chips'), recBlue = $('#rec-blue');
  chips.replaceChildren(); recBlue.replaceChildren();
  if (!m) {
    chips.appendChild(el('p', 'dim', '暂无已训练模型 — 运行一次 predict 后展示形态参考'));
    chart('chart-rec')?.clear();
    return;
  }
  [['和值区间', `${m.sum_range[0]} – ${m.sum_range[1]}`],
   ['奇偶比', m.odd_ratio], ['大小比', m.big_ratio]].forEach(([l, v]) => {
    const chip = el('span', 'shape-chip');
    chip.appendChild(el('span', 'sc-l', l));
    chip.appendChild(el('span', 'sc-v', v));
    chips.appendChild(chip);
  });

  const top = [...m.red_top].reverse();  // 横向条形图: 大值在上
  chart('chart-rec')?.setOption({
    grid: { left: 42, right: 60, top: 8, bottom: 8 },
    xAxis: { type: 'value', show: false },
    yAxis: { type: 'category', data: top.map(d => String(d.ball).padStart(2, '0')),
      axisLabel: { color: '#2b2f36', fontFamily: 'Consolas' },
      axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: 'bar', data: top.map(d => d.score), barWidth: 16,
      itemStyle: { color: '#d5281f', borderRadius: [0, 8, 8, 0] },
      label: { show: true, position: 'right', color: '#8a93a3',
        fontSize: 11, formatter: d => (d.value * 100).toFixed(0) + '%' } }],
  });
  m.blue_top.forEach(d => {
    const chip = el('span', 'nball b', String(d.ball).padStart(2, '0'));
    chip.title = `概率 ${(d.p * 100).toFixed(1)}%`;
    recBlue.appendChild(chip);
  });
}

/* ---------- 任务运行与轮询 ---------- */
async function onRun() {
  const kind = $('#f-task').value;
  const params = {};
  PARAM_DEFS[kind].forEach(([key]) => { params[key] = parseFloat($(`#p-${key}`).value); });
  const r = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, name: game, ...params }),
  });
  const d = await r.json();
  if (!d.ok) { $('#run-hint').textContent = d.message; return; }
  $('#btn-run').disabled = true;
  pollStatus();
}

function pollStatus() {
  clearInterval(pollTimer);
  const tick = async () => {
    const s = await (await fetch('/api/status')).json();
    renderFlow(s);
    renderLog(s.logs);
    if (!s.running) {
      clearInterval(pollTimer);
      pollTimer = null;
      $('#btn-run').disabled = false;
      if (s.error) $('#run-hint').textContent = `任务失败: ${s.error}`;
      else {
        $('#run-hint').textContent = '任务完成';
        await loadArtifacts();
        if (s.kind === 'predict') {
          renderPredResult();
          if (currentTab === 'pipeline') loadAnalysis();  // 模型已重训, 分析同步刷新
        }
        if (s.kind === 'fetch') { renderOverview(); }
        if (s.kind === 'backtest') { switchTab('backtest'); }
        if (s.kind === 'tune') { switchTab('tune'); }
      }
    }
  };
  tick();
  pollTimer = setInterval(tick, 1500);
}

async function runTask(kind) {
  switchTab('pipeline');
  $('#f-task').value = kind;
  renderTaskParams();
  const r = await fetch('/api/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, name: game }),
  });
  const d = await r.json();
  $('#run-hint').textContent = d.ok ? '' : d.message;
  if (d.ok) { $('#btn-run').disabled = true; pollStatus(); }
}

/* ---------- 运行时健康徽章 ---------- */
async function loadHealth() {
  const b = $('#health-badge');
  try {
    const d = await (await fetch('/api/health')).json();
    const fails = (d.checks || []).flatMap(g =>
      (g.items || []).filter(i => !i.passed).map(i => `${g.game}: ${i.name} (${i.detail})`));
    b.classList.remove('ok', 'bad');
    b.classList.add(d.ok ? 'ok' : 'bad');
    $('#health-text').textContent = d.ok ? '自检通过' : `自检异常 ${fails.length} 项`;
    b.title = fails.length ? `异常明细:\n${fails.join('\n')}` : '全部运行时检查通过';
  } catch {
    b.classList.remove('ok');
    b.classList.add('bad');
    $('#health-text').textContent = '自检不可用';
    b.title = '无法连接 /api/health';
  }
}

/* ---------- 导航 ---------- */
const TAB_RENDER = {
  overview: renderOverview,
  pipeline: async () => {
    await loadArtifacts();
    renderPredResult();
    await loadAnalysis();
    await loadForecastRecord();
  },
  backtest: renderBacktest,
  tune: renderTune,
};

function switchTab(name) {
  currentTab = name;
  document.querySelectorAll('.tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${name}`));
  TAB_RENDER[name]?.();
  Object.values(charts).forEach(c => c.resize());
}

async function loadArtifacts() {
  artifactsData = await (await fetch(`/api/artifacts?name=${game}`)).json();
}

function switchGame(g) {
  game = g;
  $('#g-ssq').classList.toggle('active', g === 'ssq');
  $('#g-dlt').classList.toggle('active', g === 'dlt');
  loadArtifacts().then(() => switchTab(currentTab));
  if (currentTab === 'pipeline') renderPredResult();
}

/* ---------- 初始化 ---------- */
$('#f-task').addEventListener('change', renderTaskParams);
document.querySelectorAll('.tabs button').forEach(b =>
  b.addEventListener('click', () => switchTab(b.dataset.tab)));
$('#g-ssq').addEventListener('click', () => switchGame('ssq'));
$('#g-dlt').addEventListener('click', () => switchGame('dlt'));
window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));

if (window.__echartsFail) $('#echarts-warn').classList.remove('hidden');
renderTaskParams();
renderFlow({ kind: null, stage: 0, running: false, error: null, logs: [] });
pollStatus();  // 页面加载即拉取状态(服务器可能已有任务在跑)
loadArtifacts().then(() => switchTab('overview'));
loadHealth();
setInterval(loadHealth, 5 * 60 * 1000);
