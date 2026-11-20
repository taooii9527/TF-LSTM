"""L3 API 契约测试: flask test_client 遍历全部端点。"""

import pytest


@pytest.fixture(scope='module')
def client():
    from lottery import webapp as W
    return W.app.test_client()


def _get(client, path):
    r = client.get(path)
    assert r.status_code == 200, f'{path} -> {r.status_code}'
    return r.get_json()


def test_overview_contract(client, cfg):
    d = _get(client, f'/api/overview?name={cfg.name}')
    assert d['ok'] and d['draws'] > 500
    rnd = d['randomness']
    assert len(rnd['parity']['counts']) == cfg.red_count + 1
    assert len(rnd['size']['theo']) == cfg.red_count + 1
    assert 0 <= rnd['consec_rate'] <= 1


def test_analysis_contract(client, cfg):
    d = _get(client, f'/api/analysis?name={cfg.name}')
    assert d['ok']
    assert len(d['red_heat']) == cfg.red_max
    assert len(d['blue_heat']) == cfg.blue_max
    m = d.get('model')
    assert m and {'red_top', 'blue_top', 'sum_range', 'odd_ratio', 'big_ratio'} <= set(m)
    assert len(m['red_top']) == 8 and len(m['blue_top']) == 3


def test_forecast_record_contract(client, cfg):
    d = _get(client, f'/api/forecast-record?name={cfg.name}')
    assert 'ok' in d and 'record' in d


def test_artifacts_predict_nonempty(client, cfg):
    """生产事故回归: artifacts 的 predict.groups 不允许静默为空。"""
    d = _get(client, f'/api/artifacts?name={cfg.name}')
    assert d['ok']
    assert len(d['predict']['groups']) >= 1, \
        f"{cfg.name} 存在预测记录但 artifacts.predict.groups 为空"


def test_ledger_consistency(client, cfg):
    d = _get(client, f'/api/ledger?name={cfg.name}')
    if not d['ok']:
        pytest.skip('无回测明细')
    assert d['invest'] == d['steps'] * 2
    assert d['net'] == d['prize'] - d['invest']
    assert abs(d['ret_rate'] - d['prize'] / d['invest']) < 1e-9


def test_health_contract(client):
    d = _get(client, '/api/health')
    assert isinstance(d['ok'], bool)
    assert len(d['checks']) >= 2
    for game in d['checks']:
        assert {'game', 'ok', 'items'} <= set(game)
        for item in game['items']:
            assert {'name', 'passed', 'detail'} <= set(item)


def test_bad_game_rejected(client):
    r = client.get('/api/overview?name=hack')
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        assert r.get_json()['ok'] is False
