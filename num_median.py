import numpy as np

cfg = {
    'name': 'Num - Median Value',
    'group': 'Statistical processing',
    'desc': 'Calculate median value from numeric track',
    'interval': 60,
    'overlap': 0,
    'inputs': [{'name':'p', "type": "num"}],
    'outputs': [{"type": "num"}]
}


def run(inp, opt, cfg):
    p = inp['p']

    if len(p) == 0:
        return

    ret = np.median([o['val'] for o in p])

    return [[{"dt": cfg['interval'], "val": ret}]]
