import numpy as np

cfg = {
    'name': 'Num - Cutoff extreme value',
    'group': 'Statistical processing',
    'desc': 'Remove lower or upper extremeties based on percentile thresholds',
    'interval': 60,
    'overlap': 0,
    'inputs': [{'name': 'par', "type": "num"}],
    'options': [
        {'name': 'upper_th', 'init': 90},
        {'name': 'lower_th', 'init': 10}
        ],
    'outputs': [{"type": "num"}]
}


def run(inp, opt, cfg):
    p = inp['par']

    vals = [o['val'] for o in p]

    ut = np.percentile(vals, opt['upper_thres'])
    lt = np.percentile(vals, opt['lower_thres'])

    ret = []
    for o in p:
        if lt <= o['val'] <= ut:
            ret.append(o)

    return [ret]
