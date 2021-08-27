from .. import arr
import numpy as np

cfg = {
    'name': 'Wav - Mean Value',
    'group': 'Statistical processing',
    'desc': 'Mean value of samples in wave',
    'interval': 10,
    'overlap': 0,
    'inputs': [{'name': 'w', 'type': 'wav'}],
    'outputs': [{'type': 'num'}]
}


def run(inp, opt, cfg):
    data = arr.exclude_undefined(inp['w']['vals'])

    ret = np.mean(data)

    return [
        [{"dt": cfg['interval'], "val": ret}]
    ]
