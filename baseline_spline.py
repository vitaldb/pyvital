import arr

cfg = {
    'name': 'Baseline - Cubic Spline',
    'group': 'Signal processing',
    'desc': 'Remove baseline wander using cubic spline curve',
    'reference': 'VB Romero, ECG baseline wander removal and noise suppression in an embedded platform',
    'overlap': 2,
    'interval': 20,
    'inputs': [{"name": 'ecg', "type": 'wav'}],
    'outputs': [{"name": 'ecg_filtered', "type":'wav'}]
}


def run(inp, opt, cfg):
    srate = inp['ecg']['srate']
    data = arr.interp_undefined(inp['ecg']['vals'])

    ret = arr.remove_wander_spline(data, srate)

    return [
        {"srate": srate, "vals": ret}
    ]
