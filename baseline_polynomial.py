import numpy as np
import arr

cfg = {
    'name': 'Baseline - Polynomial',
    'group': 'Signal processing',
    'desc': "Remove baseline wander using polynomial curve",
    'reference': "Chouhan, V.S. Computing: Theory and Applications, 2007. ICCTA '07. International Conference on",
    'interval': 30,
    'inputs': [{'name': 'ecg', 'type': 'wav'}],
    'outputs': [
        {'type': 'wav', 'name': 'ecg_filtered'},
        {'type': 'wav', 'name': 'baseline'}
    ]
}


def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['ecg']['vals'])
    srate = inp['ecg']['srate']

    span = int(srate * 10)  # Cut every 10 second regardless of interval
    baseline = [0] * len(data)
    x = np.arange(0, span)
    for spos in range(0, len(data), span):  # start position of this segment
        if spos + span > len(data):
            span = len(data) - spos
            x = x[0:span]

        med = np.median(data[spos:spos + span])  # compute overall median for the entire waveform
        for j in range(span):
            data[spos + j] -= med  # shift each sample of the entire waveform by this median value
        y = data[spos:spos + span]
        p = np.polyfit(x, y, 4)
        y = np.polyval(p, x)

        for j in range(span):
            baseline[spos + j] = y[j]
            data[spos + j] -= y[j]

    r_list = arr.detect_qrs(data, srate) # detect r-peak

    for i in range(len(r_list)-1):  # for each rr interval
        idx1 = r_list[i]
        idx2 = r_list[i+1]
        if idx1 + 1 > idx2:
            continue
        med = np.median(data[idx1 + 1:idx2])
        for j in range(idx1 + 1, idx2):
            data[j] -= med

    return [
        {'srate': srate, 'vals': data},
        {'srate': srate, 'vals': baseline}
    ]
