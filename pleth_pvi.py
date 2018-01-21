import arr
import numpy as np
import math

cfg = {
    'name': 'PVI - Plethysmographic Variability Index',
    'group': 'Medical algorithms',
    'desc': 'Calculate pulse pressure variation',
    'reference': 'Aboy et al, An Enhanced Automatic Algorithm for Estimation of Respiratory Variations in Arterial Pulse Pressure During Regions of Abrupt Hemodynamic Changes. IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 56, NO. 10, OCTOBER 2009',
    'overlap': 3,
    'interval': 40,
    'inputs': [{'name': 'pleth', 'type': 'wav'}],
    'outputs': [{'name': 'rr', 'type': 'num', 'min': 0, 'max': 30, 'unit': '/min'}, {'name': 'pvi', 'type': 'num', 'min': 0, 'max': 30, 'unit': '%'}],
    'pp': 0
}


def b(u):
    if -5 <= u <= 5:
        return math.exp(-u * u / 2)
    else:
        return 0


def run(inp, opt, cfg):
    """
    calculate ppv from arterial waveform
    :param art: arterial waveform
    :return: max, min, upper envelope, lower envelope, respiratory rate, ppv
    """
    data = arr.interp_undefined(inp['pleth']['vals'])
    srate = inp['pleth']['srate']

    data = arr.resample_hz(data, srate, 100)
    srate = 100

    if len(data) < 30 * srate:
        return [{}, {}, {}, {}, {}, [], []]

    minlist, maxlist = arr.detect_peaks(data, srate)
    maxlist = maxlist[1:]

    # estimates the upper ue(n) and lower le(n) envelopes
    xa = np.array([data[idx] for idx in minlist])
    le = np.array([0] * len(data))
    for i in range(len(data)):
        be = np.array([b((i - idx) / (0.2 * srate)) for idx in minlist])
        s = sum(be)
        if s != 0:
            le[i] = np.dot(xa, be) / s

    xb = np.array([data[idx] for idx in maxlist])
    ue = np.array([0] * len(data))
    for i in range(len(data)):
        be = np.array([b((i - idx) / (0.2 * srate)) for idx in maxlist])
        s = sum(be)
        if s != 0:
            ue[i] = np.dot(xb, be) / s

    re = ue - le
    re[re < 0] = 0

    # estimates resp rate
    rr = arr.estimate_resp_rate(re, srate)

    # split by respiration
    nsamp_in_breath = int(srate * 60 / rr)
    m = int(len(data) / nsamp_in_breath)  # m segments exist
    pps = []
    for i in range(m - 1):
        imax = arr.max_idx(re, i * nsamp_in_breath, (i+2) * nsamp_in_breath)  # 50% overlapping
        imin = arr.min_idx(re, i * nsamp_in_breath, (i+2) * nsamp_in_breath)
        ppmax = re[imax]
        ppmin = re[imin]
        ppe = 2 * (ppmax - ppmin) / (ppmax + ppmin) * 100  # estimate
        if ppe > 50 or ppe < 0:
            continue

        pp = cfg['pp']
        if pp == 0:
            pp = ppe

        err = abs(ppe - pp)
        if err < 1:
            pp = ppe
        elif err < 25:
            pp = (pp + ppe) / 2
        else:
            pass  # dont update

        cfg['pp'] = pp

        pps.append({'dt': (i * nsamp_in_breath) / srate, 'val': pp})

    return [
        [{'dt': cfg['interval'], 'val': rr}],
        pps
    ]
