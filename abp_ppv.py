import arr
import numpy as np
import math

last_ppv = 0;

cfg = {
    'name': 'ABP - Pulse Pressure Variation',
    'group': 'Medical algorithms',
    'desc': 'Calculate pulse pressure variation using modified version of the method in the reference',
    'reference': 'Aboy et al, An Enhanced Automatic Algorithm for Estimation of Respiratory Variations in Arterial Pulse Pressure During Regions of Abrupt Hemodynamic Changes. IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 56, NO. 10, OCTOBER 2009',
    'overlap': 3,
    'interval': 30,
    'inputs': [{'name': 'art1', 'type': 'wav'}],
    'outputs': [
        {'name': 'ppv', 'type': 'num', 'min': 0, 'max': 30, 'unit': '%'},
        {'name': 'pulse_val', 'type': 'num', 'min': 0, 'max': 100, 'unit': 'mmHg'},
        {'name': 'rr', 'type': 'num', 'min': 0, 'max': 30, 'unit': '/min'}
        ]
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
    global last_ppv

    data = arr.interp_undefined(inp['art1']['vals'])
    srate = inp['art1']['srate']

    data = arr.resample_hz(data, srate, 100)
    srate = 100

    if len(data) < 30 * srate:
        print('hr < 30')
        return

    # beat detection
    minlist, maxlist = arr.detect_peaks(data, srate)
    maxlist = maxlist[1:]

    # beat lengths
    beatlens = []
    beats_128 = []
    beats_128_valid = []
    for i in range(0, len(minlist)-1):
        beatlen = minlist[i+1] - minlist[i]  # in samps
        if not 30 < beatlen < 300:
            beats_128.append(None)
            continue

        pp = data[maxlist[i]] - data[minlist[i]]  # pulse pressure
        if not 20 < pp < 100:
            beats_128.append(None)
            continue

        beatlens.append(beatlen)
        beat = data[minlist[i]:minlist[i+1]]
        resampled = arr.resample(beat, 128)
        beats_128.append(resampled)
        beats_128_valid.append(resampled)

    if not beats_128_valid:
        return

    avgbeat = np.array(beats_128_valid).mean(axis=0)

    meanlen = np.mean(beatlens)
    stdlen = np.std(beatlens)
    if stdlen > meanlen * 0.2: # irregular rhythm
        return

    # remove beats with correlation < 0.9
    pulse_vals = []
    for i in range(0, len(minlist)-1):
        if not beats_128[i]:
            continue
        if np.corrcoef(avgbeat, beats_128[i])[0, 1] < 0.9:
            continue
        pp = data[maxlist[i]] - data[minlist[i]]  # pulse pressure
        pulse_vals.append({'dt': minlist[i] / srate, 'val': pp})

    # estimates the upper env(n) and lower env(n) envelopes
    xa = np.array([data[idx] for idx in minlist])
    lower_env = np.array([0.0] * len(data))
    for i in range(len(data)):
        be = np.array([b((i - idx) / (0.2 * srate)) for idx in minlist])
        s = sum(be)
        if s != 0:
            lower_env[i] = np.dot(xa, be) / s

    xb = np.array([data[idx] for idx in maxlist])
    upper_env = np.array([0.0] * len(data))
    for i in range(len(data)):
        be = np.array([b((i - idx) / (0.2 * srate)) for idx in maxlist])
        s = sum(be)
        if s != 0:
            upper_env[i] = np.dot(xb, be) / s

    pulse_env = upper_env - lower_env
    pulse_env[pulse_env < 0.0] = 0.0

    # estimates resp rate
    rr = arr.estimate_resp_rate(pulse_env, srate)

    # split by respiration
    nsamp_in_breath = int(srate * 60 / rr)
    m = int(len(data) / nsamp_in_breath)  # m segments exist
    raw_pps = []
    pps = []
    for ibreath in np.arange(0, m - 1, 0.5):
        pps_breath = []
        for ppe in pulse_vals:
            if ibreath * nsamp_in_breath < ppe['dt'] * srate < (ibreath + 1) * nsamp_in_breath:
                pps_breath.append(ppe['val'])
        if len(pps_breath) < 4:
            continue

        pp_min = min(pps_breath)
        pp_max = max(pps_breath)

        ppv = 2 * (pp_max - pp_min) / (pp_max + pp_min) * 100  # estimate
        if not 0 < ppv < 50:
            continue

#       raw_pps.append({'dt': (ibreath * nsamp_in_breath) / srate, 'val': pp})
        #
        # kalman filter
        if last_ppv == 0: # first time
            last_ppv = ppv
        elif abs(last_ppv - ppv) <= 1.0:
            ppv = last_ppv
        elif abs(last_ppv - ppv) <= 25.0:  # ppv cannot be changed abruptly
            ppv = (ppv + last_ppv) * 0.5
            last_ppv = ppv
        else:
            continue  # no update

        pps.append({'dt': ((ibreath + 1) * nsamp_in_breath) / srate, 'val': int(ppv)})

    return [
        pps,
        pulse_vals,
        [{'dt': cfg['interval'], 'val': rr}]
    ]
