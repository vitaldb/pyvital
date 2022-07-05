import pyvital.arr as arr
import numpy as np
import time
import scipy.interpolate
import scipy.signal

last_ppv = 0
last_spv = 0

cfg = {
    'name': 'ART - Pulse Pressure Variation',
    'group': 'ABP',
    'desc': 'Calculate pulse pressure variation using modified version of the method in the reference',
    'reference': 'Aboy et al, An Enhanced Automatic Algorithm for Estimation of Respiratory Variations in Arterial Pulse Pressure During Regions of Abrupt Hemodynamic Changes. IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 56, NO. 10, OCTOBER 2009',
    'overlap': 20,
    'interval': 30, # 30초는 되어야 rr을 추정 가능함
    'inputs': [{'name': 'ART', 'type': 'wav'}],
    'outputs': [
        {'name': 'PPV', 'type': 'num', 'min': 0, 'max': 30, 'unit': '%'},
        {'name': 'SPV', 'type': 'num', 'min': 0, 'max': 30, 'unit': '%'},
        {'name': 'ART_RR', 'type': 'num', 'min': 0, 'max': 30, 'unit': '/min'}
        ]
}


def run(inp, opt, cfg):
    """
    calculate ppv from arterial waveform
    :param art: arterial waveform
    :return: max, min, upper envelope, lower envelope, respiratory rate, ppv
    """
    global last_ppv, last_spv

    data = arr.interp_undefined(inp['ART']['vals'])
    srate = inp['ART']['srate']

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
    pp_vals = []
    sp_vals = []
    for i in range(0, len(minlist)-1):
        if beats_128[i] is None or not len(beats_128[i]):
            continue
        if np.corrcoef(avgbeat, beats_128[i])[0, 1] < 0.9:
            continue
        pp = data[maxlist[i]] - data[minlist[i]]  # pulse pressure
        sp = data[maxlist[i]]
        pp_vals.append({'dt': minlist[i] / srate, 'val': pp})
        sp_vals.append({'dt': minlist[i] / srate, 'val': sp})

    dtstart = time.time()

    # estimates resp rate
    # upper env
    idx_start = max(min(minlist),min(maxlist))
    idx_end = min(max(minlist),max(maxlist))
    xa = scipy.interpolate.CubicSpline(maxlist, [data[idx] for idx in maxlist])(np.arange(idx_start, idx_end))

    # lower env
    xb = scipy.interpolate.CubicSpline(minlist, [data[idx] for idx in minlist])(np.arange(idx_start, idx_end))
    rr = arr.estimate_resp_rate(xa-xb, srate)

    dtend = time.time()
    #print('rr {}'.format(rr))

    # split by respiration
    nsamp_in_breath = int(srate * 60 / rr)
    m = int(len(data) / nsamp_in_breath)  # m segments exist

    raw_pps = []
    raw_sps = []
    ppvs = []
    spvs = []
    for ibreath in np.arange(0, m - 1, 0.5):
        pps_breath = []
        sps_breath = []

        for ppe in pp_vals:
            if ibreath * nsamp_in_breath < ppe['dt'] * srate < (ibreath + 1) * nsamp_in_breath:
                pps_breath.append(ppe['val'])

        for spe in sp_vals:
            if ibreath * nsamp_in_breath < spe['dt'] * srate < (ibreath + 1) * nsamp_in_breath:
                sps_breath.append(spe['val'])

        if len(pps_breath) < 4:
            continue

        if len(sps_breath) < 4:
            continue

        pp_min = min(pps_breath)
        pp_max = max(pps_breath)
        sp_min = min(sps_breath)
        sp_max = max(sps_breath)

        ppv = (pp_max - pp_min) / (pp_max + pp_min) * 200
        if not 0 < ppv < 50:
            continue

        spv = (sp_max - sp_min) / (sp_max + sp_min) * 200
        if not 0 < spv < 50:
            continue

        # kalman filter
        if last_ppv == 0: # first time
            last_ppv = ppv
        elif abs(last_ppv - ppv) <= 1.0:
            ppv = last_ppv
        elif abs(last_ppv - ppv) <= 25.0:  # ppv cannot be changed abruptly
            ppv = (ppv + last_ppv) * 0.5
            last_ppv = ppv
        else:
            continue

        if last_spv == 0: # first time
            last_spv = spv
        elif abs(last_spv - spv) <= 1.0:
            spv = last_spv
        elif abs(last_spv - spv) <= 25.0:  # ppv cannot be changed abruptly
            spv = (spv + last_spv) * 0.5
            last_spv = spv
        else:
            continue

        ppvs.append(ppv)
        spvs.append(spv)

    median_ppv = np.median(ppvs)
    median_spv = np.median(spvs)

    return [
        [{'dt': cfg['interval'], 'val': median_ppv}],
        [{'dt': cfg['interval'], 'val': median_spv}],
        [{'dt': cfg['interval'], 'val': rr}]
    ]
