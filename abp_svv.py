import arr
import numpy as np
import math

cfg = {
    'name': 'ABP - Stroke Volume Variation',
    'group': 'Medical algorithms',
    'desc': 'Calculate respiratory deviation of standard deviation and Liljestrad-Zander parameter of the ABP',
    'reference': '',
    'overlap': 3,
    'interval': 30,
    'inputs': [{'name': 'art1', 'type': 'wav'}, {'name': 'vent_rr', 'type': 'num'}],
    'outputs': [
        {'name': 'std', 'type': 'num', 'min':0, 'max':20, 'unit':'mmHg'},
        {'name': 'svv_std', 'type': 'num', 'min':0, 'max':20, 'unit':'%'},
        {'name': 'lz', 'type': 'num', 'min':0, 'max':0.5, 'unit':''},
        {'name': 'svv_lz', 'type': 'num', 'min':0, 'max':20, 'unit':'%'}
        ],
    'svv_std': 0,
    'svv_lz': 0
}


def run(inp, opt, cfg):
    """
    calculate svv from arterial waveform
    :param art: arterial waveform
    :return: max, min, upper envelope, lower envelope, respiratory rate, ppv
    """
    data = arr.interp_undefined(inp['art1']['vals'])
    srate = inp['art1']['srate']

    data = arr.resample_hz(data, srate, 100)
    srate = 100

    if len(data) < 30 * srate:
        return [[], [], [], []]

    minlist, maxlist = arr.detect_peaks(data, srate)
    maxlist = maxlist[1:]  # make the same length

    # calculate each beat's std and put it at the peak time
    stds = []
    lzs = []
    for i in range(len(minlist) - 1):
        maxidx = maxlist[i]

        beat = data[minlist[i]:minlist[i+1]]
        if max(beat) - min(beat) < 20:
            continue

        s = np.std(beat)
        stds.append({'dt':maxidx / srate, 'val': s})

        sbp = np.max(beat)
        dbp = beat[0]
        lz = (sbp-dbp) / (sbp+dbp)  # 0.1~0.3
        lzs.append({'dt':maxidx / srate, 'val': lz})

    # estimates resp rate
    rr = np.median([o['val'] for o in inp['vent_rr']])
    if not rr > 1:
        return [[], [], [], []]

    # split by respiration
    nsamp_in_breath = int(srate * 60 / rr)
    m = int(len(data) / nsamp_in_breath)  # m segments exist

    # std
    svv_stds = []
    for i in range(m - 1): # 50% overlapping
        this_breath_stds = []
        for j in range(len(stds)):
            if i * nsamp_in_breath <= stds[j]['dt'] * srate < (i+2) * nsamp_in_breath:
                this_breath_stds.append(stds[j]['val'])
        svmax = np.max(this_breath_stds)
        svmin = np.min(this_breath_stds)

        svv_stde = 2 * (svmax - svmin) * 100 / (svmax + svmin)  # estimate
        if svv_stde > 40 or svv_stde < 0:
            continue
        svv_stds.append(svv_stde)

    svv_stde = np.median(svv_stds)
    if svv_stde < 0:
        svv_stde = 0

    svv_std = cfg['svv_std']
    if svv_std == 0 or svv_std is None:
        svv_std = svv_stde
    err = abs(svv_stde - svv_std)
    if err < 5:
        svv_std = svv_stde
    elif err < 25:
        svv_std = (svv_std + svv_stde) / 2
    else:
        pass  # dont update
    cfg['svv_std'] = svv_std

    # lz
    svv_lzs = []
    for i in range(m - 1): # 50% overlapping
        this_breath_lzs = []
        for j in range(len(lzs)):
            if i * nsamp_in_breath <= lzs[j]['dt'] * srate < (i+2) * nsamp_in_breath:
                this_breath_lzs.append(lzs[j]['val'])
        svmax = np.max(this_breath_lzs)
        svmin = np.min(this_breath_lzs)

        svv_lze = 2 * (svmax - svmin) * 100 / (svmax + svmin)  # estimate

        if svv_lze > 40 or svv_lze < 0:
            continue
        svv_lzs.append(svv_lze)

    svv_lze = np.median(svv_lzs)
    if svv_lze < 0:
        svv_lze = 0

    svv_lz = cfg['svv_lz']
    if svv_lz == 0 or svv_lz is None:
        svv_lz = svv_lze
    err = abs(svv_lze - svv_lz)
    if err < 5:
        svv_lz = svv_lze
    elif err < 25:
        svv_lz = (svv_lz + svv_lze) / 2
    else:
        pass  # dont update
    cfg['svv_lz'] = svv_lz

    return [
        stds,
        [{'dt': cfg['interval'], 'val': svv_std}],
        lzs,
        [{'dt': cfg['interval'], 'val': svv_lz}]
    ]
