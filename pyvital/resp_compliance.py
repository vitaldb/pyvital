import arr
import numpy as np
import math

cfg = {
    'name': 'Resp - Intratidal Compliance Profiles',
    'group': 'Medical algorithms',
    'desc': 'Calculate intratidal compliance using gliding-SLICE method',
    'reference': 'Schumann et al, Estimating intratidal nonlinearity of respiratory system mechanics: a model study using the enhanced gliding-SLICE method. Physiological measurement, 30 (2009) 1341-56',
    'overlap': 0,
    'interval': 10,
    'inputs': [
        {'name': 'volume', 'type': 'wav'},
        {'name': 'flow', 'type': 'wav'},
        {'name': 'awp', 'type': 'wav'}
    ],
    'outputs': [
        {'name': 'v', 'type': 'num', 'min': 0, 'max': 600, 'unit': 'mL'},
        {'name': 'c', 'type': 'num', 'min': 0, 'max': 100, 'unit': 'mL/cmH2O'},
        {'name': 'r', 'type': 'num', 'min': 0, 'max': 20, 'unit': 'cmH2Osec/L'},
        {'name': 'p0', 'type': 'num', 'min': 0, 'max': 30, 'unit': 'cmH2O'}
        ]
}


def run(inp, opt, cfg):
    """
    calculate ppv from arterial waveform
    :param art: arterial waveform
    :return: max, min, upper envelope, lower envelope, respiratory rate, ppv
    """
    vsrate = inp['volume']['srate']
    psrate = inp['awp']['srate']
    fsrate = inp['flow']['srate']
    if vsrate != psrate or vsrate != fsrate:
        print("sampling rates of volume, flow and awp are different")
        return
    srate = vsrate

    vdata = arr.interp_undefined(inp['volume']['vals'])
    fdata = arr.interp_undefined(inp['flow']['vals'])
    pdata = arr.interp_undefined(inp['awp']['vals'])

    # if srate < 200:
    #     vdata = arr.resample_hz(vdata, srate, 200)
    #     fdata = arr.resample_hz(fdata, srate, 200)
    #     pdata = arr.resample_hz(pdata, srate, 200)
    #     srate = 200

    vdata = np.array(vdata)
    fdata = np.array(fdata) / 60  # L/min -> L/sec
    pdata = np.array(pdata)

    #fdata = np.diff(vdata) * srate / 1000  # make difference to rate
    #vdata = vdata[:-1]  # remove the last sample
    #pdata = pdata[:-1]  # remove the last sample

    vmax = max(vdata)
    vmin = min(vdata)
    v95 = vmax - (vmax - vmin) * 0.1
    v5 = vmin + (vmax - vmin) * 0.1
    vret = []
    cret = []
    rret = []
    p0ret = []

    nstep = 31
    vstep = (v95 - v5) / nstep
    for i in range(nstep):
        # collect data
        vfrom = v5 + vstep * i
        seg_idx = np.logical_and(vfrom < vdata, vdata <= vfrom + vstep)

        if sum(seg_idx) < 3:
            print('number of samples in data seg < 3')
            continue

        pseg = pdata[seg_idx]
        vseg = vdata[seg_idx]
        fseg = fdata[seg_idx]

        A = np.vstack([vseg, fseg, np.ones(len(vseg))]).T
        cinv, r, p0 = np.linalg.lstsq(A, pseg)[0]
        c = 1/cinv

        vret.append({'dt': i * 0.02, 'val': vfrom})
        cret.append({'dt': i * 0.02, 'val': c})
        rret.append({'dt': i * 0.02, 'val': r})
        p0ret.append({'dt': i * 0.02, 'val': p0})

    return [
        #{'dt':0, 'srate':srate, 'vals':list(fdata)},
        vret,
        cret,
        rret,
        p0ret
    ]
