import arr
import numpy as np

cfg = {
    'name': 'Pleth - Pulse Transit Time',
    'group': 'Medical algorithms',
    'desc': 'Calculate pulse transit time.',
    'reference': '',
    'overlap': 5,
    'interval': 30,
    'inputs': [{'name': 'ecg', 'type': 'wav'}, {'name': 'pleth', 'type': 'wav'}],
    'outputs': [
        {'name': 'PTT_min', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 500},
        {'name': 'PTT_dmax', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 500},
        {'name': 'PTT_max', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 500},
        {'name': 'R_peak', 'type': 'num', 'min': 0, 'max': 2}
        ]
    }


def run(inp, opt, cfg):
    ecg_data = arr.interp_undefined(inp['ecg']['vals'])
    ecg_srate = inp['ecg']['srate']

    pleth_data = arr.interp_undefined(inp['pleth']['vals'])
    pleth_srate = inp['pleth']['srate']
    pleth_data = arr.band_pass(pleth_data, pleth_srate, 0.5, 15)

    ecg_rlist = arr.detect_qrs(ecg_data, ecg_srate)
    pleth_minlist, pleth_maxlist = arr.detect_peaks(pleth_data, pleth_srate)

    dpleth = np.diff(pleth_data)
    pleth_dmaxlist = [] # index of the maximum slope between peak and nadir in pleth
    for i in range(len(pleth_minlist)):  # maxlist is one less than minlist
        dmax_idx = arr.max_idx(dpleth, pleth_minlist[i], pleth_maxlist[i+1])
        pleth_dmaxlist.append(dmax_idx)

    pttmax_list = []
    pttmin_list = []
    pttdmax_list = []
    for i in range(len(ecg_rlist) - 1):
        if len(pleth_minlist) == 0:
            continue
        if len(pleth_maxlist) == 0:
            continue

        rpeak_dt = ecg_rlist[i] / ecg_srate
        rpeak_dt_next = ecg_rlist[i+1] / ecg_srate
        if rpeak_dt < cfg['overlap']:
            continue

        # find first min in pleth after rpeak_dt in ecg
        found_minidx = 0
        for minidx in pleth_minlist:
            if minidx > rpeak_dt * pleth_srate:
                found_minidx = minidx
                break
            elif minidx > rpeak_dt_next * pleth_srate:
                break
        if found_minidx == 0:
            continue

        # find first dmax in pleth after rpeak_dt in ecg
        found_dmaxidx = 0
        for dmaxidx in pleth_dmaxlist:
            if dmaxidx > rpeak_dt * pleth_srate:
                found_dmaxidx = dmaxidx
                break
            elif dmaxidx > rpeak_dt_next * pleth_srate:
                break
        if found_dmaxidx == 0:
            continue

        # find first dmax in pleth after rpeak_dt in ecg
        found_maxidx = 0
        for maxidx in pleth_maxlist:
            if maxidx > rpeak_dt * pleth_srate:
                found_maxidx = maxidx
                break
            elif maxidx > rpeak_dt_next * pleth_srate:
                break
        if found_maxidx == 0:
            continue

        max_dt = found_maxidx / pleth_srate
        if max_dt > cfg['interval']:
            continue
        min_dt = found_minidx / pleth_srate
        dmax_dt = found_dmaxidx / pleth_srate

        pttmax_list.append({'dt': max_dt, 'val': (max_dt - rpeak_dt) * 1000})
        pttdmax_list.append({'dt': dmax_dt, 'val': (dmax_dt - rpeak_dt) * 1000})
        pttmin_list.append({'dt': min_dt, 'val': (min_dt - rpeak_dt) * 1000})

    return [
        pttmin_list, 
        pttdmax_list, 
        arr.get_samples(ecg_data, ecg_srate, ecg_rlist), 
        pttmax_list]
