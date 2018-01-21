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
        {'name': 'R_peak', 'type': 'num', 'min': 0, 'max': 2},
        {'name': 'PTT_min', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 1000},
        {'name': 'PTT_dmax', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 1000},
        {'name': 'PTT_max', 'type': 'num', 'unit': 'ms', 'min': 100, 'max': 1000}
        ]
    }


def run(inp, opt, cfg):
    ecg_data = arr.interp_undefined(inp['ecg']['vals'])
    ecg_srate = inp['ecg']['srate']

    pleth_data = arr.interp_undefined(inp['pleth']['vals'])
    pleth_srate = inp['pleth']['srate']

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

        ecg_ridx = ecg_rlist[i]
        ecg_ridx_next = ecg_rlist[i+1]

        rdt = ecg_ridx / ecg_srate
        rdt_next = ecg_ridx_next / ecg_srate
        if rdt < cfg['overlap']:
            continue

        # find first min in pleth after rdt in ecg
        found_minidx = 0
        for minidx in pleth_minlist:
            if minidx > rdt * pleth_srate:
                found_minidx = minidx
                break
            elif minidx > rdt_next * pleth_srate:
                break
        if found_minidx == 0:
            continue

        # find first dmax in pleth after rdt in ecg
        found_dmaxidx = 0
        for dmaxidx in pleth_dmaxlist:
            if dmaxidx > rdt * pleth_srate:
                found_dmaxidx = dmaxidx
                break
            elif dmaxidx > rdt_next * pleth_srate:
                break
        if found_dmaxidx == 0:
            continue

        # find first dmax in pleth after rdt in ecg
        found_maxidx = 0
        for maxidx in pleth_maxlist:
            if maxidx > rdt * pleth_srate:
                found_maxidx = maxidx
                break
            elif maxidx > rdt_next * pleth_srate:
                break
        if found_maxidx == 0:
            continue

        max_dt = found_maxidx / pleth_srate
        if max_dt > cfg['interval']:
            continue
        min_dt = found_minidx / pleth_srate
        dmax_dt = found_dmaxidx / pleth_srate

        pttmax_list.append({'dt': max_dt, 'val': (max_dt - rdt) * 1000})
        pttdmax_list.append({'dt': dmax_dt, 'val': (dmax_dt - rdt) * 1000})
        pttmin_list.append({'dt': min_dt, 'val': (min_dt - rdt) * 1000})

    return [arr.get_samples(ecg_data, ecg_srate, ecg_rlist), pttmin_list, pttdmax_list, pttmax_list]
