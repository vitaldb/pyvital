import arr
import numpy as np
import math

cfg = {
    'name': 'ECG - Heart Rate Variability',
    'group': 'Medical algorithms',
    'desc': 'Calculate Heart Rate Variability. Approximately 60-second data is required for calculating HF component and 120-second for LF. To calculate VLF, a longer signal is needed.',
    'reference': 'Heart rate variability. Standards of measurement, physiological interpretation, and clinical use. European Heart Journal (1996)17,354-381',
    'overlap': 2,  # 2 sec overlap for HR=30
    'interval': 300,  # 5 min
    'inputs': [{'name': 'ecg', 'type': 'wav'}],
    'outputs': [
        {'name': 'SDNN', 'type': 'num', 'unit': 'ms', 'min': 0, 'max': 100},
        {'name': 'RMSSD', 'type': 'num', 'unit': 'ms', 'min': 0, 'max': 10},
        {'name': 'pNN50', 'type': 'num', 'unit': '%', 'min': 0, 'max': 5},
        {'name': 'NNI', 'type': 'num', 'unit': 'ms', 'min': 500, 'max': 2500},
        {'name': 'TP', 'type': 'num', 'unit': 'ms2', 'min': 0, 'max': 200000},
        {'name': 'VLF', 'type': 'num', 'unit': 'ms2', 'min': 0, 'max': 200000},
        {'name': 'LF', 'type': 'num', 'unit': 'ms2', 'min': 0, 'max': 10000},
        {'name': 'HF', 'type': 'num', 'unit': 'ms2', 'min': 0, 'max': 10000},
        {'name': 'LF_HF', 'type': 'num', 'unit': '', 'min': 0, 'max': 100}
        ]
}


def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['ecg']['vals'])
    srate = inp['ecg']['srate']

    rlist = arr.detect_qrs(data, srate)  # detect r-peaks

    # remove qrs before and after overlap
    new_rlist = []
    for ridx in rlist:
        if cfg['overlap'] <= ridx / srate:
            new_rlist.append(ridx)
    rlist = new_rlist

    ret_rpeak = [{'dt': ridx / srate, 'val': 1} for ridx in rlist]

    # average qrs
    qrs_width = int(0.1 * srate)
    qrslist = []
    for ridx in rlist:
        qrslist.append(data[ridx - qrs_width: ridx + qrs_width])
    avg_qrs = np.mean(np.array(qrslist), axis=0)

    # correlation coefficient
    celist = []
    for qrs in qrslist:
        ce = arr.corr(qrs, avg_qrs)
        celist.append(ce)

    # rr interval (ms)
    rri_list = np.diff(rlist) / srate * 1000

    nni_list = []  # nn interval (ms)
    ret_nni = []
    for i in range(len(rlist) - 1):
        if celist[i] < 0.9 or celist[i+1] < 0.9:
            continue

        # median RR interval nearest 10 beats
        med_rri = np.median(rri_list[max(0, i-5): min(len(rri_list), i+5)])

        rri = rri_list[i]

        if med_rri * 0.5 <= rri <= med_rri * 1.5:
            nni_list.append(rri)
            ret_nni.append({'dt': rlist[i+1] / srate, 'val': rri})

    # make time domain nni_data function by linear interpolation (200 hz)
    nni_srate = 200
    nni_data = [None] * int(math.ceil(len(data) / srate * nni_srate))
    for nni in ret_nni:
        nni_data[int(nni['dt'] * nni_srate)] = nni['val']
    nni_data = arr.interp_undefined(nni_data)

    # hamming window
    nni_data *= np.hamming(len(nni_data))

    vlf = 0  # <= 0.04 Hz
    lf = 0  # 0.04-0.15 Hz
    hf = 0  # 0.15-0.4 Hz

    # A power spectral density (PSD) takes the amplitude of the FFT, multiplies it by its complex conjugate and normalizes it to the frequency bin width.
    # This allows for accurate comparison of random vibration signals that have different signal lengths.
    psd = abs(np.fft.fft(nni_data)) ** 2 / (len(nni_data) * nni_srate)  # power density per bin (ms2/hz) from fft
    psd *= 2  #  In order to conserve the total power,
    # multiply all frequencies that occur in both sets -- the positive and negative frequencies -- by a factor of 2.
    # Zero frequency (DC) and the Nyquist frequency do not occur twice
    for k in range(len(nni_data)):
        f = k * nni_srate / len(nni_data)
        if f < 0.0033:
            pass
        elif f < 0.04:
            vlf += psd[k]
        elif f < 0.15:
            lf += psd[k]
        elif f < 0.4:
            hf += psd[k]
        else:
            break

    # multiply the width (hz) to get the area under curve
    vlf *= nni_srate / len(nni_data)
    lf *= nni_srate / len(nni_data)
    hf *= nni_srate / len(nni_data)
    tp = vlf + lf + hf

    # lf_arrnorm = 0
    # hf_arrnorm = 0
    # if tp - vlf:
    #     lf_arrnorm = lf / (tp-vlf)
    #     hf_arrnorm = hf / (tp-vlf)

    lf_hf = 0
    if hf:
        lf_hf = lf / hf

    sdnn = np.std(nni_list)

    dnni_list = abs(np.diff(nni_list))  # Difference between adjacent nn intervals
    nn50 = 0
    ret_dnni = []
    for i in range(len(dnni_list)):
        dnni = dnni_list[i]
        if dnni > 50:
            nn50 += 1
        ret_dnni.append({'dt': ret_nni[i+1]['dt'], 'val': dnni})

    pnn50 = nn50 * 100 / len(dnni_list)

    rmssdnni = 0
    if len(dnni_list) > 0:
        for dnni in dnni_list:
            rmssdnni += dnni * dnni
        rmssdnni = (rmssdnni / len(dnni_list)) ** 0.5

    dt_last = cfg['interval']
    return [
        [{'dt': dt_last, 'val': sdnn}],
        [{'dt': dt_last, 'val': rmssdnni}],
        [{'dt': dt_last, 'val': pnn50}],
        ret_nni,
        [{'dt': dt_last, 'val': tp}],
        [{'dt': dt_last, 'val': vlf}],
        [{'dt': dt_last, 'val': lf}],
        [{'dt': dt_last, 'val': hf}],
        [{'dt': dt_last, 'val': lf_hf}]
        ]
