import arr
import numpy as np
from math import factorial

cfg = {
    'name': 'EEG - Frequency Analysis',
    'group': 'Medical algorithms',
    'desc': 'Frequency Analysis of EEG.',
    'reference': '',
    'overlap': 58,
    'interval': 60,
    'inputs': [{'name': 'eeg', 'type': 'wav'}],
    'outputs': [
        {'name': 'TOTPOW', 'type': 'num', 'unit': 'dB', 'min': 0, 'max': 100},
        {'name': 'SEF', 'type': 'num', 'unit': 'Hz', 'min': 0, 'max': 30},
        {'name': 'MF', 'type': 'num', 'unit': 'Hz', 'min': 0, 'max': 30},

        {'name': 'DELTA', 'type': 'num', 'unit': '%', 'min': 0, 'max': 100},
        {'name': 'THETA', 'type': 'num', 'unit': '%', 'min': 0, 'max': 100},
        {'name': 'ALPHA', 'type': 'num', 'unit': '%', 'min': 0, 'max': 100},
        {'name': 'BETA', 'type': 'num', 'unit': '%', 'min': 0, 'max': 100},
        {'name': 'GAMMA', 'type': 'num', 'unit': '%', 'min': 0, 'max': 100}
        ]
    }

# http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
     try:
         window_size = np.abs(np.int(window_size))
         order = np.abs(np.int(order))
     except ValueError:
         raise ValueError("window_size and order have to be of type int")

     if window_size % 2 != 1 or window_size < 1:
         raise TypeError("window_size size must be a positive odd number")

     if window_size < order + 2:
         raise TypeError("window_size is too small for the polynomials order")

     order_range = range(order+1)
     half_window = (window_size -1) // 2
     # precompute coefficients
     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
     # pad the signal at the extremes with values taken from the signal itself
     firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
     y = np.concatenate((firstvals, y, lastvals))
     return np.convolve(m[::-1], y, mode='valid')

def smooth(y):
    #return butter_bandpass(y, 0.5, 50, 128)
    #return lowess(y)
    return savitzky_golay(y, window_size=91, order=3)

def fromhz(f, fres):
    # if type(f) is np.array:
    #     return (f / fres).astype(int)
    return int(f / fres)

def tohz(i, fres):
    return fres * i

def repcols(v, nreps):
    return np.tile(v[:, None], nreps)

def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['eeg']['vals'])
    data -= smooth(np.array(data))
    srate = int(inp['eeg']['srate'])
    nfft = srate * 2  # srate * epoch size
    fres = srate / nfft  # frequency resolution (hz)

    # frequency domain analysis
    EPOCH_SIZE = int(srate * 2)
    STRIDE_SIZE = int(srate * 0.5)
    ps = []
    for epoch_start in range(0, len(data) - EPOCH_SIZE + 1, STRIDE_SIZE):  # 0.5초 마다 겹침
        epoch_w = data[epoch_start:epoch_start + EPOCH_SIZE]  # 2초 epoch
        epoch_w = (epoch_w - np.mean(epoch_w)) * np.blackman(EPOCH_SIZE)  # detrend and windowing
        dft = np.fft.fft(epoch_w)[:srate]  # 실수를 fft 했으므로 절반만 필요하다
        dft[0] = 0  # dc 성분은 지움
        ps.append(2 * np.abs(dft) ** 2) # 파워의 절대값인데 절반 날렸으므로
    ps = np.mean(np.array(ps), axis=0)
    pssum = np.cumsum(ps)  # cummulative sum
    pssum = pssum[1:]
    totpow = pssum[fromhz(30, fres)]
    sef = tohz(np.argmax(pssum > 0.95 * totpow), fres)
    mf = tohz(np.argmax(pssum > 0.5 * totpow), fres)

    delta = pssum[fromhz(4, fres) - 1] / pssum[-1] * 100
    theta = (pssum[fromhz(8, fres) - 1] - pssum[fromhz(4, fres)]) / pssum[-1] * 100
    alpha = (pssum[fromhz(12, fres) - 1] - pssum[fromhz(8, fres)]) / pssum[-1] * 100
    beta = (pssum[fromhz(30, fres) - 1] - pssum[fromhz(12, fres)]) / pssum[-1] * 100
    gamma = (pssum[-1] - pssum[fromhz(30, fres)]) / pssum[-1] * 100

    # pttmax_list.append()
    # pttdmax_list.append({'dt': dmax_dt, 'val': (dmax_dt - rpeak_dt) * 1000})
    # pttmin_list.append({'dt': min_dt, 'val': (min_dt - rpeak_dt) * 1000})
    #
    return [
        [{'dt': cfg['interval'], 'val': 10 * np.log10(totpow)}],
        [{'dt': cfg['interval'], 'val': sef}],
        [{'dt': cfg['interval'], 'val': mf}],
        [{'dt': cfg['interval'], 'val': delta}],
        [{'dt': cfg['interval'], 'val': theta}],
        [{'dt': cfg['interval'], 'val': alpha}],
        [{'dt': cfg['interval'], 'val': beta}],
        [{'dt': cfg['interval'], 'val': gamma}]
    ]
    # pttmin_list,
    # pttdmax_list,
    # arr.get_samples(ecg_data, ecg_srate, ecg_rlist),
    # pttmax_list]
