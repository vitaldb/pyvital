import math
import numpy as np
import numbers
import scipy
from scipy.signal import butter, filtfilt, argrelextrema, find_peaks

def print_all(data):
    """
    print full array
    """
    print('[' + ', '.join([str(x) for x in data]) + ']')


def corr(a, b):
    """
    correlation coefficient
    """
    return np.corrcoef(a, b)[0, 1]

def max_idx(data, idxfrom = 0, idxto=None):
    idxfrom = max(0, idxfrom)
    if idxto == None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmax(data[idxfrom:idxto])


def min_idx(data, idxfrom = 0, idxto=None):
    idxfrom = max(0, idxfrom)
    if idxto == None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmin(data[idxfrom:idxto])


def get_samples(data, srate, idxes):
    """
    Gets a sample of the wave with the indexes specified by idxes
    returns has a form of [{'dt' :, 'val':}, ...]
    """
    return [{"dt": idx / srate, "val": data[idx]} for idx in idxes]


def is_num(x):
    if not isinstance(x, numbers.Number):
        return False
    return math.isfinite(x)


def exclude_undefined(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a[~np.isnan(a)]

def extend_undefined(a):
    # [np.nan, 1, np.nan, 2, 3, np.nan] -> [ 1.  1. nan  2.  3.  3.]
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    valid_index = np.where(~np.isnan(a))[0]
    first_valid_index, last_valid_index = valid_index[0], valid_index[-1]
    a[:first_valid_index] = a[first_valid_index]
    a[last_valid_index + 1:] = a[last_valid_index]
    return a

def interp_undefined(a):
    valid_mask = ~np.isnan(a)
    return np.interp(np.arange(len(a)), np.arange(len(a))[valid_mask], a[valid_mask])

def ffill(a):
    valid_mask = ~np.isnan(a)
    idx = np.where(valid_mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]

def bfill(a): 
    return ffill(a[::-1])[::-1]

# ffill nan
def replace_undefined(a):
    # [np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, np.nan] -> [1,1,2,3,3,3,4,5,5]
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return ffill(extend_undefined(a))

def detect_window_maxima(a, wind=1):
    # [1, 2, 3, 4, 3, 3] -> [3] (wind=2)
    # [1, 2, 3, 4, 4, 4, 3, 3] -> [3] (wind=2)
    # [1, 2, 3, 4, 3] -> [] (wind=2)
    span = int(wind / 2)
    ret = np.array(argrelextrema(a, np.greater_equal, order=span)[0])
    # remove the first and the last maxima
    # it should be greather than before at least
    valid_ret = []
    for i in ret:
        if i > span and i < len(a) - span:
            if a[i-1] < a[i]:
                valid_ret.append(i)
    return np.array(valid_ret)

def detect_maxima(data, tr=None):
    """
    Find indexes of x such that xk-1 <= x >= xk+1
    data: arr
    tr: percentile threshold (0-100)
    return: sorted peak index above tr
    """
    if tr is None:
        return np.array(argrelextrema(data, np.greater_equal)[0])
    else:
        tval = np.percentile(data, tr)
        ret = []
        for i in argrelextrema(data, np.greater_equal)[0]:
            if data[i] > tval:
                ret.append(i)
        return np.array(ret)
    
    # for i in range(1, len(data) - 1):
    #     if data[i-1] < data[i]: # Increased value compared to previous value
    #         if data[i] > tval:
    #             is_peak = False
    #             for j in range(i+1, len(data)): # find next increase or decrease
    #                 if data[i] == data[j]:
    #                     continue
    #                 if data[i] < data[j]:
    #                     break # value increased  -> not a peak
    #                 if data[i] > data[j]:
    #                     is_peak = True
    #                     break # value decreased -> peak!
    #             if is_peak:
    #                 ret.append(i)
    # return ret


def detect_minima(data, tr=100):
    """
    Find indexes of x such that xk-1 <= x >= xk+1
    x: arr
    tr: percentile threshold (0-100)
    return: detect peak above tr
    """
    tval = np.percentile(data, tr)
    ret = []
    for i in range(1, len(data) - 1):
        if data[i-1] > data[i]: # value decreased
            if data[i] < tval:
                is_nadir = False
                for j in range(i+1, len(data)): # find next increase or decrease
                    if data[i] == data[j]:
                        continue
                    if data[i] > data[j]:
                        break # value increased -> minima!
                    if data[i] < data[j]:
                        is_nadir = True
                        break
                if is_nadir:
                    ret.append(i)
    return ret


def next_power_of_2(x):
    """
    Find power of 2 greater than x
    """
    return 2 ** math.ceil(math.log(x) / math.log(2))


def band_pass(data, srate, fl, fh, order=5):
    if fl > fh:
        return band_pass(data, srate, fh, fl)
    nyq = 0.5 * srate
    b, a = butter(order, [fl / nyq, fh / nyq], btype='band')
    return filtfilt(b, a, data)

# low pass filter
# faster x20 times than that using fft
def low_pass(data, srate, fl, order=5):
    """
    low pass filter
    """
    nyq = 0.5 * srate
    low = fl / nyq
    b, a = butter(order, low, btype='lowpass')
    return filtfilt(b, a, data)

def find_nearest(a, value):
    """
    Find the nearest value in a "sorted" np array
    :param data: array
    :param value: value to find
    :return: nearest value
    
    find_nearest([10,20,30,40,50], 21) -> 20
    find_nearest([10,20,30,40,50], 27) -> 30
    """
    idx = np.searchsorted(a, value)
    if idx > 0 and (idx == len(a) or math.fabs(value - a[idx-1]) < math.fabs(value - a[idx])):
        return a[idx-1]
    else:
        return a[idx]

    """
    Find the nearest value in a np array
    """
    idx = np.abs(np.array(a) - value).argmin()
    return a[idx]

def moving_average3(x, N):
    x = np.pad(x, (N//2, N-1-N//2), mode='edge')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def moving_average2(x, N):  # slowest
    x = np.pad(x, (N//2, N-1-N//2), mode='edge')
    return np.convolve(x, np.ones(N) / float(N), 'valid')

# this is fastest when we can use scipy
def moving_average(x, N):
    x = np.pad(x, (N//2, N-1-N//2), mode='edge')
    return scipy.ndimage.filters.uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]


def detect_qrs(data, srate):
    """
    find qrs and return the indexes
    Pan and Tompkins, A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering BME-32.3 (1985)
    """
    y1 = band_pass(data, srate, 5, 15)  # The qrs value must be at 10-20 hz
    y2 = np.convolve(y1, [-2,-1,0,1,2], 'same')  # derivative
    y3 = np.square(y2)  # square
    y4 = moving_average(y3, int(srate * 0.15))  # moving average filter
    p1 = detect_window_maxima(y4, 0.3 * srate)  # find peaks

    min_distance = int(0.25 * srate)
    p2 = []
    signal_level = 0.0
    noise_level = 0.0
    thval = 0.0
    rrlast = 0
    for peak in p1:  # iterate all peaks
        if y4[peak] > thval and ((len(p2) == 0) or (peak - p2[-1] > 0.3 * srate)):
            p2.append(peak)
            # update signal level with cropping
            signal_level = 0.125 * y4[peak] + 0.875 * signal_level            
            # find the false negatives
            if rrlast > 0:  # if there are more than 9 p2
                if p2[-1] - p2[-2] > int(1.66 * rrlast):
                    missed_section_peaks = p1[(p2[-2] < p1) & (p1 < p2[-1])]
                    peak_candidates = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak - p2[-2] > min_distance and \
                            p2[-1] - missed_peak > min_distance and \
                            y4[missed_peak] > 0.5 * thval:
                            peak_candidates.append(missed_peak)
                    if len(peak_candidates) > 0:
                        missed_peak = peak_candidates[np.argmax(y4[peak_candidates])]
                        p2.append(p2[-1])
                        p2[-2] = missed_peak
                        # we missed some signal
                        signal_level = 0.125 * y4[missed_peak] + 0.875 * signal_level            
                        noise_level = 0
            if len(p2) > 8:
                rrlast = int(np.mean(np.diff(p2[-9:])))
        else:
            noise_level = 0.125 * y4[peak] + 0.875 * noise_level
        
        thval = noise_level + 0.25 * (signal_level - noise_level)
        #print(f'@{peak}, signal_level={signal_level:.3f}, thval={thval:.3f}, noise_level={noise_level:.3f}, y4[peak]={y4[peak]:.3f}')

    # find the nearest extreme within 150ms
    # it should be based on the filterd signal because it is centered
    p3 = []
    ya = np.abs(y1)
    for i in p2:
        cand1 = max_idx(ya, i - int(srate * 0.15), i + int(srate * 0.15))
        cand2 = max_idx(y1, i - int(srate * 0.15), i + int(srate * 0.15))
        if abs(i - cand1) < abs(i - cand2):
            p3.append(cand1)
        else:
            p3.append(cand2)

    return p3
    #return y1, y2, y3, y4, p1, p2, p3

    # find closest peak within 80 ms
    p3 = []
    last = -1
    pcand = detect_window_maxima(data, 0.08 * srate)  # 80 ms local peak
    for x in p2:
        idx_cand = find_nearest(pcand, x)
        if idx_cand != last:
            p3.append(idx_cand)
        last = idx_cand

    # threshold -> 0.5 times the median value of the peak within 10 seconds before and after
    p2 = []
    for idx in p1:
        val = y4[idx]
        peak_vals = []
        for idx2 in p1:
            if abs(idx - idx2) < srate * 10:
                peak_vals.append(y4[idx2])
        th = np.median(peak_vals) * 0.1
        if val >= th:
            p2.append(idx)

    # find closest peak within 80 ms
    p3 = []
    last = -1
    pcand = detect_window_maxima(data, 0.08 * srate)  # 80 ms local peak
    for x in p2:
        idx_cand = find_nearest(pcand, x)
        if idx_cand != last:
            p3.append(idx_cand)
        last = idx_cand

    # remove false positives
    p4 = list(p3)
    i = 0
    while i < len(p4) - 1:
        idx1 = p4[i]
        idx2 = p4[i+1]
        if idx2 - idx1 < 0.2 * srate:  # physiological refractory period of about 200 ms
            if i == 0:
                dele = i
            elif i >= len(p4) - 2:
                dele = i + 1
            else:  # minimize heart rate variability
                idx_prev = p4[i-1]
                idx_next = p4[i+2]
                # find center point distance
                if abs(idx_next + idx_prev - 2 * idx1) > abs(idx_next + idx_prev - 2 * idx2):
                    dele = i
                else:
                    dele = i+1
            p4.pop(dele)
            if dele == i:
                i -= 1
        i += 1
    return p4


def remove_wander_spline(data, srate):
    """
    cubic spline ECG wander removal
    http://jh786t2saqs.tistory.com/416
    http://blog.daum.net/jty71/10850833
    """
    # calculate downslope
    # downslope = [0, 0, 0]
    # for i in range(3, len(data) - 3):
    #     downslope.append(data[i-3] + data[i-1] - data[i+1] - data[i+3])
    # downslope += [0, 0, 0]
    downslope = np.convolve(data, [-1,0,-1,0,1,0,1], 'same')  # calculate downslope

    r_list = detect_qrs(data, srate)  # detect r-peak

    rsize = int(0.060 * srate)  # knots from r-peak
    jsize = int(0.066 * srate)
    knots = []  # indexes of the kot
    for ridx in r_list:
        th = 0.6 * max(downslope[ridx:ridx + rsize])
        for j in range(ridx, ridx + rsize):
            if downslope[j] >= th:  # R detected
                knots.append(j - jsize)
                break

    # cubic spline for every knots
    baseline = [0] * len(data)
    for i in range(1, len(knots)-2):
        x1 = knots[i]
        x2 = knots[i+1]
        y1 = data[x1]
        y2 = data[x2]
        d1 = (data[x2] - data[knots[i-1]]) / (x2 - knots[i-1])
        d2 = (data[knots[i+2]] - data[x1]) / (knots[i+2] - x1)
        a = -2 * (y2-y1) / (x2-x1)**3 + (d2+d1) / (x2-x1)**2
        b = 3 * (y2-y1) / (x2-x1)**2 - (d2+2*d1) / (x2-x1)
        c = d1
        d = y1
        for x in range(x1, x2):
            x_a = (x-x1)  # x-a
            x_a2 = x_a * x_a
            x_a3 = x_a2 * x_a
            baseline[x] = a * x_a3 + b * x_a2 + c * x_a + d

    for i in range(len(data)):
        data[i] -= baseline[i]

    return data


def resample(data, dest_len, avg=False):
    """
    resample the data
    avg: If True, the data is averaged and resampled (slower)
    applied only for downsampling. It is meaningless for upsampling
    """
    if dest_len == 0:
        return []

    src_len = len(data)
    if src_len == 0:
        return np.zeros(dest_len)

    if dest_len == 1: # average
        if avg:
            return np.array([np.mean(data)])
        else:
            return np.array([data[0]])

    if src_len == 1: # copy
        return np.full(dest_len, data[0])

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if src_len == dest_len:
        return np.copy(data)

    if src_len < dest_len:  # upsample -> linear interpolate
        ret = []
        for x in range(dest_len):
            srcx = x / (dest_len - 1) * (src_len - 1) # current position of x
            srcx1 = math.floor(srcx) # index 1 on the original
            srcx2 = math.ceil(srcx) # index 2 on the original
            factor = srcx - srcx1 # how close to index 2
            val1 = data[srcx1]
            val2 = data[srcx2]
            ret.append(val1 * (1 - factor) + val2 * factor)
        return np.array(ret)

    #if src_len > dest_len: # downsample -> nearest or avg
    if avg:
        ret = []
        for x in range(dest_len):
            src_from = int(x * src_len / dest_len)
            src_to = int((x + 1) * src_len / dest_len)
            ret.append(np.mean(data[src_from:src_to]))
        return np.array(ret)

    ret = []
    for x in range(dest_len):
        srcx = int(x * src_len / dest_len)
        ret.append(data[srcx])
    return np.array(ret)


def resample_hz(data, srate_from, srate_to, avg=False):
    dest_len = int(math.ceil(len(data) / srate_from * srate_to))
    return resample(data, dest_len, avg)


def estimate_heart_freq(data, srate, fl=30/60, fh=200/60):
    """
    An automatic beat detection algorithm for pressure signals
    http://www.ncbi.nlm.nih.gov/pubmed/16235652
    data: input signal
    srate: sampling rate
    fl: lower bound of freq
    """
    # Fourier transformed, and squared to obtain a frequency-dependent power
    # estimate psd in data
    p = np.abs(np.fft.fft(data)) ** 2
    maxf = 0
    maxval = 0
    for w in range(len(data)):
        f = w * srate / len(data)
        # add 11 harmonics, which do not exceed double of the default power
        # sampling
        if fl <= f <= fh:
            h = 0  # harmonic pds
            for k in range(1, 12):
                h += min(2 * p[w], p[(k * w) % len(data)])
            if h > maxval:
                maxval = h
                maxf = f
    return maxf


def detect_peaks(data, srate):
    """
    obrain maximum and minimum values from blood pressure or pleth waveform
    the minlist is always one less than the maxlist
    """
    ret = []

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    raw_data = np.copy(data)
    raw_srate = srate

    # resampling rate to 100Hz
    data = resample_hz(data, srate, 100)
    srate = 100

    # upper and lower bound of the heart rate (Hz = /sec)
    # heart rate = hf * 60;
    fh = 200 / 60  # 3.3
    fl = 30 / 60  # 0.5

    # estimate hr
    y1 = band_pass(data, srate, 0.5 * fl, 3 * fh)

    # Divide the entire x into four regions and use the median of these
#  hf = []
#  for(var i = 0; i < 4; i++) {
#        var subw = new Wav(srate, y1.vals.copy(data.length / 4 * i, data.length / 4 * (i+1)));
#        hf[i] = subw.estimate_heart_rate(fl, fh);
#        if(hf[i] == 0) {
#            console.log("HR estimation failed, assume 75");
#            hf[i] = 75 / 60;
#        }
#  }
#    hf = hf.median();

    # Whole heart freq estimation
    hf = estimate_heart_freq(y1, srate)
    if hf == 0:
        print("HR estimation failed, assume 75")
        hf = 75 / 60

    # band pass filter again with heart freq estimation
    y2 = band_pass(data, srate, 0.5 * fl, 2.5 * hf)
    d2 = np.diff(y2)

    # detect peak in gradient
    p2 = detect_maxima(d2, 90)

    # detect real peak
    y3 = band_pass(data, srate, 0.5 * fl, 10 * hf)
    p3 = detect_maxima(y3, 60)

    # find closest p3 that follows p2
    p4 = []
    last_p3 = 0
    for idx_p2 in p2:
        idx_p3 = 0
        for idx_p3 in p3:
            if idx_p3 > idx_p2:
                break
        if idx_p3 != 0:
            if last_p3 != idx_p3:
                p4.append(idx_p3)
                last_p3 = idx_p3

    # nearest neighbor and inter beat interval correction
    # p: location of detected peaks
    pc = []

    # find all maxima before preprocessing
    m = detect_maxima(data, 0)
    m = np.array(m)

    # correct peaks location error due to preprocessing
    last = -1
    for idx_p4 in p4:
        cand = find_nearest(m, idx_p4)
        if cand != last:
            pc.append(cand)
            last = cand

    ht = 1 / hf  # beat interval (sec)

    # correct false negatives (FN)
    # Make sure if there is rpeak not included in the PC.
    i = -1
    while i < len(pc):
        if i < 0:
            idx_from = 0
        else:
            idx_from = pc[i]
        
        if i >= len(pc) - 1:
            idx_to = len(data)-1
        else:
            idx_to = pc[i+1]

        # find false negative and fill it
        if idx_to - idx_from < 1.75 * ht * srate:
            i += 1
            continue

        # It can not be within 0.2 of both sides
        idx_from += 0.2 * ht * srate
        idx_to -= 0.2 * ht * srate

        # Find missing peak and add it
        # find the maximum value from idx_from to idx_to
        idx_max = -1
        val_max = 0
        
        for j in range(np.searchsorted(m, idx_from), len(m)):
            idx_cand = m[j]
            if idx_cand >= idx_to:
                break
            if idx_max == -1 or val_max < data[idx_cand]:
                val_max = data[idx_cand]
                idx_max = idx_cand

        # There is no candidate to this FN. Overtake
        if idx_max != -1:  # add idx_max and restart trom there
            pc.insert(i+1, idx_max)
            i -= 1
        i += 1

    # correct false positives (FP)
    i = 0
    while i < len(pc) - 1:
        idx1 = pc[i]
        idx2 = pc[i+1]
        if idx2 - idx1 < 0.75 * ht * srate:  # false positive
            idx_del = i + 1 # default: delete i+1
            if 1 < i < len(pc) - 2:
                # minimize heart rate variability
                idx_prev = pc[i-1]
                idx_next = pc[i+2]

                # find center point distance
                d1 = abs(idx_next + idx_prev - 2 * idx1)
                d2 = abs(idx_next + idx_prev - 2 * idx2)

                if d1 > d2:
                    idx_del = i
                else:
                    idx_del = i+1

            elif i == 0:
                idx_del = i
            elif i == len(pc) - 2:
                idx_del = i+1

            pc.pop(idx_del)
            i -= 1
        i += 1

    # remove dupilcates
    i = 0
    for i in range(0,  len(pc) - 1):
        if pc[i] == pc[i+1]:
            pc.pop(i)
            i -= 1
        i += 1

    # find nearest peak in real data
    # We downsample x to srate to get maxidxs. ex) 1000 Hz -> 100 Hz
    # Therefore, the position found by maxidx may differ by raw_srate / srate.
    maxlist = []
    ratio = math.ceil(raw_srate / srate)
    for maxidx in pc:
        idx = int(maxidx * raw_srate / srate) # extimated idx -> not precise
        maxlist.append(max_idx(raw_data, idx - ratio - 1, idx + ratio + 1))

    # get the minlist from maxlist
    minlist = []
    for i in range(len(maxlist) - 1):
        minlist.append(min_idx(raw_data, maxlist[i], maxlist[i+1]))

    return [minlist, maxlist]

def estimate_resp_rate(data, srate):
    """
    count-adv algorithm
    doi: 10.1007/s10439-007-9428-1
    """
    filted = band_pass(data, srate, 0.1, 0.5)

    # find maxima
    maxlist = detect_maxima(filted)
    minlist = []  # find minima
    for i in range(len(maxlist) - 1):
        minlist.append(min_idx(data, maxlist[i] + 1, maxlist[i+1]))
    extrema = maxlist + minlist
    extrema.sort()  # min, max, min, max

    while len(extrema) >= 4:
        diffs = []  # diffs of absolute value
        for i in range(len(extrema) - 1):
            diffs.append(abs(filted[extrema[i]] - filted[extrema[i + 1]]))
        th = 0.1 * np.percentile(diffs, 75)
        minidx = np.argmin(diffs)
        if diffs[minidx] >= th:
            break
        extrema.pop(minidx)
        extrema.pop(minidx)

    if len(extrema) < 3:
        print("warning: rr estimation failed, 13 used")
        return 13

    # Obtain both even-numbered or odd-numbered distances
    resp_len = np.mean(np.diff(extrema)) * 2
    rr = 60 * srate / resp_len

    return rr
    # # count-orig algorithm
    # tval = 0.2 * np.percentile(max_vals, 75)  # find 75 percentile value
    #
    # # check the maxima is over 75 percentile
    # max_over = [(maxval > tval) for maxval in max_vals]
    # resp_lens = []
    # for i in range(len(maxlist) - 1):
    #     if max_over[i] and max_over[i+1]:
    #         cnt = 0
    #         minval = 0
    #         for minidx in min_idxs:
    #             if minidx > maxlist[i+1]:
    #                 break
    #             if minidx < maxlist[i]:
    #                 continue
    #             cnt += 1
    #             if cnt > 1:
    #                 break
    #             minval = filted[minidx]
    #
    #         if cnt == 1 and minval < 0:
    #             resp_len = maxlist[i+1] - maxlist[i]
    #             if resp_len > 0:
    #                 resp_lens.append(resp_len)
    # if len(resp_lens) == 0:
    #     print("warning: rr estimation failed, 13 used")
    #     return 13
    #
    # rr = 60 * srate / np.mean(resp_lens)
    #
    # return rr


if __name__ == '__main__':
    import pandas as pd

    df_trks = pd.read_csv("https://api.vitaldb.net/trks")

    caseid = 1
    tid = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'SNUADC/ART')]['tid'].values[0]
    vals = pd.read_csv('https://api.vitaldb.net/' + tid).values

    srate = 500

    art = vals[:,1]
    art = exclude_undefined(art)
    # peaks = detect_peaks(art, srate)
    # print(peaks)
    #import cProfile
    #cProfile.run('detect_peaks(art, srate)')
    #quit()

    idx_start = 3600 * srate
    art = vals[idx_start:idx_start + 8 * srate, 1]
    art = exclude_undefined(art)
    peaks = detect_peaks(art, srate)

    print(peaks)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.plot(art, color='r')
    plt.plot(peaks[0], [art[i] for i in peaks[0]], 'bo')
    plt.plot(peaks[1], [art[i] for i in peaks[1]], 'go')
    plt.show()