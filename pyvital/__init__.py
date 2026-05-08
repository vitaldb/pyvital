import math
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, argrelextrema


def corr(a, b):
    """
    correlation coefficient
    """
    return np.corrcoef(a, b)[0, 1]

def max_idx(data, idxfrom=0, idxto=None):
    idxfrom = max(0, idxfrom)
    if idxto == None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmax(data[idxfrom:idxto])


def min_idx(data, idxfrom=0, idxto=None):
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


def interp_undefined(a):
    if not isinstance(a, np.ndarray):
        # Convert None values to np.nan for consistent handling
        a = [x if x is not None else np.nan for x in a]
        a = np.array(a)
    valid_mask = ~np.isnan(a)
    if not np.any(valid_mask):
        return a
    return np.interp(np.arange(len(a)), np.arange(len(a))[valid_mask], a[valid_mask])


def _detect_window_maxima(a, wind=1):
    span = int(wind / 2)
    ret = np.array(argrelextrema(a, np.greater_equal, order=span)[0])
    valid_ret = []
    for i in ret:
        if i > span and i < len(a) - span:
            if a[i-1] < a[i]:
                valid_ret.append(i)
    return np.array(valid_ret)


def _detect_maxima(data, tr=None):
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


def band_pass(data, srate, fl, fh, order=5):
    if fl > fh:
        return band_pass(data, srate, fh, fl)
    nyq = 0.5 * srate
    b, a = butter(order, [fl / nyq, fh / nyq], btype='band')
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


def _moving_average(x, N):
    x = np.pad(x, (N//2, N-1-N//2), mode='edge')
    return scipy.ndimage.filters.uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]


def detect_qrs(data, srate):
    """Find QRS R-peaks and return their sample indices.

    Delegates to ``openecg.detect_qrs`` (gradient-thresholded detector,
    micro-F1 = 0.994 on MIT-BIH Arrhythmia DB at 100 ms tolerance — beats
    the prior pyvital Pan-Tompkins implementation by +5.7 F1 points across
    all 48 records). Validation script: openecg/scripts/validate_qrs_mitdb.py.

    Algorithm: Makowski's gradient-thresholded QRS detector (originally
    in NeuroKit2, vendored under MIT license; see ``openecg/qrs.py``).
    """
    from openecg import detect_qrs as _openecg_detect_qrs
    return _openecg_detect_qrs(data, srate).tolist()


def remove_wander_spline(data, srate):
    """
    cubic spline ECG wander removal
    http://jh786t2saqs.tistory.com/416
    http://blog.daum.net/jty71/10850833
    """
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

    # Whole heart freq estimation
    hf = estimate_heart_freq(y1, srate)
    if hf == 0:
        print("HR estimation failed, assume 75")
        hf = 75 / 60

    # band pass filter again with heart freq estimation
    y2 = band_pass(data, srate, 0.5 * fl, 2.5 * hf)
    d2 = np.diff(y2)

    # detect peak in gradient
    p2 = _detect_maxima(d2, 90)

    # detect real peak
    y3 = band_pass(data, srate, 0.5 * fl, 10 * hf)
    p3 = _detect_maxima(y3, 60)

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
    m = _detect_maxima(data, 0)
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
    maxlist = list(_detect_maxima(filted))
    minlist = []  # find minima
    for i in range(len(maxlist) - 1):
        minlist.append(min_idx(data, maxlist[i] + 1, maxlist[i+1]))
    extrema = sorted(maxlist + minlist)

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
