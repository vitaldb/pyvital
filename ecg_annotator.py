import numpy as np
import math
import arr
import copy
import pywt


cfg = {
    'name': 'ECG - Annotator',
    'group': 'Medical algorithms',
    'desc': 'ECG annotator based on YC Chesnokov\'s implementation',
    'reference': 'YC Chesnokov, D Nerukh, RC Glen, Individually Adaptable Automatic QT Detector',
    'overlap': 3,  # 2 sec overlap for HR=30
    'interval': 30,
    'inputs': [{'name': 'ecg', 'type': 'wav'}],
    'outputs': [{'name': 'ann', 'type': 'str', 'unit': ''}],
    'license': 'GPL'
}


def minimax(data):
    return np.std(data) * (0.3936 + 0.1829 * math.log(len(data)))


def denoise(data, wsize):
    # hard minmax denoise
    for i in range(0, len(data), wsize):
        iend = min(len(data), i + wsize)
        th = minimax(data[i: iend])
        for j in range(i, iend):
            if abs(data[j]) <= th:
                data[j] = 0


def cwt(data, srate, wname, freq):
    scale = 0.16 * srate / freq  # for gaus1
    sig = pywt.cwt(data, [scale], wname)[0].flatten()
    return sig


def qmf(w):
    ret = []
    for i in range(len(w)):
        if i % 2 == 1:
            ret.append(-w[len(w)-1-i])
        else:
            ret.append(w[len(w)-1-i])
    return ret


def orthfilt(w):
    lor = w / np.linalg.norm(w)
    lod = lor[::-1]
    hir = qmf(lor)
    hid = hir[::-1]
    return [lod, hid, lor, hir]


def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['ecg']['vals'])
    srate = inp['ecg']['srate']

    min_hr = 40     # min bpm
    max_hr = 200    # max bpm
    min_qrs = 0.04   # min qslist duration
    max_qrs = 0.2    # max qslist duration
    min_umv = 0.2    # min UmV of R,S peaks
    min_pq = 0.07    # min PQ duration
    max_pq = 0.20    # max PQ duration
    min_qt = 0.21    # min QT duration
    max_qt = 0.48    # max QT duration
    pfreq = 9.0     # cwt Hz for pidx wave
    tfreq = 2.5     # cwt Hz for tidx wave
    min_sq = (60.0 / max_hr) - max_qrs  # from s to next q
    if min_sq * srate <= 0:
        min_sq = 0.1
        max_hr = int(60.0 / (max_qrs + min_sq))

    # denoised ecg
    depth = int(math.ceil(np.log2(srate / 0.8))) - 1
    ad = pywt.wavedec(data, 'db2', level=depth)
    ad[0].fill(0)  # low frequency approx -> 0
    ecg_denoised = pywt.waverec(ad, 'db2')

    # interpolation filter
    inter1 = pywt.Wavelet('inter1', filter_bank=orthfilt([0.25, 0.5, 0.25]))

    # qrs augmented ecg
    sig = cwt(data, srate, 'gaus1', 13)  # 13 Hz gaus convolution
    depth = int(math.ceil(np.log2(srate / 23))) - 2
    ad = pywt.wavedec(sig, inter1, level=depth)
    for level in range(depth):  # remove [0-30Hz]
        wsize = int(2 * srate / (2 ** (level+1)))  # 2 sec window
        denoise(ad[depth-level], wsize)  # Remove less than 30 hz from all detail
    ad[0].fill(0)  # most lowest frequency approx -> 0
    ecg_qrs = pywt.waverec(ad, inter1)

    # start parsing
    qslist = []  # qrs list [startqrs, endqrs, startqrs, endqrs, ...]
    vpclist = []  # abnormal beat

    # save greater than 0 after min_sq
    prev_zero = 0
    ipos = 0
    while ipos < len(ecg_qrs) - int(max_qrs * srate):
        if ecg_qrs[ipos] == 0:
            prev_zero += 1
        else:
            if prev_zero > min_sq * srate:
                iend = ipos + int(max_qrs * srate)  # find the position of the end of the current qrs
                while iend > ipos:
                    if ecg_qrs[iend] != 0:
                        break
                    iend -= 1

                # Check if it is the minimum length or if there is a pause
                if ipos + min_qrs * srate > iend or np.any(ecg_qrs[iend + 1:iend + 1 + int(min_sq * srate)]):
                    vpclist.append(ipos)  # push vpc
                else:
                    qslist.append(ipos)
                    qslist.append(iend)

                ipos = iend
            prev_zero = 0
        ipos += 1

    # qlist = [qslist[i] for i in range(0, len(qslist), 2)]

    complist = []
    for n in range(int(len(qslist) / 2)):
        start_qrs = qslist[n * 2]
        end_qrs = qslist[n * 2 + 1]

        qidx = -1
        ridx = arr.max_idx(ecg_denoised, start_qrs, end_qrs)
        if ecg_denoised[ridx] < min_umv:
            ridx = -1

        sidx = arr.min_idx(ecg_denoised, start_qrs, end_qrs)
        if -ecg_denoised[sidx] < min_umv:
            sidx = -1

        # ridxpeak > 0mV sidxpeak < 0mV
        if ridx != -1 and sidx != -1:
            if sidx < ridx:  # check for sidx
                if ecg_denoised[ridx] > -ecg_denoised[sidx]:
                    qidx = sidx
                    sidx = arr.min_idx(ecg_denoised, ridx, end_qrs + 1)
                    if sidx == ridx or sidx == end_qrs or abs(ecg_denoised[end_qrs] - ecg_denoised[sidx]) < 0.05:
                        sidx = -1
            else:  # check for qidx
                qidx = arr.min_idx(ecg_denoised, start_qrs, ridx + 1)
                if qidx == ridx or qidx == start_qrs or abs(ecg_denoised[start_qrs] - ecg_denoised[qidx]) < 0.05:
                    qidx = -1
        elif sidx != -1:  # only sidx --> Find small r if only sidx detected  in rsidx large tidx lead
            ridx = arr.max_idx(ecg_denoised, start_qrs, sidx + 1)
            if ridx == sidx or ridx == start_qrs or abs(ecg_denoised[start_qrs] - ecg_denoised[ridx]) < 0.05:
                ridx = -1
        elif ridx != -1:  # only ridx --> Find small q,s
            qidx = arr.min_idx(ecg_denoised, start_qrs, ridx + 1)
            if qidx == ridx or qidx == start_qrs or abs(ecg_denoised[start_qrs] - ecg_denoised[qidx]) < 0.05:
                qidx = -1
            sidx = arr.min_idx(ecg_denoised, ridx, end_qrs + 1)
            if sidx == ridx or sidx == end_qrs or abs(ecg_denoised[end_qrs] - ecg_denoised[sidx]) < 0.05:
                sidx = -1
        else:
            vpclist.append(start_qrs)
            continue

        o = {'q': qslist[n*2], 's': qslist[n*2+1]}  # always exists

        if qidx != -1:
            o['q'] = qidx
        if ridx != -1:
            o['r'] = ridx
        if sidx != -1:
            o['s'] = sidx

        complist.append(o)

    # for each QRS --> find tidx and pidx wave
    for n in range(len(complist) - 1):
        pree = complist[n]['q']
        nows = complist[n]['s']
        nowe = complist[n+1]['q']
        size = nowe - nows  # s-q interval
        size = int(min(size, srate * max_qt - (nows - pree)))

        rr = (nowe - pree) / srate
        if (60.0 / rr < min_hr) or (60.0 / rr > max_hr - 20):
            continue

        # all are in this
        block = [data[nows + i] for i in range(size)]

        ecg_qrs = cwt(block, srate, 'gaus1', tfreq)
        tidx1 = arr.min_idx(ecg_qrs) + nows
        tidx2 = arr.max_idx(ecg_qrs) + nows
        if tidx1 > tidx2:
            tidx1, tidx2 = tidx2, tidx1

        # additional constraints on [tidx1 tidx tidx2] duration, symmetry, QT interval
        ist = False
        if ecg_qrs[tidx1-nows] < 0 < ecg_qrs[tidx2-nows]:
            ist = True
        elif ecg_qrs[tidx1-nows] > 0 > ecg_qrs[tidx2-nows]:
            ist = True

        if ist:
            if (tidx2 - tidx1) >= 0.09 * srate: # and (tidx2-tidx1)<=0.24 * srate)   #check for tidx wave duration
                ist = True                # QT interval = .4 * sqrt(RR)
                if min_qt * srate <= (tidx2 - pree) <= max_qt * srate:
                    ist = True
                else:
                    ist = False
            else:
                ist = False

        if ist:
            tidx = 0 # zero crossing
            sign = (ecg_qrs[tidx1-nows] >= 0)
            for i in range(tidx1 - nows, tidx2 - nows):
                if sign == (ecg_qrs[i] >= 0):
                    continue
                tidx = i + nows
                break

            # check for tidx wave symetry
            if tidx2 - tidx < tidx - tidx1:
                ratio = (tidx2 - tidx) / (tidx - tidx1)
            else:
                ratio = (tidx - tidx1) / (tidx2 - tidx)
            if ratio < 0.4:
                ist = False

        if ist:
            tmin = arr.min_idx(data, tidx1, tidx2)
            tmax = arr.max_idx(data, tidx1, tidx2)
             # find the most nearest values from 0-cross, tmin, tmax
            tidx = arr.find_nearest((tidx, tmin, tmax), (tidx2 + tidx1) / 2)
            complist[n]['(t'] = tidx1
            complist[n]['t'] = tidx
            complist[n]['t)'] = tidx2

        # search for P-WAVE
        size = nowe - nows  # s-q interval
        size = int(min(size, srate * max_pq))

        if ist:
            if tidx2 > nowe - size - int(0.04 * srate):   # isp wnd far from Twave at least on 0.04 sec
                size -= tidx2 - (nowe - size - int(0.04 * srate))

        nskip = (nowe - nows) - size

        if size <= 0.03 * srate:
            continue  # impresize QRS begin detection

        block = [data[nows + nskip + i] for i in range(size)]

        ecg_qrs = cwt(block, srate, 'gaus1', pfreq)
        p1 = arr.min_idx(ecg_qrs) + nows + nskip
        p2 = arr.max_idx(ecg_qrs) + nows + nskip
        if p1 > p2:
            p1, p2 = p2, p1

        # additional constraints on [p1 pidx p2] duration, symmetry, PQ interval
        isp = False
        if ecg_qrs[p1-nows-nskip] < 0 < ecg_qrs[p2-nows-nskip]:
            isp = True
        elif ecg_qrs[p1-nows-nskip] > 0 > ecg_qrs[p2-nows-nskip]:
            isp = True

        if isp:
            if 0.03 * srate <= (p2 - p1) <= 0.15 * srate:  # check for pidx wave duration  9Hz0.03 5Hz0.05
                isp = (min_pq * srate <= (nowe - p1) <= max_pq * srate)  # PQ interval = [0.07 - 0.12,0.20]
            else:
                isp = False

        if not isp:
            continue

        pidx = 0  # zero crossing
        sign = (ecg_qrs[p1-nows-nskip] >= 0)
        for i in range(p1 - nows - nskip, p2 - nows - nskip):
            if sign == (ecg_qrs[i] >= 0):
                continue
            pidx = i + nows + nskip
            break

        # check for pidx wave symetry
        if p2 - pidx < pidx - p1:
            ratio = (p2 - pidx) / (pidx - p1)
        else:
            ratio = (pidx - p1) / (p2 - pidx)

        if ratio < 0.4:
            isp = False  # not a p wave
        if isp:
            complist[n]['(p'] = p1
            complist[n]['p'] = pidx
            complist[n]['p)'] = p2

    # add annotation
    ret_ann = []
    for n in range(len(complist)):
        for k, v in complist[n].items():
            if k == 'q' and abs(ecg_denoised[v]) > 0.5:
                k = 'Q'
            elif k == 'r' and abs(ecg_denoised[v]) > 0.5:
                k = 'R'
            elif k == 's' and abs(ecg_denoised[v]) > 0.5:
                k = 'S'
            elif k == '(t':
                k = '(T'
            elif k == 't':
                k = 'T'
            elif k == 't)':
                k = 'T)'
            elif k == '(p':
                k = '(P'
            elif k == 'p':
                k = 'P'
            elif k == 'p)':
                k = 'P)'

            ret_ann.append({"dt": v / srate, "val": k})

    for n in range(len(vpclist)):
        ret_ann.append({"dt": vpclist[n] / srate, "val": 'A'})

    return [ret_ann]
