import arr
import numpy as np
import math
import copy

cfg = {
    'name': 'ECG - T-wave alternans',
    'group': 'Medical algorithms',
    'desc': 'Calculate T-wave alternance',
    'reference': 'Narayan SM1, Smith JM. Spectral analysis of periodic fluctuations in electrocardiographic repolarization. IEEE Trans Biomed Eng. 1999 Feb;46(2):203-12.',
    'overlap': 1.5,  # for HR=40
    'interval': 60*5,  # 5 min
    'inputs': [{'name': 'ecg', 'type': 'wav'}],
    'outputs': [
        {'name': 'ecg_filtd', 'type':'wav'},
        {'name': 'avg_beat', 'type':'wav'},
        {'name': 'peaks', 'type': 'num'},
        {'name': 'twa volt', 'type': 'num', 'unit': 'uv', 'min': 0, 'max': 10},
        {'name': 'twa ratio', 'type': 'num', 'unit': '', 'min': 0, 'max': 10},
        {'name': 'median twa volt', 'type': 'num', 'unit': 'uv', 'min': 0, 'max': 10},
        {'name': 'median twa ratio', 'type': 'num', 'unit': '', 'min': 0, 'max': 10}
        ]
}


def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['ecg']['vals'])
    srate = inp['ecg']['srate']

    ecg_500 = arr.resample(data, math.ceil(len(data) / srate * 500))  # resample to 500 Hz
    srate = 500
    ecg_filt = arr.band_pass(ecg_500, srate, 0.01, 100)  # filtering
    ecg_filt = arr.remove_wander_spline(ecg_filt, srate)  # remove baseline wander

    r_list = arr.detect_qrs(ecg_filt, srate) # detect r-peak
    new_r_list = []
    for ridx in r_list:  # remove qrs before and after overlap
        if cfg['overlap'] <= ridx / srate:
            new_r_list.append(ridx)
    r_list = new_r_list

    ret_rpeak = []
    for ridx in r_list:
        ret_rpeak.append({'dt': ridx / srate })

    if len(r_list) < 128:
        print("# of beats "+str(len(r_list))+" < 128")
        return [{'srate': srate, 'vals':ecg_filt.tolist()}, {}, ret_rpeak, [], [], [], []]

    # for each segments
    twavs = []
    twars = []
    ret_twav = []
    ret_twar = []
    nseg = 0
    ret_avg_beat = {'srate': srate, 'vals': [0] * len(ecg_500)}

    for seg_start in range(0, len(r_list) - 128, 32):  # Separates in 128-bit units regardless of input length
        nseg += 1

        hrs = [] # calculate hrs
        for i in range(127):
            hr = srate / (r_list[seg_start + i + 1] - r_list[seg_start + i])
            hrs.append(hr)

        if max(hrs) - min(hrs) > 20:
            print('seg ' + nseg + ' excluded HR diff > ' + diff_hr)
            continue

        # only -250 to 350 ms from R peak
        idx_r = int(math.floor(0.25 * srate))
        beat_len = int(math.ceil(0.6 * srate))
        beats = []
        for i in range(128):
            ridx = r_list[seg_start + i]
            beat = ecg_filt[ridx - idx_r:ridx - idx_r + beat_len]
            beats.append(beat)
        beats = np.array(beats)

        # remove each beat's baseline voltage
        # for(i = 0 i < 128 i++)
        #         idx_base = beats[i].min_idx(idx_r - Math.floor(0.15 * srate), idx_r)
        #         minstd = 999999
        #         for(j = idx_base - Math.floor(0.03 * srate) j < idx_base + Math.floor(0.03 * srate) j++)
        #                 # The baseline is the point at which the standard deviation of around 15ms is minimized.
        #                 curstd = beats[i].std(j - Math.floor(0.007 * srate), j + Math.floor(0.007 * srate))
        #                 if(curstd < minstd)
        #                         idx_base = j
        #                         minstd = curstd
        #         val_base = beats[i].mean(idx_base - Math.floor(0.007 * srate), idx_base + Math.floor(0.007 * srate))
        #         beats[i].sub_inplace(val_base)

        # calculate average beat
        avg_beat = np.mean(beats, axis=0)  # average beat of the 128 beats

        # find minimum values from avg_beat in both sides
        idx_start = idx_r - int(0.15 * srate)
        idx_end = idx_r + int(0.1 * srate)
        idx_base = np.argmin(avg_beat[idx_start:idx_r])
        min_right = np.min(avg_beat[idx_r:idx_end])

        # find minimum std value for finding avg_beat's baseline
        minstd = 999999
        for j in range(idx_base - int(0.03 * srate), idx_base + int(0.03 * srate)):
            idx_from = max(0, j - int(0.007 * srate))
            idx_to = min(len(avg_beat), j + int(0.007 * srate))
            curstd = np.std(avg_beat[idx_from:idx_to])
            if curstd < minstd:
                idx_base = j
                minstd = curstd
        min_left = np.mean(avg_beat[idx_base - int(0.007 * srate):idx_base + int(0.007 * srate)])

        # threshold = 5% of max val
        th_left = min_left + 0.05 * (avg_beat[idx_r] - min_left)
        th_right = min_right + 0.05 * (avg_beat[idx_r] - min_right)
        idx_qrs_start = idx_r - int(0.05 * srate)
        idx_qrs_end = idx_r + int(0.05 * srate)
        for j in range(idx_r, idx_r - int(0.1 * srate), -1): # idx_r = 110 ms
            if avg_beat[j] < th_left:
                idx_qrs_start = j
                break
        for j in range(idx_r, idx_r + int(0.1 * srate)):
            if avg_beat[j] < th_right:
                idx_qrs_end = j
                break

        # find offset with maximum correlation
        offsets = []
        qrs_coeffs = []
        offset_width = int(0.006 * srate) # range for finding offset
        for i in range(128):
            maxoffset = -offset_width
            maxce = -999999
            for offset in range(-offset_width, offset_width + 1):
                ce = arr.corr(avg_beat[idx_qrs_start:idx_qrs_end],
                              beats[i][offset + idx_qrs_start:offset + idx_qrs_end])
                if maxce < ce:
                    maxoffset = offset
                    maxce = ce
            offsets.append(maxoffset)
            qrs_coeffs.append(maxce)

        # move beats by offsets
        new_beats = []
        for i in range(128):
            ost = offsets[i]
            beat = beats[i].tolist()
            if ost < 0:
                beat = [0] * -ost + beat[:ost]
            else:
                beat = beat[ost:] + [0] * ost
            new_beats.append(beat)

        beats = np.array(new_beats)
        avg_beat = np.mean(beats, axis=0)

        # peak alignment
        # idx_r = 125
        # len(avg_beat) = 300
        for i in range(128):
            beats[i] += avg_beat[idx_r] - beats[i][idx_r]

        # reject correlation qrs_coeffsicient < 0.95
        nreplaced = 0
        for i in range(128):
            if qrs_coeffs[i] < 0.95:
                nreplaced += 1
                beats[i] = copy.deepcopy(avg_beat)
                offsets[i] = 0

        if nreplaced > 25:
            print('excluded VPC > ' + str(nreplaced))
            continue

        # recalculate average beat
        avg_beat = np.mean(beats, axis=0)

        # gather 128 beats
        mat = {}
        for idx in range(-idx_r, beat_len - idx_r):  # idx = sample number from peak
            mat[idx] = []
            for i in range(128):
                mat[idx].append(beats[i][idx + idx_r])

        # power spectrums of 128 beats
        spect = {}
        for idx in mat:
            spect[idx] = abs(np.fft.fft(mat[idx])) ** 2  # each 128 beat fft result

        # t-wave - cumulative spectrum
        cum_spect = np.array([0.0] * 128)
        cum_cnt = 0
        for idx in spect:
            t = idx / srate
            if 0.100 <= t < 0.250: # add values between 100 and 250 ms from rpeak
                cum_spect += spect[idx]
                cum_cnt += 1

        # print waves
#        for(i = 0 i < 128 i++)
#            print("plot([" + beats[i].copy(offsets[i] + offset_width) + "], '"+((i%2)?'r':'b')+"')")

        spect_noise = cum_spect[57:62]  # noise level: 0.44-0.49 (57-62)

        alternans_peak = cum_spect[64]
        avg_noise = np.mean(spect_noise)
        std_noise = np.std(spect_noise)

        for j in range(len(avg_beat)):
            ret_avg_beat['vals'][r_list[seg_start + 127] + j] = avg_beat[j]

        if alternans_peak > avg_noise:
            twav = 1000 * (alternans_peak - avg_noise) ** 0.5 / cum_cnt  # 150 ms * 0.5 sample/msec (500hz)
            twar = (alternans_peak - avg_noise) / std_noise
            twavs.append(twav)
            twars.append(twar)
            ret_twav.append({'dt': r_list[seg_start + 127] / srate, 'val': twav})
            ret_twar.append({'dt': r_list[seg_start + 127] / srate, 'val': twar})

    dt_last = r_list[-1] / srate

    return [
        {'srate': srate, 'vals': ecg_filt.tolist()},
        ret_avg_beat, ret_rpeak, ret_twav, ret_twar,
        [{'dt': dt_last, 'val': np.median(twavs)}],
        [{'dt': dt_last, 'val': np.median(twars)}]
    ]
