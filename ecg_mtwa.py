import arr
import numpy as np
import math
import copy
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

cfg = {
    'name': 'ECG - T-wave alternans',
    'group': 'Medical algorithms',
    'desc': 'Calculate microvolt T-wave alternans',
    'reference': 'Narayan SM1, Smith JM. Spectral analysis of periodic fluctuations in electrocardiographic repolarization. IEEE Trans Biomed Eng. 1999 Feb;46(2):203-12.',
    'overlap': 1.5,  # for HR=40
    'interval': 60 * 5,  # 5 min
    'inputs': [{'name': 'ecg', 'type': 'wav'}],
    'outputs': [
        {'name': 'ecg_filtd', 'type': 'wav'},
        {'name': 'avg_beat', 'type': 'wav'},
        {'name': 'peaks', 'type': 'num'},
        {'name': 'twa_volt', 'type': 'num', 'unit': 'uv', 'min': 0, 'max': 100},
        {'name': 'twa_ratio', 'type': 'num', 'unit': '', 'min': 0, 'max': 10}
    ]
}


def run(inp, opt, cfg):
    data = arr.interp_undefined(inp['ecg']['vals'])
    srate = inp['ecg']['srate']

    ecg_500 = data
    if srate != 500:
        ecg_500 = arr.resample(data, math.ceil(len(data) / srate * 500))  # resample to 500 Hz
        srate = 500
    ecg_filt = arr.band_pass(ecg_500, srate, 0.01, 100)  # filtering
    ecg_filt = arr.remove_wander_spline(ecg_filt, srate)  # remove baseline wander

    r_list = arr.detect_qrs(ecg_filt, srate)  # detect r-peak
    new_r_list = []
    for ridx in r_list:  # remove qrs before and after overlap
        if cfg['overlap'] <= ridx / srate:
            new_r_list.append(ridx)
    r_list = new_r_list

    ret_rpeak = []
    for ridx in r_list:
        ret_rpeak.append({'dt': ridx / srate})

    segbeats = 128
    segsteps = 32  # int(segbeats/4)

    # for each segments
    twavs = []
    twars = []
    ret_twav = []
    ret_twar = []
    ret_avg_beat = {'srate': srate, 'vals': [0] * len(ecg_500)}

    iseg = 0
    for seg_start in range(0, len(r_list) - segbeats,
                           segsteps):  # Separates in 128-beat units regardless of input length
        iseg += 1

        hrs = []  # calculate hrs
        for i in range(segbeats - 1):
            hr = srate / (r_list[seg_start + i + 1] - r_list[seg_start + i])
            hrs.append(hr)

        if max(hrs) - min(hrs) > 20:
            # print('seg ' + iseg + ' excluded HR diff > ' + diff_hr)
            continue

        # only -250 to 350 ms from R peak
        idx_r = int(0.25 * srate)  # idx_r == 125
        beat_len = int(0.6 * srate)  # beat_len == 300
        beats = []
        for i in range(segbeats):
            ridx = r_list[seg_start + i]
            beat = ecg_filt[ridx - idx_r:ridx - idx_r + beat_len]
            beats.append(beat)
        beats = np.array(beats)

        # remove each beat's baseline voltage
        # no effect because of R peak leveling is below
        # Baseline correction included estimation of the baseline in the isoelectric PQ
        # segment by averaging 16 successive samples in this time window
        pq_width = int(0.008 * srate)
        # for i in range(segbeats):
        #     idx_base = arr.min_idx(beats[i], idx_r - int(0.15 * srate), idx_r)
        #     min_std = 999999
        #     for j in range(idx_base - int(0.03 * srate), idx_base + int(0.03 * srate)):
        #         # The baseline is the point at which the standard deviation of around 15ms is minimized.
        #         this_std = np.std(beats[i][j - pq_width:j + pq_width])
        #         if this_std < min_std:
        #             idx_base = j
        #             min_std = this_std
        #     beats[i] -= np.mean(beats[i][idx_base - pq_width:idx_base + pq_width])

        # calculate average beat
        avg_beat = np.mean(beats, axis=0)  # average beat of the segbeats beats

        # find minimum values from avg_beat in both sides
        idx_start = idx_r - int(0.15 * srate)  # idx_start == 50
        idx_end = idx_r + int(0.1 * srate)  # idx_end == 175

        idx_base = arr.min_idx(avg_beat, idx_start, idx_r)  # avg_beat's baseline
        min_std = 999999  # find minimum std value
        for j in range(idx_base - int(0.03 * srate), idx_base + int(0.03 * srate)):
            idx_from = max(0, j - pq_width)
            idx_to = min(len(avg_beat), j + pq_width)
            this_std = np.std(avg_beat[idx_from:idx_to])
            # print("{} {}".format(j, this_std))
            if this_std < min_std:
                idx_base = j
                min_std = this_std
        # print("idx_base={}", idx_base)
        min_left = np.mean(avg_beat[idx_base - pq_width:idx_base + pq_width])
        min_right = np.min(avg_beat[idx_r:idx_end])

        # threshold = 5% of max val
        th_left = min_left + 0.05 * (avg_beat[idx_r] - min_left)
        th_right = min_right + 0.05 * (avg_beat[idx_r] - min_right)
        idx_qrs_start = idx_r - int(0.05 * srate)
        idx_qrs_end = idx_r + int(0.05 * srate)
        for j in range(idx_r, idx_r - int(0.1 * srate), -1):  # idx_r = 125
            if avg_beat[j] < th_left:
                idx_qrs_start = j
                break
        for j in range(idx_r, idx_r + int(0.1 * srate)):
            if avg_beat[j] < th_right:
                idx_qrs_end = j
                break

        # find offset with maximum correlation
        offsets = []  # for each beat, likes [0, -1, 0, 0, 1, ...]
        qrs_coeffs = []
        offset_width = int(0.01 * srate)  # 3 = range for finding offset
        for i in range(segbeats):  # for each beat
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

        # move beats by the offset
        new_beats = []
        for i in range(segbeats):
            ost = offsets[i]
            beat = beats[i].tolist()
            if ost < 0:
                beat = [0] * -ost + beat[:ost]
            else:
                beat = beat[ost:] + [0] * ost
            new_beats.append(beat)
        beats = np.array(new_beats)  # beats.shape == (segbeats,300)

        # calculate average beat
        avg_beat = np.mean(beats, axis=0)  # average beat of the segbeats beats

        # replace vpc as template
        nreplaced = 0
        for i in range(segbeats):
            ce = arr.corr(avg_beat, beats[i])
            if ce < 0.95:
                nreplaced += 1
                beats[i] = copy.deepcopy(avg_beat)
                offsets[i] = 0

        #print('{} beats are replaced'.format(nreplaced))
        if nreplaced > 0.1 * segbeats:
            print('excluded VPC > {}'.format(nreplaced))
            continue

        # qrs level alignment
        # idx_r == 125
        # len(avg_beat) == beat_len == 300
        for i in range(segbeats):
            beats[i] -= beats[i][idx_r]

        # plot for debugging
        # plt.plot()
        # for i in range(segbeats):
        #     col = 'blue'
        #     if i % 2:
        #         col = 'red'
        #     plt.plot(beats[i], c=col, ls='-')
        # plt.savefig('{:02d}_{}.png'.format(opt['ifile'], iseg))
        # plt.close()

        # gather segbeats beats from idx_r(125) to beat_len(300)
        # power spectrums of segbeats beats
        spect = []
        for idx_from_r in range(beat_len - idx_r):
            timed_samples = beats[:, idx_r + idx_from_r]
            # timed_samples *= np.hamming(len(timed_samples))
            segfft = 2 ** np.abs(np.fft.fft(timed_samples))
            spect.append(segfft)  # each segbeats beat fft result
        spect = np.array(spect)  # rows == idx_from_r, cols == frequency(0-segbeats)

        # power spectra are summed into a composite in which
        # the magnitude at 0.5 cycles/beat indicates raw alternans (in mv2)
        # cum_spect.shape == segbeats

        # cumulative spectum of beats
        st_start = int(0.1 * srate)  # idx_qrs_end  - idx_r  #int(0.1*srate)
        st_end = int(0.25 * srate)
        avg_spect = np.mean(spect[st_start:st_end, :], axis=0)  # between 100 (50) and 250 ms (125) from rpeak
        avg_alt = avg_spect[int(0.5 * segbeats)]

        # cum_spect_noise = cum_spect[int(0.4*segbeats):int(0.46*segbeats)]  # noise level: 0.44-0.49 cycles / beat
        avg_spect_noise = avg_spect[int(0.44 * segbeats):int(0.49 * segbeats)]  # noise level: 0.44-0.49 cycles / beat
        # cum_spect_noise = cum_spect[int(0.33 * segbeats):int(0.48 * segbeats)]  # noise level: 0.44-0.49 cycles / beat
        avg_noise_avg = np.mean(avg_spect_noise)
        avg_noise_std = np.std(avg_spect_noise)

        # return avg beat
        # avg_beat = np.mean(beats, axis=0)
        for j in range(len(avg_beat)):
            if len(ret_avg_beat['vals']) > r_list[seg_start + segbeats - 1] + j:
                ret_avg_beat['vals'][r_list[seg_start + segbeats - 1] + j] = avg_beat[j]

        # print('avg alt {}, noise {}'.format(cum_alt, cum_noise_avg))
        twar = 0
        if avg_alt > avg_noise_avg:
            twav = 1000 * (avg_alt - avg_noise_avg) ** 0.5
            twar = (avg_alt - avg_noise_avg) / avg_noise_std
            twavs.append(twav)
            ret_twav.append({'dt': r_list[seg_start + segbeats - 1] / srate, 'val': twav})

        twars.append(twar)
        ret_twar.append({'dt': r_list[seg_start + segbeats - 1] / srate, 'val': twar})

        # plt.figure(figsize=(30, 5))
        # plt.plot(ecg_filt.tolist(), color='black', lw=1)
        # plt.savefig('e:/{}_raw.pdf'.format(twar), bbox_inches="tight", pad_inches=0.5)
        # plt.close()
        #
        # plt.figure(figsize=(10, 5))
        # for i in range(len(beats)):
        #     c = 'red'
        #     if i % 2 == 0:
        #         c = 'blue'
        #     plt.plot(beats[i], color=c, lw=1)
        # plt.savefig('e:/{}_ecg.pdf'.format(twar), bbox_inches="tight", pad_inches=0.5)
        # plt.close()
        #
        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(1, 65) / 128, avg_spect[1:65], lw=1)
        # plt.savefig('e:/{}_spect.pdf'.format(twar), bbox_inches="tight", pad_inches=0.5)
        # plt.close()

    dt_last = r_list[-1] / srate - cfg['overlap']

    return [
        {'srate': srate, 'vals': ecg_filt.tolist()},
        ret_avg_beat,
        ret_rpeak,
        ret_twav,
        ret_twar
    ]
