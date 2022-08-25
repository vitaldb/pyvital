import os
import pyvital.arr as arr
import numpy as np
import math
import keras.models

cfg = {
    'name': 'ECG - AI classifier',
    'group': 'Medical algorithms',
    'desc': '',
    'reference': '',
    'overlap': 5,
    'interval': 15,
    'inputs': [{'name': 'ECG', 'type': 'wav'}],
    'outputs': [
        {'name': 'RHYTHM', 'type': 'str'}, 
        {'name': 'BEAT', 'type': 'str'}, 
        {'name': 'RTYPE', 'type': 'num', 'min':-1, 'max':4},
        {'name': 'BTYPE', 'type': 'num', 'min':-1, 'max':3},
        ]
}

model_beat = None
model_rhythm = None
def run(inp, opt, cfg):
    global model_beat, model_rhythm

    trk_name = [k for k in inp][0]

    if 'srate' not in inp[trk_name]:
        return

    data = arr.interp_undefined(inp[trk_name]['vals'])
    srate = inp[trk_name]['srate']

    if model_beat is None:
        model_beat = keras.models.load_model(f'{os.path.dirname(__file__)}/model_beat.h5')

    if model_rhythm is None:
        model_rhythm = keras.models.load_model(f'{os.path.dirname(__file__)}/model_rhythm.h5')

    # resample
    data = arr.resample_hz(data, srate, 100)
    srate = 100

    # detect r-peaks
    peaks = np.array(arr.detect_qrs(data, srate), dtype=int)
    valid_mask = (srate <= peaks) & (peaks < len(data) - srate)
    peaks = peaks[valid_mask]  # remove qrs before overlap
    if len(peaks) == 0:
        return

    # output tracks
    out_bstr = []
    out_bnum = []
    out_rstr = []
    out_rnum = []

    # collect beat samples
    x = []
    for peak in peaks:
        seg = data[peak - srate:peak + srate]
        if max(seg) - min(seg) > 0:
            x.append(seg)
    
    if len(x) > 0:
        x = np.array(x, dtype=np.float32)

        # min-max normalization
        x -= x.min(axis=1)[...,None]
        x /= x.max(axis=1)[...,None]
        x = x[..., None]  # add dimension for cnn

        # predict
        y = np.argmax(model_beat.predict(x), axis=1)
        
        # beat label
        for i in range(len(y)):
            if y[i] == 0:
                s = 'N'
            elif y[i] == 1:
                s = 'S'
            elif y[i] == 2:
                s = 'V'
            else:
                continue
            out_bstr.append({'dt': peaks[i] / srate, 'val': s})
            out_bnum.append({'dt': peaks[i] / srate, 'val': y[i]})

    # rhythm label
    if len(peaks) >= 3:
        x = []
        seglen = 10 * srate
        if len(data) >= seglen:
            for i in range(0, len(data) - seglen, seglen):
                seg = data[i:i+seglen]
                if max(seg) - min(seg) > 0:
                    x.append(seg)
            if len(x) > 0:
                x = np.array(x, dtype=np.float32)
                x = x[x.min(axis=1) < x.max(axis=1)]

                # min-max normalization
                x -= x.min(axis=1)[...,None]
                # x /= x.max(axis=1)[...,None]
                x = x[..., None]  # add dimension for cnn

                # prediction
                y = np.argmax(model_rhythm.predict(x), axis=1)
                for i in range(len(y)):
                    if y[i] == 0:
                        s = 'SR'
                    elif y[i] == 1:
                        s = 'AF'
                    elif y[i] == 2:
                        s = 'Others'
                    elif y[i] == 3:
                        s = 'Noise'
                    else:
                        continue
                    out_rstr.append({'dt': i * seglen / srate, 'val': s})
                    out_rnum.append({'dt': i * seglen / srate, 'val': y[i]})

    return [out_rstr, out_bstr, out_rnum, out_bnum]


if __name__ == '__main__':
    import vitaldb
    for caseid in (2432, ): #2693, 603, 3323, 4636, 1204, 1738, 1776, 1901, 1926):
        print(f'{caseid}', end='...', flush=True)
        vf = vitaldb.VitalFile(caseid, 'ECG_II')
        print(f'filtering', end='...', flush=True)
        vf.run_filter(run, cfg)
        print(f'saving', end='...', flush=True)
        vf.to_vital(f'filtered_{caseid}.vital')
        print(f'done')