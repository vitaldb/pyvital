import math
import numpy as np

cfg = {
    'name': 'NIRS - Cerebral Oximeter Index',
    'group': 'Medical algorithms',
    'desc': 'Calculate Pearson correlation coefficient between blood pressure and cerebral oxymetry',
    'reference': 'Brady. et al. Continuous time-domain analysis of cerebrovascular autoregulation using near-infrared spectroscopy. Stroke. 2007 October; 38(10):2818-2825.',
    'overlap': 60*4,  # for HR=40
    'interval': 60*5,  # 5 min
    'inputs': [
        {'name': 'mbp', 'type': 'num'},
        {'name': 'sco', 'type': 'num'}
        ],
    'outputs': [
        {'name': 'mbp', 'type':'num'},
        {'name': 'cox', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_45', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_50', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_55', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_60', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_65', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_70', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_75', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_80', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_85', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_90', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_95', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_100', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_110', 'type':'num', 'min':-0.1, 'max':1},
        {'name': 'avg_115', 'type':'num', 'min':-0.1, 'max':1}
        ],

    # bins result average
    'bins': {}
}


def run(inp, opt, cfg):
    dt_last = cfg['interval']

    mbp_vals = [] # 10 sec average
    sco_vals = []
    for dt in range(0,dt_last,10):
        mbp_10sec_vals = []
        for e in inp['mbp']:
            if 20 < e['val'] < 200 and dt < e['dt'] < dt + 10:
                mbp_10sec_vals.append(e['val'])

        sco_10sec_vals = []
        for e in inp['sco']:
            if 20 < e['val'] and dt < e['dt'] < dt + 10:
                sco_10sec_vals.append(e['val'])

        if len(mbp_10sec_vals) == 0 or len(sco_10sec_vals) == 0:
            continue
        
        mbp_vals.append(np.mean(mbp_10sec_vals))
        sco_vals.append(np.mean(sco_10sec_vals))

    if len(mbp_vals) == 0 or len(sco_vals) == 0:
        return [] * len(inp)
    
    mbp = math.floor(np.mean(mbp_vals) / 5) * 5
    #cox = np.corrcoef(mbp_vals, sco_vals)[0, 1]
    cox, _ = np.linalg.lstsq(np.vstack([mbp_vals, np.ones(len(mbp_vals))]).T, sco_vals)[0]

    ret = [
        [{'dt': dt_last, 'val': mbp}],
        [{'dt': dt_last, 'val': cox}]
    ]

    if not math.isnan(cox):
        bins = cfg['bins']  # saving results

        if len(bins) == 0:  
            for i in range(45,120,5):
                bins[i] = []

        if 45 <= mbp < 120:  # save this result
            bins[mbp].append(cox)

        for i in range(45,120,5):
            if len(bins[i]) == 0:
                ret.append([])
            else:
                ret.append([{'dt': dt_last, 'val': np.median(bins[i][-30:])}])

    return ret
