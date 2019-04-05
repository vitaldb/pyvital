import numpy as np

cfg = {
    'name': 'NIRS - Cerebral Oximeter Index',
    'group': 'Medical algorithms',
    'desc': 'Calculate Pearson correlation coefficient between blood pressure and cerebral oxymetry',
    'reference': 'Brady. et al. Continuous time-domain analysis of cerebrovascular autoregulation using near-infrared spectroscopy. Stroke. 2007 October; 38(10):2818-2825.',
    'interval': 300,
    'overlap': 0,
    'inputs': [
        {'name': 'ART1_MBP', 'type': 'num'},
        {'name': 'SCO2_R', 'type': 'num'}
        ],
    'outputs': [
        {'name': 'mbp', 'type':'num', 'unit':'mmHg'},
        {'name': 'cox_slope', 'type':'num', 'unit':'%/mmHg', 'min': -1, 'max': 1},
        {'name': 'cox_pearson', 'type':'num', 'min':-1, 'max':1},
        {'name': 'art_10sec', 'type':'num', 'unit':'mmHg', 'min': 0, 'max': 150},
        {'name': 'sco_10sec', 'type':'num', 'unit':'%', 'min':40, 'max':100}
        ]
}


def run(inp, opt, cfg):
    dt_last = cfg['interval']

    avg_interval = 10

    sco_vals_10avg = []  # 10 sec avg
    mbp_vals_10avg = []
    mbp_10sec = []  # trk
    sco_10sec = []  # trk
    for dt_bin_from in range(0, dt_last, avg_interval):
        dt_bin_to = min(dt_bin_from + avg_interval, dt_last)
        scos = []  # collect for 10 sec
        mbps = []
        for esco in inp['SCO2_R']:
            if 20 < esco['val']:
                if dt_bin_from < esco['dt'] < dt_bin_to:
                    scos.append(esco['val'])
        for embp in inp['ART1_MBP']:
            if 20 < embp['val'] < 200:
                if dt_bin_from < embp['dt'] < dt_bin_to:
                    mbps.append(embp['val'])
        if not mbps or not scos:
            continue
        sco_vals_10avg.append(np.mean(scos))
        mbp_vals_10avg.append(np.mean(mbps))
        mbp_10sec.append({'dt':dt_bin_to, 'val':np.mean(mbps)})
        sco_10sec.append({'dt':dt_bin_to, 'val':np.mean(scos)})

    if len(mbp_vals_10avg) == 0:
        return
#
#    if np.max(mbp_vals_10avg) - np.min(mbp_vals_10avg) < 20:
#        return

    mbp = np.mean(mbp_vals_10avg)
    r = np.corrcoef(mbp_vals_10avg, sco_vals_10avg)[0, 1]
    cox, _ = np.linalg.lstsq(np.vstack([mbp_vals_10avg, np.ones(len(mbp_vals_10avg))]).T, sco_vals_10avg)[0]

    return [
        [{'dt': dt_last, 'val': mbp}],
        [{'dt': dt_last, 'val': cox}],
        [{'dt': dt_last, 'val': r}],
        mbp_10sec,
        sco_10sec,
    ]
