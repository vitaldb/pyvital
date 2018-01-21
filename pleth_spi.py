import arr
import numpy as np


class Histogram:
    def __init__(self, minval = 0, maxval = 100, resolution = 1000):
        self.minval = minval
        self.maxval = maxval
        self.bins = [0] * resolution
        self.total = 0

    def getbin(self, v):
        if v < self.minval:
            return 0
        if v > self.maxval:
            return self.bins[-1]
        bin = int((v-self.minval) / (self.maxval - self.minval) * len(self.bins))
        if bin >= len(self.bins):
            return self.bins.length-1
        return bin

    def learn(self, v):
        """
        add tnew data
        """
        bin = self.getbin(v)
        self.bins[bin] += 1
        self.total += 1

    # minimum value -> 0, maximum value -> 1
    def percentile(self, v):
        if self.total == 0:
            return 0
        # number of values less than the value
        cnt = 0
        bin = self.getbin(v)
        for i in range(bin):
            cnt += self.bins[i]
        return cnt / self.total * 100


cfg = {
    'name': 'Pleth - Surgical Pleth Index',
    'group': 'Medical algorithms',
    'reference': 'Br J Anaesth. 2007 Apr98(4):447-55',
    'interval': 30, # for 4096 sample/call
    'overlap': 3, # 2 sec overlap for HR=30
    'inputs': [{'name': 'pleth', 'type': 'wav'}],
    'outputs': [{'name': 'beat', 'type': 'num', 'max':2},
        {'name': 'ppga', 'type': 'num', 'max':100},
        {'name': 'hbi', 'type': 'num', 'min':240, 'max':2000},
        {'name': 'spi', 'type': 'num', 'max':100}],

    # filter should be called sequentially
    'hist_ppga': Histogram(0, 100, 1000),
    'hist_hbi': Histogram(240, 2000, 1000)  # HR 30-250 --> HBI 240-2000
}


def run(inp, opt, cfg):
    """
    http:#ocw.utm.my/file.php/38/SEB4223/07_ECG_Analysis_1_-_QRS_Detection.ppt%20%5BCompatibility%20Mode%5D.pdf
    """
    data = arr.interp_undefined(inp['pleth']['vals'])
    srate = inp['pleth']['srate']

    minlist, maxlist = arr.detect_peaks(data, srate)  # extract beats
    beat_res = [{'dt':idx / srate, 'val':1} for idx in maxlist]

    ppga_res = []
    hbi_res = []
    spi_res = []
    for i in range(len(maxlist) - 1):
        dt = maxlist[i+1] / srate

        pp = data[maxlist[i+1]] - data[minlist[i]]
        ppga_res.append({'dt': dt, 'val': pp})

        hbi = (maxlist[i+1] - maxlist[i]) / srate * 1000
        hbi_res.append({'dt': dt, 'val': hbi})

        hist_hbi = cfg['hist_hbi']
        hist_ppga = cfg['hist_ppga']
        arrnorm_hbi = hist_hbi.percentile(hbi)
        arrnorm_ppga = hist_ppga.percentile(pp)
        spi = 100 - (0.7 * arrnorm_ppga + 0.3 * arrnorm_hbi)
        spi_res.append({'dt':dt, 'val':spi})
        hist_hbi.learn(hbi)
        hist_ppga.learn(pp)

    return [beat_res, ppga_res, hbi_res, spi_res]
