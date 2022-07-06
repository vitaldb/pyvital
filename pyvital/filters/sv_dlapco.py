import os
import sys
import pyvital.arr as arr
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class InceptionModule(nn.Module):

    def __init__(self, in_channel, nfilter=32):
        super(InceptionModule, self).__init__()
        # implement same padding with (kernel_size//2) for pytorch
        # Ref: https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606

        # 1x1 conv path
        self.path0 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1X1 conv -> 3x3 conv path
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv path
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=5, padding=(5 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv path
        self.path3 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )


    def forward(self, x):
        print('x shape: {}'.format(x.shape))
        y0 = self.path0(x)
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)
        print('y0 shape: {}'.format(y0.shape))
        print('y1 shape: {}'.format(y1.shape))
        print('y2 shape: {}'.format(y2.shape))
        print('y3 shape: {}'.format(y3.shape))
        print('cat shape: {}'.format(torch.cat([y0, y1, y2, y3], 1).shape))

        return torch.cat([y0, y1, y2, y3], 1)


class InceptionModule_dilated(nn.Module):

    def __init__(self, in_channel, nfilter=32):
        super(InceptionModule_dilated, self).__init__()
        # implement same padding with (kernel_size//2) for pytorch
        # Ref: https://discuss.pytorch.org/t/convolution-1d-and-simple-function/11606

        # 1x1 conv path
        self.path0 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1X1 conv -> 3x3 conv path
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=3, padding=int(((3-1)*5)/2),
                      dilation=5),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv path
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True),
            nn.Conv1d(nfilter, nfilter, kernel_size=5, padding=int(((5-1)*7)/2),
                      dilation=7),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv path
        self.path3 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channel, nfilter, kernel_size=1),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(True)
        )

        # Dilation output size calculation
        # o = output
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation
        # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1

        # padding = ((s-1)*i + (k-1)*d)/2

    def forward(self, x):
        y0 = self.path0(x)
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)

        # print(y0.shape)
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        return torch.cat([y0, y1, y2, y3], 1)


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
       revised to 1d
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale

        super(SpatialNL, self).__init__()
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, d = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, d)

        x = self.z(x)
        x = self.bn(x) + residual

        return x

class Inception1DNet(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(True)
        )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        while (i < nlayer):
            dynamicInception[str(j)] = InceptionModule(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_aswh = self.fnn(x[0])

        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)
        concat = torch.cat([out_aswh, out_cnn], 1)

        out = self.regressor(concat)

        return out


class Inception1DNet_NL_compact_dilated(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet_NL_compact_dilated, self).__init__()

        self.fnn = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(True)
        )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter * 4),
            nn.Dropout(0.5)
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        k = 0
        while (i < nlayer):
            if i > (nlayer - 3):
                dynamicInception[str(j)] = SpatialNL(nfilter * 4, nfilter * 4)
                j += 1
            dynamicInception[str(j)] = InceptionModule_dilated(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Linear((nfilter * 4) + 4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_aswh = self.fnn(x[0])

        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)
        concat = torch.cat([out_aswh, out_cnn], 1)

        out = self.regressor(concat)

        return out


class Inception1DNet_NL_compact_dilated_no_ashw(nn.Module):

    def __init__(self, nlayer, nfilter=32):
        super(Inception1DNet_NL_compact_dilated_no_ashw, self).__init__()

        # remove ashw
        # self.fnn = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.ReLU(True)
        # )

        self.stem = nn.Sequential(
            nn.Conv1d(1, nfilter, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter),
            nn.Conv1d(nfilter, nfilter * 4, kernel_size=3, padding=(3 // 2)),
            nn.BatchNorm1d(nfilter * 4),
            nn.Dropout(0.5)
        )

        dynamicInception = OrderedDict()
        i = 0
        j = 0
        k = 0
        while (i < nlayer):
            if i > (nlayer - 3):
                dynamicInception[str(j)] = SpatialNL(nfilter * 4, nfilter * 4)
                j += 1
            dynamicInception[str(j)] = InceptionModule_dilated(nfilter * 4, nfilter)
            j += 1
            if i % 2 == 0:
                dynamicInception[str(j)] = nn.AvgPool1d(2)
                j += 1
            i += 1

        self.Inception = nn.Sequential(dynamicInception)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            # for noashw
            nn.Linear(nfilter*4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x must be shape of 2dim list,
        1st is : aswh
        2nd is : nsec signal
        """
        out_cnn = self.stem(x[1])
        out_cnn = self.Inception(out_cnn)
        out_cnn = self.GAP(out_cnn)
        out_cnn = torch.squeeze(out_cnn, 2)

        out = self.regressor(out_cnn)

        return out

model = None

cfg = {
    'name': 'Stroke Volume',
    'group': 'Medical algorithms',
    'desc': 'Calculate stroke volume from arterial blood pressure using deep learning',
    'reference': 'DLAPCO',
    'overlap': 10,
    'interval': 20,
    'inputs': [{'name': 'ART', 'type': 'wav'}],
    'outputs': [
        {'name': 'SV', 'type': 'num', 'min': 0, 'max': 200, 'unit': 'mL'},
        {'name': 'CO', 'type': 'num', 'min': 0, 'max': 10, 'unit': 'L'},
    ]
}

def run(inp, opt, cfg):
    """
    calculate SV from DeepLearningAPCO
    :param inp: input waves
    inp['ART']['vals'] must be 1-dimensional, (#,)
    :param opt: demographic information
    :param cfg:
    :return: SV
    """
    global model
    trk_name = [k for k in inp][0]
    
    if 'srate' not in inp[trk_name]:
        return
    
    signal_data = arr.interp_undefined(inp[trk_name]['vals'])
    srate = inp[trk_name]['srate']

    signal_data = arr.resample_hz(signal_data, srate, 100)
    srate = 100

    # Whole heart freq estimation
    hr = arr.estimate_heart_freq(signal_data, srate) * 60

    signal_data = np.array(signal_data) / 100.

    if len(np.squeeze(signal_data)) < 20 * srate:
        return

    #age_data = opt['age']
    #sex_data = opt['sex']
    #wt_data = opt['weight']
    #ht_data = opt['height']
    age_data = 60
    sex_data = 1.
    wt_data = 65.8
    ht_data = 164.9

    if all (k in opt for k in ('age','sex','weight','height')) :
        age_data = int(opt['age'])
        sex_data = int(opt['sex'])
        wt_data = int(opt['weight'])
        ht_data = int(opt['height'])
    
    #print(age_data, sex_data, wt_data, ht_data)

    if isinstance(sex_data, str):
        if sex_data == 'M':
            sex_data = int(1)
        elif sex_data == 'F':
            sex_data = int(0)
        else:
            raise ValueError('opt_sex must be "M" or "F". current value: {}'.format(sex_data))

    else:
        if not ((int(sex_data) == 1) or (int(sex_data) == 0)):
            raise ValueError('opt_sex must be 1 or 0 current value: {}'.format(str(sex_data)))

    ashw_data = np.array([age_data, sex_data, wt_data, ht_data])

    x_input = [torch.Tensor(np.expand_dims(ashw_data, axis=0)), torch.Tensor(np.expand_dims(signal_data, axis=(0, 1)))]

    if model is None:
        model = Inception1DNet_NL_compact_dilated(nlayer=15, nfilter=32)
        model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/model_dlapco_v1.pth'))
        model = model.cpu()
        model.eval()

    output = model(x_input)
    sv = float(output.detach().numpy())

    return [
        [{'dt': cfg['interval'], 'val': sv}],
        [{'dt': cfg['interval'], 'val': sv * hr / 1000}],
    ]

if __name__ == '__main__':
    import vitaldb
    vf = vitaldb.VitalFile(1, 'ART')
    vf.run_filter(run, cfg)
    vf.to_vital(f'filtered.vital')