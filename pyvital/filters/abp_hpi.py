import os
import sys
import pyvital.arr as arr
import torch
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding='valid')
        self.conv2 = nn.Conv1d(64, 64, 3, padding='valid')
        self.conv3 = nn.Conv1d(64, 64, 3, padding='valid')
        self.conv4 = nn.Conv1d(64, 64, 3, padding='valid')
        self.conv5 = nn.Conv1d(64, 64, 3, padding='valid')
        self.batchnorm = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.maxpool(self.batchnorm(torch.relu(self.conv1(x))))
        x = self.maxpool(self.batchnorm(torch.relu(self.conv2(x))))
        x = self.maxpool(self.batchnorm(torch.relu(self.conv3(x))))
        x = self.maxpool(self.batchnorm(torch.relu(self.conv4(x))))
        x = self.maxpool(self.batchnorm(torch.relu(self.conv5(x))))
        # print(x.shape)

        x = self.gap(x)
        x = torch.squeeze(x, 2)
        # print(x.shape)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

model = None

cfg = {
    'name': 'ABP - Hypotension Prediction Index',
    'group': 'Medical algorithms',
    'desc': 'Predict hypotension 5 minutes before the event from arterial blood pressure using deep learning',
    'reference': 'HPI_CNN',
    'overlap': 10,
    'interval': 20,
    'inputs': [{'name': 'ART', 'type': 'wav'}],
    'outputs': [{'name': 'HPI', 'type': 'num', 'min': 0, 'max': 100}]
}

def run(inp, opt, cfg):
    """
    Predict hypotension 5 minute before the event from abp
    :param inp: arterial blood pressure (input wave)
    input wave must be 1-dimensional (#,)
    :param opt:
    :param cfg:
    :return: HPI index score
    """
    global model

    trk_name = [k for k in inp][0]
    if 'srate' not in inp[trk_name]:
        return

    signal_data = np.array(inp[trk_name]['vals'])
    prop_nan = np.mean(np.isnan(signal_data))
    if prop_nan > 0.1:
        return # raise ValueError(inp)#'nan {}'.format(prop_nan))
    # print('abp_hpi: input is:', inp)
    # print('abp_hpi: input vals:', inp[trk_name]['vals'])

    signal_data = arr.interp_undefined(signal_data)

    srate = inp[trk_name]['srate']
    signal_data = arr.resample_hz(signal_data, srate, 100)
    srate = 100

    if len(signal_data) < 2000 * 0.9:
        return  # raise ValueError('len < 18 sec')

    if len(signal_data) != 2000:
        signal_data = signal_data[:2000]
        if len(signal_data) < 2000:
            signal_data = np.pad(signal_data, (0, 2000 - len(signal_data)), 'constant', constant_values=np.nan)

    signal_data = arr.interp_undefined(signal_data)

    if np.nanmax(signal_data) > 200:
        return 
    if np.nanmin(signal_data) < 20:
        return 
    if np.nanmax(signal_data) - np.nanmin(signal_data) < 30:
        return 
    if any(np.abs(np.diff(signal_data[~np.isnan(signal_data)])) > 30):
        return 

    signal_data = signal_data.reshape((-1, 1, 2000))

    # print('abp_hpi: signal data process done, shape:', signal_data.shape)

    # normalize signal_data
    signal_data -= 65
    signal_data /= 65

    signal_data_torch = torch.from_numpy(signal_data)

    if model is None:
        model = Net()
        model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/model_hpi_state_dict_v1.pth'))

    # print('abp_hpi: model input is: ', signal_data_torch)
    prediction = model(signal_data_torch.float())
    hpi = int(np.squeeze(prediction.detach().numpy()).tolist() * 100)
    # print('abp_hpi: model output is: ', hpi)

    return [
        [{'dt': cfg['interval'], 'val': hpi}]
    ]

if __name__ == '__main__':
    import vitaldb
    vf = vitaldb.VitalFile(1, 'ART')
    vf.run_filter(run, cfg)
    vf.to_vital(f'filtered.vital')
