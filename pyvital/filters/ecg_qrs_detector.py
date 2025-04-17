import pyvital.arr as arr

cfg = {
    'name': 'ECG - QRS detector',
    'group': 'Medical algorithms',
    'desc': 'Simple QRS detector',
    'reference': 'http://ocw.utm.my/file.php/38/SEB4223/07_ECG_Analysis_1_-_QRS_Detection.ppt%20%5BCompatibility%20Mode%5D.pdf',
    'overlap': 3,  # 3 sec overlap for HR=20
    'interval': 40,
    'inputs': [{"name": 'ECG', "type": 'wav'}],
    'outputs': [{"name": 'RPEAK', "type": 'num', "min": 0, "max": 2}]
}


def run(inp, opt, cfg):
    trk_name = [k for k in inp][0]

    if 'srate' not in inp[trk_name]:
        return

    data = arr.interp_undefined(inp[trk_name]['vals'])
    srate = inp[trk_name]['srate']

    r_list = arr.detect_qrs(data, srate)  # detect r-peak
    ret_rpeak = []
    for idx in r_list:
        dt = idx / srate
        ret_rpeak.append({'dt': dt, 'val': 1})
    return [ret_rpeak]
