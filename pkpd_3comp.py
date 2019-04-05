import numpy as np

last_a1 = 0
last_a2 = 0
last_a3 = 0
last_a4 = 0
last_vol = 0

cfg = {
    'name': 'PKPD - 3 Compartment Model',
    'group': 'Medical algorithms',
    'desc': '3 compartment PKPD model',
    'reference': '',
    'overlap': 0,
    'interval': 10,
    'inputs': [
        {'name': 'pump1_vol', 'type': 'num'},
        {'name': 'pump1_conc', 'type': 'num'}
    ],'options': [
        {'name': 'model', 'sels': 'Marsh/Modified Marsh/Schnider/Paedfusor/Kataria/Kim/Minto', 'init': 'Schnider'},
        {'name': 'age', 'init': 50},
        {'name': 'sex', 'sels': 'F/M'},
        {'name': 'ht', 'init': 160},
        {'name': 'wt', 'init': 63}
    ],
    'outputs': [
        {'name': 'cp', 'type': 'num', 'min': 0, 'max': 10},
        {'name': 'ce', 'type': 'num', 'min': 0, 'max': 10}
        ]
}


def get_model(name, age, sex, wt, ht):
    v1 = 0
    k10 = 0
    k12 = 0
    k13 = 0
    k21 = 0
    k31 = 0
    ke0 = 0
    if name == 'Marsh':
        v1 = 0.228 * wt
        k10 = 0.119
        k12 = 0.114
        k13 = 0.0419
        k21 = 0.055
        k31 = 0.0033
        ke0 = 0.26  # diprifusor
    elif name == "Modified Marsh":
        v1 = 0.228 * wt
        k10 = 0.119
        k12 = 0.114
        k13 = 0.0419
        k21 = 0.055
        k31 = 0.0033
        ke0 = 1.2195  # stanpump, orchestra
    elif name == "Schnider":
        lbm = james(sex, wt, ht)
        v1 = 4.27
        v2 = 18.9 - 0.391 * (age - 53)
        v3 = 238
        cl1 = 1.89 + 0.0456 * (wt - 77) - 0.0681 * (lbm - 59) + 0.0264 * (ht - 177)
        cl2 = 1.29 - 0.024 * (age - 53)
        cl3 = 0.836
        k10 = cl1 / v1
        k12 = cl2 / v1
        k13 = cl3 / v1
        k21 = cl2 / v2
        k31 = cl3 / v3
        ke0 = 0.456
    elif name == "Paedfusor":
        if 1 <= age < 13:
            v1 = 0.4584 * wt
            k10 = 0.1527 * wt ** -0.3
        elif age <= 13:
            v1 = 0.4 * wt
            k10 = 0.0678
        elif age <= 14:
            v1 = 0.342 * wt
            k10 = 0.0792
        elif age <= 15:
            v1 = 0.284 * wt
            k10 = 0.0954
        elif age <= 16:
            v1 = 0.22857 * wt
            k10 = 0.119
        else:
            v1 = None
        k12 = 0.114
        k13 = 0.0419
        k21 = 0.055
        k31 = 0.0033
        ke0 = 0.26  # from diprifusor (for adults)
        ke0 = 0.91  # Munoz et al Anesthesiology 2004:101(6)
    elif name == "Kataria":  # Kataria et al. Anesthesiology 1994;80:104
        v1 = 0.41 * wt
        v2 = 0.78 * wt + 3.1 * age - 15.5
        v3 = 6.9 * wt
        cl1 = 0.035 * wt
        cl2 = 0.077 * wt
        cl3 = 0.026 * wt
        k10 = cl1 / v1
        k12 = cl2 / v1
        k13 = cl3 / v1
        k21 = cl2 / v2
        k31 = cl3 / v3
        ke0 = 0.41  # Munoz et al Anesthesiology 2004:101(6)
    elif name == "Kim":
        v1 = 1.69
        v2 = 27.2 + 0.93 * (wt - 25)
        cl1 = 0.89 * (wt / 23.6) ** 0.97
        cl2 = 1.3
        k10 = cl1 / v1
        k12 = cl2 / v1
        k13 = 0
        k21 = cl2 / v2
        k31 = 0
    elif name == "Minto":
        lbm = james(sex, wt, ht)
        v1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        v2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        v3 = 5.42
        cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        cl2 = 2.05 - 0.0301 * (age - 40)
        cl3 = 0.076 - 0.00113 * (age - 40)
        k10 = cl1 / v1
        k12 = cl2 / v1
        k13 = cl3 / v1
        k21 = cl2 / v2
        k31 = cl3 / v3
        ke0 = 0.595 - 0.007 * (age - 40)
    return v1, k10, k12, k13, k21, k31, ke0


def james(sex, wt, ht):
    if sex == "M":
        return 1.1 * wt - 128 * (wt / ht) ** 2
    elif sex == "F":
        return 1.07 * wt - 148 * (wt / ht) ** 2
    return None


def run(inp, opt, cfg):
    global last_vol, last_a1, last_a2, last_a3, last_a4

    interval = cfg['interval']

    conc = 0
    for cone in inp['pump1_conc']:
        if cone['val'] > 0:
            conc = cone['val']
            break
    if conc == 0:
        print("conc = {}".format(conc))
        return

    v1, k10, k12, k13, k21, k31, k41 = get_model(opt['model'], opt['age'], opt['sex'], opt['wt'], opt['ht'])
    k10 /= 60
    k12 /= 60
    k13 /= 60
    k21 /= 60
    k31 /= 60
    k41 /= 60
    v4 = v1 / 1000
    k14 = k41 * v4 / v1

    # collect volumes every 1 sec
    vols = [last_vol] + [0] * interval
    for t in range(interval):
        for vole in inp['pump1_vol']:
            if t < vole['dt'] <= t+1:
                vols[t+1] = vole['val']

    # fill blanks
    for t in range(1, interval+1):
        if vols[t] == 0:
            vols[t] = vols[t-1]

    # cache last vols for next call
    last_vol = vols[-1]

    # convert volumes to doses
    doses = np.diff(vols) * conc

    # update amount of drug in each compartment
    cp_list = []
    ce_list = []
    for t in range(interval):
        next_a1 = last_a1 - last_a1 * (k10 + k12 + k13) + last_a2 * k21 + last_a3 * k31 + doses[t]
        next_a2 = last_a2 + last_a1 * k12 - last_a2 * k21
        next_a3 = last_a3 + last_a1 * k13 - last_a3 * k31
        next_a4 = last_a4 + last_a1 * k14 - last_a4 * k41

        last_a1 = next_a1
        last_a2 = next_a2
        last_a3 = next_a3
        last_a4 = next_a4

        cp_list.append({"dt": t + 1, "val": last_a1 / v1})
        ce_list.append({"dt": t + 1, "val": last_a4 / v4})

    return [
        cp_list,
        ce_list
    ]
