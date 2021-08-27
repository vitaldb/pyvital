import vitaldb
vals = vitaldb.load_case(caseid=1, tnames=['SNUADC/ECG_II'], interval=1/100)

import pyvital

ecg = vals[120000:121000, 0]
ecg = pyvital.exclude_undefined(ecg)
peaks = pyvital.detect_qrs(ecg, 100)

import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(ecg, color='g')
plt.plot(peaks, [ecg[i] for i in peaks], 'ro')
plt.show()
