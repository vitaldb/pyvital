import os
import vitaldb
import sys
import filters.ecg_hrv as f

print(f.cfg)
assert f.cfg['interval'] > f.cfg['overlap']

# read files
filenames = ["1.vital"]

# idir = 'rootFolderPath'
# for root, dirs, files in os.walk(idir):
#     for filename in files:
#         filenames.append(os.path.join(root, filename))

track_names = ['SNUADC/ECG_II']
srate = 500

for path in filenames:
    # read file
    vf = vitaldb.VitalFile(path, track_names)
    vals = vf.to_numpy(track_names, 1/srate).flatten()

    # run filter
    import numpy as np
    output_recs = []
    for output in f.cfg['outputs']:
        output_recs.append([])
    for dtstart_seg in np.arange(vf.dtstart, vf.dtend, f.cfg['interval'] - f.cfg['overlap']):
        dtend_seg = dtstart_seg + f.cfg['interval']
        idx_dtstart = int((dtstart_seg - vf.dtstart) * srate)
        idx_dtend = int((dtend_seg - vf.dtstart) * srate)
        try:
            outputs = f.run({f.cfg['inputs'][0]['name']: {'srate':srate, 'vals': vals[idx_dtstart:idx_dtend]}}, {}, f.cfg)
        except Exception as e:
            print(e)
            continue
        if outputs is None:
            continue
        for i in range(len(f.cfg['outputs'])):
            output = outputs[i]
            for rec in output:  # convert relative time to absolute time
                rec['dt'] += dtstart_seg
                output_recs[i].append(rec)

    # save to vital file
    for i in range(len(f.cfg['outputs'])):
        vf.add_track(f.cfg['outputs'][i]['name'], output_recs[i])

    vf.to_vital('filtered_' + os.path.basename(path))
