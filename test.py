import vitaldb
vals = vitaldb.load_case(caseid=1, tnames=['SNUADC/ART'], interval=1/100)

import pyvital.filters.abp_ppv as f
print(f.cfg)

