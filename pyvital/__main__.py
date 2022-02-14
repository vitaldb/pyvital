import sys
import os.path
from sanic import Sanic
from sanic import response
import json
import importlib
import os
import traceback
import copy
import gzip
import time

sys.path.extend(['filters'])

cfgs = {}  # Current settings and data for the module (the corresponding invokeid)
default_cfgs = {}  # Default settings and data for the module
mods = {}  # Loaded modules
mod_cfgs = []  # load module cfgs

# load filters
for root, dirs, files in os.walk('filters'):
    for filename in files:
        #filepath = os.path.join(root, filename)
        if filename[-3:] != ".py":
            continue

        m_modname = filename[:-3]  #filepath[:-3].replace(os.sep, ".")
        print('importing ' + m_modname)
        o = importlib.import_module(m_modname)
        mods[m_modname] = o  # modules are saved for later reloading

        if not hasattr(o, 'cfg'):
            continue
        if not hasattr(o, 'run'):
            continue

        if m_modname not in default_cfgs:  # if the module was first loaded or changed?
            default_cfgs[m_modname] = copy.deepcopy(o.cfg)
        cfg = copy.deepcopy(default_cfgs[m_modname])

        if 'name' in cfg:
            name = cfg['name']
        else:
            name = m_modname
        if 'group' in cfg:
            group = cfg['group']
        else:
            group = ''
        if 'desc' in cfg:
            desc = cfg['desc']
        else:
            desc = ''
        if 'interval' in cfg:
            interval = cfg['interval']
        else:
            interval = 60
        if 'overlap' in cfg:
            overlap = cfg['overlap']
        else:
            overlap = 0
        if 'inputs' in cfg:
            inputs = cfg['inputs']
        else:
            inputs = []
        if 'options' in cfg:
            opt = cfg['options']
        else:
            opt = []
        if 'license' in cfg:
            licen = cfg['license']
        else:
            licen = ""
        if 'reference' in cfg:
            refer = cfg['reference']
        else:
            refer = ""
        if 'outputs' in cfg:
            outputs = cfg['outputs']
        else:
            outputs = []

        mod_cfgs.append({
            "modname": m_modname,
            "name": name,
            "group": group,
            "desc": desc,
            "interval": interval,
            "overlap": overlap,
            "inputs": inputs,
            "options": opt,
            "outputs": outputs,
            "license": licen,
            "reference": refer
        })

app = Sanic("filter_server")

@app.get("/")
async def list_filter(request):
    return response.json(mod_cfgs)

@app.post('/<modname>')
async def run_filter(request, modname):
    posts = gzip.decompress(request.body)
    posts = posts.decode('utf-8')

    #print('[' + posts + ']')

    try:
        posts = json.loads(posts)
    except Exception as e:
        print(e)
        return response.raw('')

    invokeid = posts['invokeid']
    inp = posts['inputs']
    m_modname = os.path.basename(modname)  # module name

    #o = importlib.import_module(m_modname)  # load or reload the module
    o = mods[m_modname]

    if invokeid not in cfgs.keys():  # whether this invokeid is a new one?
        if m_modname not in default_cfgs.keys():  # if the module is loaded at first or changed
            default_cfgs[m_modname] = copy.deepcopy(o.cfg)
        cfg = copy.deepcopy(default_cfgs[m_modname])
        cfgs[invokeid] = cfg
    else:  # reload the data of the invokeid
        cfg = cfgs[invokeid]

    cfg['interval'] = posts['interval']  # Interval, overlap must be user-specified values
    cfg['overlap'] = posts['overlap']
    cfg['invokeid'] = invokeid

    opt = []

    ret = o.run(inp, opt, cfg)  # evoke run function
    ret = json.dumps(ret, ignore_nan=True)  # print the result
    ret = gzip.compress(ret.encode('utf-8'))

    return response.raw(ret)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, access_log=False, workers=2)