import json, struct, urllib.request, os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

BASE="https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/"
import urllib.request as _u
if not os.path.exists('index.json'):
    open('index.json','wb').write(_u.urlopen(BASE+'model.safetensors.index.json').read())
idx=json.load(open('index.json'))['weight_map']

def shard_header(shard):
    fn=f"hdr-{shard}.json"
    if os.path.exists(fn):
        return json.load(open(fn))
    req=urllib.request.Request(BASE+shard, headers={"Range":"bytes=0-7"})
    n=struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
    req=urllib.request.Request(BASE+shard, headers={"Range":f"bytes=8-{7+n}"})
    h=json.loads(urllib.request.urlopen(req).read())
    h["__data_start__"]=8+n
    json.dump(h,open(fn,"w"))
    return h

def fetch(name, byte_range=None):
    shard=idx[name]
    h=shard_header(shard)
    info=h[name]
    s0=h["__data_start__"]+info["data_offsets"][0]
    s1=h["__data_start__"]+info["data_offsets"][1]-1
    if byte_range:
        s0b=s0+byte_range[0]; s1b=s0+byte_range[1]-1
    else:
        s0b,s1b=s0,s1
    req=urllib.request.Request(BASE+shard, headers={"Range":f"bytes={s0b}-{s1b}"})
    return urllib.request.urlopen(req).read(), info

names=[n for n in idx if n.startswith('model.language_model.layers.0.linear_attn') or n=='model.language_model.layers.0.input_layernorm.weight']
os.makedirs('t',exist_ok=True)
for n in names:
    fn='t/'+n.split('layers.0.')[-1].replace('/','_')
    if os.path.exists(fn):
        print("have", fn); continue
    data,info=fetch(n)
    open(fn,'wb').write(data)
    print("got", n, info["dtype"], info["shape"], len(data))

# embedding rows for the prompt ids
ids=[248045,8678,198,24342,286,4879,369,716,310,830,11553,13,5044,1683,15060,1472,279,3274,11,9307,1328,30800,11,2814,47675,25605,11,321,60445,55404,11,27224,11,321,30246,303,279,1534,4087,13,248046,198,248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,248068,198]
H=5120
rows=b''
for t in ids:
    data,_=fetch('model.language_model.embed_tokens.weight', (t*H*2,(t+1)*H*2))
    rows+=data
open('t/embed_rows.bin','wb').write(rows)
json.dump(ids, open('t/ids.json','w'))
print("embed rows:", len(rows))
