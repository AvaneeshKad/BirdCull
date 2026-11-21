from common import *
from acu import acuity
import sys,random
from scan import decode
from pathlib import Path
from view import montage
S=load(); by=folder_stats(S); A=approved_folders(by)
random.seed(3)
N=[r for f in by if f not in A for r in by[f] if 'error' not in r and best(r) and best(r)['conf']>=0.4 and not r['persons']]
vals=[(best(r)['edge'],r) for r in N]
pick=[x for x in vals if 3<=x[0]<5]; random.shuffle(pick); pick=pick[:18]
items=[];
for e,r in pick:
    rgb=decode(Path(r['folder'])/r['raw']); import cv2
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); x1,y1,x2,y2=best(r)['box']
    a=acuity(g[y1:y2,x1:x2])
    items.append((r['folder'],r['raw'],best(r)['box'],f"acu {a['acu'] and round(a['acu'],3)} n{a['n']} {r['stem']}"))
montage(items,'/tmp/claude-1000/-home-avopti-Documents-bird-work/68724e2d-995c-4e8f-8216-e39807deecbd/scratchpad/acu1.jpg',cols=6,size=330)
