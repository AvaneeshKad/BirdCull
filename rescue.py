"""Second, more sensitive detection pass on shots where the first pass found no bird."""
import json, sys, time, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'.')
from scan import decode
from ultralytics import YOLO
import cv2
OUT=Path('scans/rescue.jsonl')
rows=[r for r in csv.DictReader(open('plan.csv')) if r['reason']=='no bird found']
done=set()
if OUT.exists():
    for l in open(OUT): d=json.loads(l); done.add((d['folder'],d['stem']))
S={}
for l in open('scans/scan3.jsonl'):
    r=json.loads(l); S[(r['folder'],r['stem'])]=r
todo=[r for r in rows if (r['folder'],r['stem']) not in done]
print(len(todo),'to do',flush=True)
m=YOLO('models/yolo26l.pt'); names=m.names
want=[k for k,v in names.items() if v in ('bird','person','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe')]
def prep(r):
    try:
        rgb=decode(Path(r['folder'])/S[(r['folder'],r['stem'])]['raw']); return r,rgb
    except Exception: return r,None
out=open(OUT,'a'); n=0; t=time.time()
with ThreadPoolExecutor(6) as ex:
    for r,rgb in ex.map(prep,todo):
        if rgb is None:
            out.write(json.dumps({'folder':r['folder'],'stem':r['stem'],'err':1})+'\n'); continue
        res=m.predict(rgb,imgsz=2560,conf=0.05,classes=want,device=0,verbose=False,half=True)[0]
        dets=[(names[int(c)],round(float(cf),3)) for c,cf in zip(res.boxes.cls.cpu().numpy(),res.boxes.conf.cpu().numpy())]
        out.write(json.dumps({'folder':r['folder'],'stem':r['stem'],'dets':dets})+'\n'); n+=1
        if n%100==0: out.flush(); print(n,f'{n/(time.time()-t):.1f}/s',flush=True)
print('done')
