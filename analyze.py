import json,sys,collections,numpy as np,re
recs=[json.loads(l) for l in open('scans/scan.jsonl')]
recs=[r for r in recs if r.get('v')==2]
by=collections.defaultdict(list)
for r in recs: by[r['folder']].append(r)
def bs(r):
    return max((b['edge'] for b in r.get('birds',[])),default=None)
rows=[]
for f,rs in sorted(by.items()):
    t=sorted([r for r in rs if r.get('time')],key=lambda r:r['time'])
    inb=0;run=1;maxrun=1
    for i,r in enumerate(t):
        prev=i>0 and r['time']-t[i-1]['time']<=1.0
        nxt=i+1<len(t) and t[i+1]['time']-r['time']<=1.0
        inb+= (prev or nxt)
        if prev: run+=1
        else: run=1
        maxrun=max(maxrun,run)
    sc=[bs(r) for r in rs if bs(r) is not None]
    nb=sum(1 for r in rs if r.get('birds'))
    pe=sum(1 for r in rs if r.get('persons'))
    xmp=sum(any(x.lower().endswith('.xmp') for x in r['files']) for r in rs)
    rows.append((f.replace('/media/avopti/Extreme SSD/',''),len(rs),inb/max(1,len(t)),maxrun,nb/len(rs),pe,xmp,
      np.percentile(sc,[5,25,50]) if sc else [0,0,0]))
for r in rows:
    print(f"{r[1]:5d} burst%={r[2]*100:3.0f} maxrun={r[3]:3d} bird%={r[4]*100:3.0f} pers={r[5]:3d} xmp={r[6]:3d} edge p5/25/50={r[7][0]:.1f}/{r[7][1]:.1f}/{r[7][2]:.1f} {r[0]}")
