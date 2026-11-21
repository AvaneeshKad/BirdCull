import json,collections,re,os
import numpy as np
def load():
    d={}
    for l in open('/home/avopti/Documents/bird-work/scans/scan3.jsonl'):
        r=json.loads(l)
        if r.get('v')==3: d[(r['folder'],r['stem'])]=r
    return d
def best(r):
    b=r.get('birds') or []
    return max(b,key=lambda x:(x.get('acu') or 0)*min(1,x['conf']*1.5)) if b else None
def burst_pct(rs):
    t=sorted(r['time'] for r in rs if r.get('time'))
    if not t: return 0
    c=0
    for i,x in enumerate(t):
        if (i>0 and x-t[i-1]<=1.0) or (i+1<len(t) and t[i+1]-x<=1.0): c+=1
    return c/len(t)
def folder_stats(scan):
    by=collections.defaultdict(list)
    for r in scan.values(): by[r['folder']].append(r)
    return by
def approved_folders(by):
    out={}
    for f,rs in by.items():
        n=len(rs); bp=burst_pct(rs)
        ed=sum(any(x.lower().endswith(('.xmp','.dng')) for x in r['files']) for r in rs)/n
        why=[]
        if n<=200: why.append(f'small ({n} photos)')
        if n<=260 and bp<0.6: why.append(f'{n} photos, only {bp*100:.0f}% in bursts')
        if ed>=0.10: why.append(f'{ed*100:.0f}% have .xmp/.dng edits')
        rawdump=re.search(r'\d{3}N[A-Z]\w*\d*$',f) and bp>=0.4 and ed<0.10
        if why and not rawdump: out[f]=(n,bp,ed,'; '.join(why))
    return out
