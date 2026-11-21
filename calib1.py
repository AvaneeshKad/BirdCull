from common import *
import sys; sys.path.insert(0,'.')
from view import montage
S=load(); by=folder_stats(S); A=approved_folders(by)
print(len(A),'approved folders;',sum(v[0] for v in A.values()),'photos')
for f,v in sorted(A.items()): print(v[0],f"{v[1]*100:.0f}%",f"{v[2]*100:.0f}%",f.split('Extreme SSD/')[1][-60:],'|',v[3])
print('NORMAL:')
for f,rs in sorted(by.items()):
    if f not in A: print(len(rs),f.split('Extreme SSD/')[1])
sc=[(best(r)['edge'],r) for f in A for r in by[f] if best(r) and best(r)['conf']>=0.4]
sc.sort(key=lambda x:x[0]); v=[x[0] for x in sc]
print('approved n',len(v),'pctiles 0,0.5,1,2,5,10,25,50',np.percentile(v,[0,.5,1,2,5,10,25,50]))
items=[(r['folder'],r['raw'],best(r)['box'],f"{e:.1f} w{best(r)['cw']} {r['stem']}") for e,r in sc[:24]]
montage(items,'/tmp/claude-1000/-home-avopti-Documents-bird-work/68724e2d-995c-4e8f-8216-e39807deecbd/scratchpad/low_approved.jpg',cols=6,size=330)
