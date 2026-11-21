from common import *
import random,sys
from view import montage
S=load(); by=folder_stats(S); A=approved_folders(by)
print(len(A),sum(len(by[f]) for f in A))
N=[r for f in by if f not in A for r in by[f] if 'error' not in r]
vals=[(best(r)['edge'],r) for r in N if best(r) and best(r)['conf']>=0.4 and not r['persons']]
v=np.array([x[0] for x in vals]); print(len(N),len(v),'no bird',sum(1 for r in N if not r.get('birds')),'persons',sum(1 for r in N if r['persons']))
for t in (2,3,3.8,5,7,9): print(t,(v<t).sum(),f"{(v<t).mean()*100:.1f}%")
random.seed(1)
for lo,hi,name in ((0,3,'lt3'),(3,5,'3to5')):
    pick=[x for x in vals if lo<=x[0]<hi]; random.shuffle(pick)
    items=[(r['folder'],r['raw'],best(r)['box'],f"{e:.1f} w{best(r)['cw']} c{best(r)['conf']:.2f} {r['stem']}") for e,r in pick[:24]]
    montage(items,f'/tmp/claude-1000/-home-avopti-Documents-bird-work/68724e2d-995c-4e8f-8216-e39807deecbd/scratchpad/n_{name}.jpg',cols=6,size=330)
