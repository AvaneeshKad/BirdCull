from common import *
import random
from view import montage
S=load(); by=folder_stats(S); A=approved_folders(by)
ap=[best(r)['acu'] for f in A for r in by[f] if best(r) and best(r).get('acu') is not None and best(r)['conf']>=0.4]
for t in (.14,.15,.16,.17,.18,.19,.20): print(t,sum(1 for x in ap if x<t))
N=[r for f in by if f not in A for r in by[f] if 'error' not in r and best(r) and best(r).get('acu') is not None and best(r)['conf']>=0.4 and not r['persons']]
random.seed(5)
D='/tmp/claude-1000/-home-avopti-Documents-bird-work/68724e2d-995c-4e8f-8216-e39807deecbd/scratchpad/'
for lo,hi in ((0,.13),(.13,.16),(.16,.18),(.18,.20)):
    p=[r for r in N if lo<=best(r)['acu']<hi]; random.shuffle(p)
    print(lo,hi,len(p))
    montage([(r['folder'],r['raw'],best(r)['box'],f"{best(r)['acu']:.3f} w{best(r)['cw']} n{best(r)['acun']} {r['stem']}") for r in p[:18]],D+f'acu_{lo}.jpg',cols=6,size=330)
