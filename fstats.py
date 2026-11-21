import os,re,sys
sys.path.insert(0,'.')
from scan import find_shots
import collections
roots=["/media/avopti/Extreme SSD/Ashwini_Photos","/media/avopti/Extreme SSD/Ashwini'sVASPLaptopJul2025"]
by=collections.defaultdict(list)
for s in find_shots(roots): by[s['folder']].append(s)
for f,ss in sorted(by.items()):
    n=len(ss)
    xmp=sum(any(x.lower().endswith('.xmp') for x in s['files']) for s in ss)
    dng=sum(s['raw'].lower().endswith('.dng') for s in ss)
    jpg=sum(any(x.lower().endswith(('.jpg','.jpeg')) for x in s['files']) for s in ss)
    nums=[int(m.group(1)) for s in ss for m in [re.search(r'(\d{3,5})',s['stem'])] if m and not s['raw'].lower().endswith('.dng')]
    dens=n/(max(nums)-min(nums)+1) if nums else 0
    print(f"{n:5d} xmp={xmp:4d} dng={dng:3d} jpg={jpg:3d} dens={dens:.2f} {f.replace('/media/avopti/Extreme SSD/','')}")
