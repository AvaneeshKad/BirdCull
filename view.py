"""montage(items, out, cols): items = [(folder, raw, box, label)] -> jpg contact sheet of bird crops (from real RAW, half-size)."""
import cv2, numpy as np, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from scan import decode
def tile(it, size=420):
    folder, raw, box, label = it
    try:
        rgb = decode(Path(folder)/raw)
    except Exception as e:
        return np.zeros((size, size, 3), np.uint8)
    H, W = rgb.shape[:2]
    if box:
        x1,y1,x2,y2 = box; cx,cy=(x1+x2)/2,(y1+y2)/2; half=max(x2-x1,y2-y1)*0.65+20
        X1,Y1,X2,Y2=int(max(0,cx-half)),int(max(0,cy-half)),int(min(W,cx+half)),int(min(H,cy+half))
        c = rgb[Y1:Y2,X1:X2]
    else: c = rgb
    s = size/max(c.shape[:2])
    c = cv2.resize(c,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_NEAREST)
    canvas = np.zeros((size+18,size,3),np.uint8); canvas[:c.shape[0],:c.shape[1]]=c
    cv2.putText(canvas,label[:60],(2,size+13),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,0),1)
    return canvas
def montage(items, out, cols=4, size=420):
    with ThreadPoolExecutor(8) as ex: tiles=list(ex.map(lambda i: tile(i,size), items))
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    rows=[np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
