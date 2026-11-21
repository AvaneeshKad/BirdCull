import cv2, numpy as np
def acuity(g, sigma=0.7):
    """g gray uint8 crop (native half-size). Returns dict: acu (p90 of edge-gradient / local-range over contrasty pixels), n (count of contrasty edge px)."""
    f = cv2.GaussianBlur(g.astype(np.float32), (0,0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = np.sqrt(gx*gx+gy*gy)
    k = np.ones((9,9),np.uint8)
    rng = cv2.dilate(f,k) - cv2.erode(f,k)
    # contrast floor relative to dynamic range of the crop (dark crops are low-contrast but still judgeable)
    lo,hi = np.percentile(f,[2,98]); floor = max(6.0, 0.15*(hi-lo))
    m = rng > floor
    # only local maxima of gradient along the edge: take pixels in top region of gradient
    if m.sum() < 30: return {"acu": None, "n": int(m.sum())}
    ratio = mag[m]/rng[m]
    # strong-edge pixels: gradient above 50% of the crop's 99.5 pct
    gt = np.percentile(mag[m], 70)
    sel = mag[m] >= gt
    return {"acu": float(np.percentile(ratio[sel], 90)), "n": int(m.sum())}
