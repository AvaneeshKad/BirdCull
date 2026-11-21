#!/usr/bin/env python3
"""Stage A: scan RAW shots -> scans/scan.jsonl (resumable, append-only).

Decodes the REAL RAW data (rawpy half-size, never the embedded preview), runs YOLO
(bird + person), measures sharpness on each bird at native half-size resolution.
"""
import argparse, collections, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from acu import acuity

HERE = Path(__file__).resolve().parent
RAW_EXTS = {".nef", ".nrw", ".arw", ".dng", ".cr2", ".cr3", ".raf", ".orf", ".rw2"}
IMG_EXTS = {".jpg", ".jpeg"}
SIDE_EXTS = {".xmp"}
SKIP_DIRS = {"_to_delete", "_culled", "blurry", "duplicates", ".comments"}
SCAN = HERE / "scans" / "scan3.jsonl"
DETECT_SIDE = 1600
MAX_BIRDS = 6
VERSION = 3


def find_shots(roots):
    """list of dict(folder, stem, files[], raw) for every stem with a RAW member."""
    out = []
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.startswith("."))
            shots = {}
            for name in fn:
                stem, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext not in RAW_EXTS | IMG_EXTS | SIDE_EXTS:
                    continue
                if ext in SIDE_EXTS and os.path.splitext(stem)[1].lower() in RAW_EXTS | IMG_EXTS:
                    stem = os.path.splitext(stem)[0]
                shots.setdefault(stem.lower(), []).append(name)
            for stem, files in shots.items():
                raws = [f for f in files if os.path.splitext(f)[1].lower() in RAW_EXTS]
                if not raws:
                    continue
                # prefer NEF/ARW over DNG as the pixel source
                raws.sort(key=lambda f: (os.path.splitext(f)[1].lower() == ".dng", f))
                out.append({"folder": dp, "stem": stem, "files": sorted(files), "raw": raws[0]})
    return out


def decode(path):
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
                              output_bps=8)
    return rgb  # RGB uint8 half-size


def exif_info(path):
    import exifread
    info = {"time": None, "gps": None, "model": None}
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        v = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if v:
            t = datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
            sub = tags.get("EXIF SubSecTimeOriginal")
            if sub and str(sub).strip().isdigit():
                s = str(sub).strip()
                t += int(s) / (10 ** len(s))
            info["time"] = t
        info["model"] = str(tags.get("Image Model", "")).strip() or None
        la, lo = tags.get("GPS GPSLatitude"), tags.get("GPS GPSLongitude")
        if la and lo:
            def dd(x, ref):
                v = [float(a.num) / float(a.den) if a.den else 0 for a in x.values]
                d = v[0] + v[1] / 60 + v[2] / 3600
                return -d if str(ref) in ("S", "W") else d
            info["gps"] = [round(dd(la, tags.get("GPS GPSLatitudeRef")), 5),
                           round(dd(lo, tags.get("GPS GPSLongitudeRef")), 5)]
    except Exception:
        pass
    return info


def prepare(shot):
    p = Path(shot["folder"]) / shot["raw"]
    try:
        rgb = decode(p)
    except Exception as e:
        return shot, None, {"err": f"decode: {type(e).__name__}"}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    s = min(1.0, DETECT_SIDE / max(H, W))
    small = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    return shot, {"small": small, "gray": gray, "scale": s, "W": W, "H": H}, exif_info(p)


def edge_metrics(g):
    """Sharpness of a gray crop at native resolution. Returns dict."""
    h, w = g.shape
    if h < 12 or w < 12:
        return None
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.8)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = np.percentile(f, [5, 95])
    contrast = max(hi - lo, 12.0)
    g99 = float(np.percentile(mag, 99))
    g90 = float(np.percentile(mag, 90))
    lap = cv2.Laplacian(f, cv2.CV_32F)
    n = max(1, min(4, min(h, w) // 24))
    best = 0.0
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            gv = f[sl].var()
            if gv > 20:
                best = max(best, float(lap[sl].var() / (gv + 10)))
    ac = acuity(g)
    return {"acu": None if ac["acu"] is None else round(ac["acu"], 4), "acun": ac["n"],
            "edge": round(g99 / contrast * 100, 2), "g90": round(g90 / contrast * 100, 2),
            "lapn": round(best * 100, 2), "contrast": round(float(contrast), 1),
            "cw": int(w), "ch": int(h)}


class Detector:
    def __init__(self, conf_bird=0.15):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(HERE / "models" / "yolo26l.pt"))
        names = self.model.names
        self.bird = [k for k, v in names.items() if v == "bird"][0]
        self.person = [k for k, v in names.items() if v == "person"][0]
        self.animals = [k for k, v in names.items() if v in ("cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")]
        self.conf = conf_bird

    def __call__(self, images):
        res = self.model.predict(images, imgsz=DETECT_SIDE, conf=self.conf,
                                 classes=[self.bird, self.person] + self.animals, device=0, verbose=False,
                                 half=True)
        out = []
        for r in res:
            b = r.boxes
            out.append([(int(c), float(cf), xy) for c, cf, xy in
                        zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy().tolist())]
                       if b is not None else [])
        return out


def score(shot, prep, exif, dets, det):
    s, H, W = prep["scale"], prep["H"], prep["W"]
    birds, persons, animals = [], [], []
    for cls, conf, (x1, y1, x2, y2) in dets:
        x1, y1, x2, y2 = x1 / s, y1 / s, x2 / s, y2 / s
        d = {"box": [round(x1), round(y1), round(x2), round(y2)], "conf": round(conf, 3),
             "area": round((x2 - x1) * (y2 - y1) / (W * H), 5)}
        if cls == det.bird:
            birds.append(d)
        elif cls == det.person:
            persons.append(d)
        elif conf >= 0.3:
            animals.append(d)
    birds.sort(key=lambda b: -b["area"] * b["conf"])
    birds = birds[:MAX_BIRDS]
    for b in birds:
        x1, y1, x2, y2 = b["box"]
        px, py = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
        crop = prep["gray"][max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
        m = edge_metrics(crop)
        b.update(m or {"acu": None, "acun": 0, "edge": 0, "g90": 0, "lapn": 0, "contrast": 0, "cw": 0, "ch": 0})
    persons = [p for p in persons if p["conf"] >= 0.25]
    fg = cv2.resize(prep["gray"], (prep["W"] // 3, prep["H"] // 3), interpolation=cv2.INTER_AREA)
    return {"v": VERSION, "folder": shot["folder"], "stem": shot["stem"], "files": shot["files"],
            "raw": shot["raw"], "W": W, "H": H, "time": exif.get("time"), "gps": exif.get("gps"),
            "model": exif.get("model"), "birds": birds, "persons": persons, "animals": animals,
            "frame_lap": round(float(cv2.Laplacian(fg, cv2.CV_32F).var()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    SCAN.parent.mkdir(exist_ok=True)
    done = set()
    if SCAN.exists():
        for line in open(SCAN):
            try:
                r = json.loads(line)
                if r.get("v") == VERSION:
                    done.add((r["folder"], r["stem"]))
            except Exception:
                pass
    shots = [s for s in find_shots(a.roots) if (s["folder"], s["stem"]) not in done]
    if a.limit:
        shots = shots[:a.limit]
    print(f"{len(done)} already scanned, {len(shots)} to scan", flush=True)
    if not shots:
        return
    det = Detector()
    out = open(SCAN, "a")
    batch, n, t0 = [], 0, time.time()

    def flush():
        nonlocal n
        res = det([b[1]["small"] for b in batch])
        for (shot, prep, ex), d in zip(batch, res):
            out.write(json.dumps(score(shot, prep, ex, d, det)) + "\n")
        out.flush()
        n += len(batch)
        batch.clear()

    with ThreadPoolExecutor(a.threads) as exr:
        futs = collections.deque()
        it = iter(shots)
        for s in it:
            futs.append(exr.submit(prepare, s))
            if len(futs) >= a.threads * 2:
                break
        while futs:
            shot, prep, ex = futs.popleft().result()
            nx = next(it, None)
            if nx is not None:
                futs.append(exr.submit(prepare, nx))
            if prep is None:
                out.write(json.dumps({"v": VERSION, "folder": shot["folder"], "stem": shot["stem"],
                                      "files": shot["files"], "raw": shot["raw"], "error": ex["err"]}) + "\n")
                n += 1
            else:
                batch.append((shot, prep, ex))
                if len(batch) >= a.batch:
                    flush()
            if n and n % 50 == 0:
                el = time.time() - t0
                print(f"\r{n}/{len(shots)} {n/el:.1f}/s eta {(len(shots)-n)/(n/el)/60:.0f}m", end="", flush=True)
        if batch:
            flush()
    print(f"\ndone {n} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
