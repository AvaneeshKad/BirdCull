"""BirdCull engine: scan RAW photos (real RAW decode, YOLO bird/person/animal detection, sharpness) and
decide which photos to SUGGEST for removal. Nothing here ever moves or deletes a photo."""
import collections, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
RAW_EXTS = {".nef", ".nrw", ".arw", ".dng", ".cr2", ".cr3", ".raf", ".orf", ".rw2"}
IMG_EXTS = {".jpg", ".jpeg"}
SIDE_EXTS = {".xmp"}
SKIP_DIRS = {"_culled", "_to_delete", ".comments", "$recycle.bin", "system volume information"}
DETECT_SIDE = 1600
VERSION = 4
# cautious defaults (everything is reviewed by a person before anything is deleted)
SOFT_CUTOFF = 0.15      # sharpness ("acuity") below this = clearly blurry bird. Blurred birds measured 0.11-0.14, sharp 0.18-0.28.
BURST_GAP = 1.5         # seconds between frames = same burst
BURST_MIN = 6           # only thin bursts with at least this many frames
BURST_KEEP = 4          # ...keeping the sharpest N
MIN_CONF, MIN_ACUN = 0.4, 3000

def data_dir():
    d = Path(os.environ.get("BIRDCULL_DATA") or (Path.home() / "BirdCull")); d.mkdir(parents=True, exist_ok=True); return d

def find_shots(roots):
    out = []
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d.lower() not in SKIP_DIRS and not d.startswith("."))
            shots = {}
            for name in fn:
                stem, ext = os.path.splitext(name); ext = ext.lower()
                if ext not in RAW_EXTS | IMG_EXTS | SIDE_EXTS: continue
                if ext in SIDE_EXTS and os.path.splitext(stem)[1].lower() in RAW_EXTS | IMG_EXTS:
                    stem = os.path.splitext(stem)[0]
                shots.setdefault(stem.lower(), []).append(name)
            for stem, files in shots.items():
                raws = [f for f in files if os.path.splitext(f)[1].lower() in RAW_EXTS]
                if not raws: continue
                raws.sort(key=lambda f: (os.path.splitext(f)[1].lower() == ".dng", f))
                out.append({"folder": dp, "stem": stem, "files": sorted(files), "raw": raws[0]})
    return out

def decode(path):
    import rawpy
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True, output_bps=8)

def exif_time(path):
    import exifread
    try:
        with open(path, "rb") as f: tags = exifread.process_file(f, details=False)
        v = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if not v: return None
        t = datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
        sub = tags.get("EXIF SubSecTimeOriginal")
        if sub and str(sub).strip().isdigit(): s = str(sub).strip(); t += int(s) / (10 ** len(s))
        return t
    except Exception: return None

def acuity(g, sigma=0.7):
    """sharpness of a gray crop: p90 of edge-gradient / local-range over contrasty edge pixels (native half-size px)."""
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8
    mag = np.sqrt(gx * gx + gy * gy)
    k = np.ones((9, 9), np.uint8); rng = cv2.dilate(f, k) - cv2.erode(f, k)
    lo, hi = np.percentile(f, [2, 98]); floor = max(6.0, 0.15 * (hi - lo))
    m = rng > floor
    if m.sum() < 30: return None, int(m.sum())
    ratio = mag[m] / rng[m]; sel = mag[m] >= np.percentile(mag[m], 70)
    return float(np.percentile(ratio[sel], 90)), int(m.sum())

def ensure_model():
    """Return the path to the YOLO weights, downloading them on first use (not stored in git)."""
    p = APP_DIR / "models" / "yolo26l.pt"
    if not p.exists():
        import os, shutil, tempfile
        from ultralytics.utils.downloads import attempt_download_asset
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp()
        old = os.getcwd(); os.chdir(tmp)
        try:
            got = attempt_download_asset("yolo26l.pt")
            shutil.move(str(Path(tmp) / "yolo26l.pt") if not os.path.isabs(str(got)) else str(got), str(p))
        finally:
            os.chdir(old); shutil.rmtree(tmp, ignore_errors=True)
    return p

class Detector:
    def __init__(self):
        import torch
        from ultralytics import YOLO
        self.dev = 0 if torch.cuda.is_available() else "cpu"
        self.kw = {"half": True} if self.dev == 0 else {}
        self.model = YOLO(str(ensure_model()))
        n = self.model.names
        self.bird = [k for k, v in n.items() if v == "bird"][0]
        self.person = [k for k, v in n.items() if v == "person"][0]
        self.animals = [k for k, v in n.items() if v in ("cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")]
        self.names = n
    def detect(self, images):
        res = self.model.predict(images, imgsz=DETECT_SIDE, conf=0.15, classes=[self.bird, self.person] + self.animals,
                                 device=self.dev, verbose=False, **self.kw)
        return [[(int(c), float(cf), xy) for c, cf, xy in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy().tolist())] for r in res]
    def sensitive(self, rgb):
        """second, more sensitive pass for frames where nothing was found (tiny birds, lizards...)."""
        r = self.model.predict(rgb, imgsz=2560 if self.dev == 0 else 1920, conf=0.05,
                               classes=[self.bird, self.person] + self.animals, device=self.dev, verbose=False, **self.kw)[0]
        return [(self.names[int(c)], round(float(cf), 3)) for c, cf in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]

def prepare(shot):
    p = Path(shot["folder"]) / shot["raw"]
    try: rgb = decode(p)
    except Exception as e: return shot, None, {"err": f"decode: {type(e).__name__}"}
    H, W = rgb.shape[:2]; s = min(1.0, DETECT_SIDE / max(H, W))
    small = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    return shot, {"rgb": rgb, "small": small, "scale": s, "W": W, "H": H}, {"time": exif_time(p)}

def score(shot, prep, ex, dets, det):
    s, H, W = prep["scale"], prep["H"], prep["W"]
    gray = cv2.cvtColor(prep["rgb"], cv2.COLOR_RGB2GRAY)
    birds, persons, animals = [], [], []
    for cls, conf, (x1, y1, x2, y2) in dets:
        x1, y1, x2, y2 = x1 / s, y1 / s, x2 / s, y2 / s
        d = {"box": [round(x1), round(y1), round(x2), round(y2)], "conf": round(conf, 3), "area": round((x2 - x1) * (y2 - y1) / (W * H), 5)}
        (birds if cls == det.bird else persons if cls == det.person else animals if conf >= 0.3 else []).append(d)
    birds.sort(key=lambda b: -b["area"] * b["conf"]); birds = birds[:6]
    for b in birds:
        x1, y1, x2, y2 = b["box"]; px, py = int(.05 * (x2 - x1)), int(.05 * (y2 - y1))
        crop = gray[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
        a, n = acuity(crop) if min(crop.shape) >= 12 else (None, 0)
        b["acu"] = None if a is None else round(a, 4); b["acun"] = n
    persons = [p for p in persons if p["conf"] >= .25]
    rec = {"v": VERSION, "folder": shot["folder"], "stem": shot["stem"], "files": shot["files"], "raw": shot["raw"], "W": W, "H": H,
           "time": ex.get("time"), "birds": birds, "persons": persons, "animals": animals}
    if not birds and not persons and not animals:
        rec["sensitive"] = det.sensitive(prep["rgb"])
    try: rec["size"] = sum(os.path.getsize(os.path.join(shot["folder"], f)) for f in shot["files"])
    except OSError: rec["size"] = 0
    return rec

class Scanner(threading.Thread):
    """Resumable scan; progress is exposed via .status. Results append to <data>/scan.jsonl."""
    def __init__(self, roots, ddir):
        super().__init__(daemon=True)
        self.roots, self.path = roots, ddir / "scan.jsonl"
        self.stop_flag = threading.Event()
        self.status = {"state": "starting", "done": 0, "total": 0, "eta_min": None, "device": "", "msg": "", "error": None}
    def run(self):
        try: self._run()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.status.update(state="error", error=f"{type(e).__name__}: {e}")
    def _run(self):
        self.status.update(state="listing", msg="Looking for photos…")
        done = set()
        if self.path.exists():
            for l in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(l)
                    if r.get("v") == VERSION: done.add((r["folder"], r["stem"]))
                except Exception: pass
        allshots = find_shots(self.roots)
        todo = [s for s in allshots if (s["folder"], s["stem"]) not in done]
        self.status.update(state="loading", msg="Loading the AI model…", total=len(allshots), done=len(allshots) - len(todo))
        if not todo: return self.status.update(state="done", msg="Scan complete")
        det = Detector(); self.status.update(device="GPU" if det.dev == 0 else "CPU (slow)")
        _keep_awake(True)
        out = open(self.path, "a", encoding="utf-8"); batch = []; t0 = time.time(); n0 = self.status["done"]
        def flush():
            for (shot, prep, ex), d in zip(batch, det.detect([b[1]["small"] for b in batch])):
                out.write(json.dumps(score(shot, prep, ex, d, det)) + "\n")
            out.flush(); self.status["done"] += len(batch); batch.clear()
        threads = 8 if det.dev == 0 else max(2, (os.cpu_count() or 4) // 2)
        self.status.update(state="running", msg="Scanning photos…")
        try:
            with ThreadPoolExecutor(threads) as ex:
                futs = collections.deque(); it = iter(todo)
                for s in it:
                    futs.append(ex.submit(prepare, s))
                    if len(futs) >= threads * 2: break
                while futs:
                    if self.stop_flag.is_set():
                        self.status.update(state="paused", msg="Paused — press Start to continue where it stopped"); return
                    shot, prep, exi = futs.popleft().result()
                    nx = next(it, None)
                    if nx is not None: futs.append(ex.submit(prepare, nx))
                    if prep is None:
                        out.write(json.dumps({"v": VERSION, "folder": shot["folder"], "stem": shot["stem"], "files": shot["files"], "raw": shot["raw"], "error": exi["err"]}) + "\n")
                        self.status["done"] += 1
                    else:
                        batch.append((shot, prep, exi))
                        if len(batch) >= 8: flush()
                    k = self.status["done"] - n0
                    if k: self.status["eta_min"] = round((len(allshots) - self.status["done"]) / (k / (time.time() - t0)) / 60)
                if batch: flush()
        finally:
            out.close(); _keep_awake(False)
        self.status.update(state="done", msg="Scan complete", eta_min=0)

def _keep_awake(on):
    """stop Windows from sleeping during a long scan"""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))

def load_scan(ddir, roots=None):
    S = {}
    p = ddir / "scan.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("v") == VERSION: S[(r["folder"], r["stem"])] = r
    if roots is not None:
        rs = [os.path.normcase(os.path.abspath(r)) for r in roots]
        S = {k: v for k, v in S.items() if any(os.path.normcase(os.path.abspath(k[0])).startswith(r) for r in rs)}
    return S

def best_bird(r):
    b = r.get("birds") or []
    return max(b, key=lambda x: (x.get("acu") or 0) * min(1, x["conf"] * 1.5)) if b else None

def decide(S, favourites=()):
    """-> list of suggestion dicts {folder, stem, files, cat, reason, acu, size}. Cautious: anything uncertain is kept."""
    soft = SOFT_CUTOFF
    fav = [best_bird(r)["acu"] for k, r in S.items() if k[0] in set(favourites) and best_bird(r) and best_bird(r).get("acu")]
    if len(fav) >= 30: soft = min(SOFT_CUTOFF, float(np.percentile(fav, 1)) * 0.95)   # never cut at or above the sharpness of her own picks
    fav_folders = set(favourites)
    byf = collections.defaultdict(list)
    for r in S.values():
        if "error" not in r: byf[r["folder"]].append(r)
    sug, protected = [], set()
    def put(r, cat, why, acu=None): sug.append({"folder": r["folder"], "stem": r["stem"], "files": r["files"], "cat": cat, "reason": why, "acu": acu, "size": r.get("size", 0)})
    def usable(b): return b["conf"] >= MIN_CONF and b.get("acu") is not None and b.get("acun", 0) >= MIN_ACUN
    good = []
    for f, rs in byf.items():
        rs.sort(key=lambda r: (r.get("time") or 0, r["stem"]))
        for r in rs:
            k = (r["folder"], r["stem"])
            if f in fav_folders or any(x.lower().endswith((".xmp", ".dng")) for x in r["files"]) or r["persons"] or r["animals"]:
                protected.add(k); continue          # her picks, edited photos, people, other animals: never touched
            if not r["birds"]:
                s2 = r.get("sensitive")
                if s2 is None or any((n == "bird" and c >= .05) or (n != "bird" and c >= .15) for n, c in s2): protected.add(k); continue
                put(r, "No bird found", "no bird, animal or person detected"); continue
            b = best_bird(r)
            if not usable(b): protected.add(k); continue
            if b["acu"] < soft: put(r, "Blurry bird", f"sharpness {b['acu']:.3f} (below {soft:.2f})", b["acu"]); continue
            good.append(r)
    # thin long bursts, keeping the sharpest frames
    bursts, cur = [], []
    for r in sorted(good, key=lambda r: (r["folder"], r.get("time") or 0, r["stem"])):
        if cur and cur[-1]["folder"] == r["folder"] and r.get("time") and cur[-1].get("time") and r["time"] - cur[-1]["time"] <= BURST_GAP: cur.append(r)
        else:
            if cur: bursts.append(cur)
            cur = [r]
    if cur: bursts.append(cur)
    for g in bursts:
        if len(g) < BURST_MIN: continue
        def sc(r):
            b = best_bird(r); x1, y1, x2, y2 = b["box"]
            edge = x1 <= 2 or y1 <= 2 or x2 >= r["W"] - 2 or y2 >= r["H"] - 2
            return b["acu"] * (0.85 if edge else 1.0) * (1 + min(b["area"], .1))
        g.sort(key=lambda r: -sc(r))
        for r in g[BURST_KEEP:]: put(r, "Burst extra", f"{len(g)}-frame burst; the {BURST_KEEP} sharpest are kept", best_bird(r)["acu"])
    sug.sort(key=lambda s: (s["folder"], s["stem"]))
    for i, s in enumerate(sug): s["id"] = i
    return sug, {"soft_cutoff": round(soft, 3), "protected": len(protected), "scanned": len(S)}
