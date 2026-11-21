#!/usr/bin/env python3
"""Review photos sitting in blurry/ and duplicates/ folders.
Keep = move photo (+RAW/JPG/xmp partners) from <folder>/<blurry|duplicates>/ up into <folder>/. Undo moves it back. Never deletes.
Rotate = display rotation only (saved in rotations.json next to this script); photo files are never modified."""
import csv, json, os, re, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import cv2, numpy as np, rawpy

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
ACTLOG = HERE / "review_actions.csv"; ROTF = HERE / "rotations.json"
BASE = "/media/avopti/Extreme SSD"
SUB = {"blurry": "Blurry", "duplicates": "Duplicates"}
PORT = int(os.environ.get("PORT", 8766))
lock = threading.Lock()
ROT = json.load(open(ROTF)) if ROTF.exists() else {}

def scan():
    items = []
    for dp, dn, fn in os.walk(BASE):
        dn[:] = [d for d in dn if d not in (".Trash-1000", ".comments")]
        sub = os.path.basename(dp)
        if sub not in SUB: continue
        groups = {}
        for f in fn:
            groups.setdefault(os.path.splitext(f)[0].lower(), []).append(f)
        for stem, files in sorted(groups.items()):
            items.append(dict(parent=os.path.dirname(dp), sub=sub, files=sorted(files), cat=SUB[sub]))
    items.sort(key=lambda i: (i["parent"], i["sub"], i["files"][0]))
    for n, i in enumerate(items): i["id"] = n
    return items
ITEMS = scan()
def dpath(it, fn): return os.path.join(it["parent"], it["sub"], fn)
def ppath(it, fn): return os.path.join(it["parent"], fn)
def state(it):
    if any(os.path.exists(dpath(it, f)) for f in it["files"]): return "pending"
    if any(os.path.exists(ppath(it, f)) for f in it["files"]): return "kept"
    return "missing"
def cur_files(it): return [(dpath if state(it) == "pending" else ppath)(it, f) for f in it["files"]]

def move_set(it, keep):
    a, b = (dpath, ppath) if keep else (ppath, dpath)
    moved = conflict = 0
    with lock:
        for f in it["files"]:
            s, d = a(it, f), b(it, f)
            if not os.path.exists(s): continue
            if os.path.exists(d): conflict += 1; continue
            os.makedirs(os.path.dirname(d), exist_ok=True); shutil.move(s, d); moved += 1
        sd = os.path.join(it["parent"], it["sub"])
        if keep and os.path.isdir(sd) and not os.listdir(sd): os.rmdir(sd)
        if moved:
            new = not ACTLOG.exists()
            with open(ACTLOG, "a", newline="") as fh:
                w = csv.writer(fh)
                if new: w.writerow(["time", "action", "folder", "sub", "files"])
                w.writerow([time.strftime("%F %T"), "keep" if keep else "undo", it["parent"], it["sub"], "|".join(it["files"])])
    return moved, conflict

def src_of(it):
    fs = [f for f in cur_files(it) if os.path.exists(f)]
    for f in fs:
        if f.lower().endswith((".jpg", ".jpeg")): return f
    for f in fs:
        if not f.lower().endswith(".xmp"): return f

def load_bgr(path, full):
    if path.lower().endswith((".jpg", ".jpeg")): return cv2.imread(path)
    with rawpy.imread(path) as raw:
        if not full:
            try:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    im = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
                    if im is not None: return im
                elif th.format == rawpy.ThumbFormat.BITMAP: return cv2.cvtColor(th.data, cv2.COLOR_RGB2BGR)
            except Exception: pass
        return cv2.cvtColor(raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8), cv2.COLOR_RGB2BGR)

def make_image(it, full):
    r = ROT.get(str(it["id"]), 0) % 360
    out = CACHE / f"{'p' if full else 't'}{it['id']}_{r}.jpg"
    if out.exists(): return out
    src = src_of(it)
    if not src: return None
    im = load_bgr(src, full)
    if im is None: return None
    m = 2600 if full else 480
    s = m / max(im.shape[:2])
    if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if r: im = cv2.rotate(im, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[r])
    tmp = out.with_suffix(".tmp.jpg"); cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 90 if full else 80]); os.replace(tmp, out)
    return out

pool = ThreadPoolExecutor(6)
def prewarm():
    for it in ITEMS:
        try: make_image(it, False)
        except Exception as e: print("thumb fail", it["files"][0], e, file=sys.stderr)
threading.Thread(target=prewarm, daemon=True).start()

def api_items():
    return [dict(id=i["id"], folder=os.path.relpath(i["parent"], BASE), sub=i["cat"], name=i["files"][0], files=len(i["files"]),
                 cat=i["cat"], rot=ROT.get(str(i["id"]), 0), st=state(i)) for i in ITEMS]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/": return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/items": return self.send(200, {"items": api_items()})
        m = re.fullmatch(r"/(thumb|full)/(\d+)", p)
        if m and int(m.group(2)) < len(ITEMS):
            try: f = pool.submit(make_image, ITEMS[int(m.group(2))], m.group(1) == "full").result()
            except Exception as e: return self.send(500, str(e).encode(), "text/plain")
            if f: return self.send(200, f.read_bytes(), "image/jpeg", {"Cache-Control": "max-age=86400"})
        self.send(404, b"", "text/plain")
    def do_POST(self):
        m = re.fullmatch(r"/api/(keep|undo|rot)/(\d+)(?:/(-?\d+))?", self.path)
        if not m or int(m.group(2)) >= len(ITEMS): return self.send(404, {"error": "bad request"})
        it = ITEMS[int(m.group(2))]
        if m.group(1) == "rot":
            with lock:
                ROT[str(it["id"])] = (ROT.get(str(it["id"]), 0) + int(m.group(3))) % 360
                json.dump(ROT, open(ROTF, "w"))
            return self.send(200, {"ok": True, "rot": ROT[str(it["id"])]})
        n, c = move_set(it, m.group(1) == "keep")
        self.send(200, {"ok": c == 0, "moved": n, "conflict": c, "st": state(it)})

if __name__ == "__main__":
    print(f"{len(ITEMS)} photos in blurry/duplicates folders. Open http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
