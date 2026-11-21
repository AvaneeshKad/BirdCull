"""BirdCull local web app. Nothing is deleted until the user presses the final button on the Finish page."""
import csv, hashlib, json, os, re, string, sys, threading, time, webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import cv2, numpy as np
from . import core

DD = core.data_dir()
CACHE = DD / "cache"; CACHE.mkdir(exist_ok=True)
lock = threading.Lock()
def jload(name, default):
    try: return json.load(open(DD / name, encoding="utf-8"))
    except Exception: return default
def jsave(name, obj):
    tmp = DD / (name + ".tmp"); json.dump(obj, open(tmp, "w", encoding="utf-8")); os.replace(tmp, DD / name)
SET = jload("settings.json", {"roots": [], "favourites": []})
KEPT = set(jload("kept.json", []))
ROT = jload("rotations.json", {})
ITEMS = jload("suggestions.json", {"items": [], "info": {}})
scanner = None
key = lambda it: it["folder"] + "|" + it["stem"]

def in_roots(path):
    p = os.path.normcase(os.path.abspath(path))
    return any(p == r or p.startswith(r + os.sep) for r in (os.path.normcase(os.path.abspath(x)) for x in SET["roots"]))

def ls(path):
    if not path:
        if sys.platform == "win32": return [{"name": f"{d}:\\", "path": f"{d}:\\"} for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")], ""
        path = os.environ.get("BIRDCULL_HOME") or str(Path.home())
    path = os.path.abspath(path)
    try: names = sorted(n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)) and not n.startswith((".", "$")))
    except OSError: names = []
    return [{"name": n, "path": os.path.join(path, n)} for n in names], path

def suggest():
    S = core.load_scan(DD, SET["roots"])
    items, info = core.decide(S, SET.get("favourites", []))
    ITEMS.update(items=items, info=info); jsave("suggestions.json", ITEMS)

def src_file(it):
    for f in it["files"]:
        p = os.path.join(it["folder"], f)
        if f.lower().endswith((".jpg", ".jpeg")) and os.path.exists(p): return p
    for f in it["files"]:
        p = os.path.join(it["folder"], f)
        if not f.lower().endswith(".xmp") and os.path.exists(p): return p

def make_image(it, full):
    r = ROT.get(key(it), 0) % 360
    out = CACHE / (("p" if full else "t") + hashlib.md5(key(it).encode()).hexdigest() + f"_{r}.jpg")
    if out.exists(): return out
    src = src_file(it)
    if not src: return None
    if src.lower().endswith((".jpg", ".jpeg")): im = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        import rawpy
        with rawpy.imread(src) as raw:
            im = None
            if not full:
                try:
                    th = raw.extract_thumb()
                    if th.format == rawpy.ThumbFormat.JPEG: im = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
                except Exception: pass
            if im is None: im = cv2.cvtColor(raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8), cv2.COLOR_RGB2BGR)
    if im is None: return None
    m = 2600 if full else 480; s = m / max(im.shape[:2])
    if s < 1: im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if r: im = cv2.rotate(im, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[r])
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90 if full else 78])
    tmp = out.with_suffix(".tmp"); tmp.write_bytes(buf.tobytes()); os.replace(tmp, out); return out
pool = ThreadPoolExecutor(4)

def pending():
    return [it for it in ITEMS["items"] if key(it) not in KEPT]
def summary():
    p = pending(); n = size = 0
    for it in p:
        for f in it["files"]:
            fp = os.path.join(it["folder"], f)
            if os.path.exists(fp): n += 1; size += os.path.getsize(fp)
    return {"photos": len(p), "files": n, "bytes": size, "kept": len(KEPT & {key(i) for i in ITEMS["items"]})}

def finish(mode):
    if mode not in ("recycle", "permanent"): raise ValueError("bad mode")
    if scanner and scanner.is_alive() and scanner.status["state"] == "running": raise RuntimeError("scan still running")
    ok = fail = 0; freed = 0; log = DD / "removed_photos_log.csv"; new = not log.exists()
    with lock, open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["time", "mode", "path", "bytes", "result"])
        for it in pending():
            for f in it["files"]:
                fp = os.path.join(it["folder"], f)
                if not os.path.isfile(fp): continue
                if not in_roots(fp): w.writerow([time.strftime("%F %T"), mode, fp, 0, "refused: outside chosen folders"]); fail += 1; continue
                try:
                    sz = os.path.getsize(fp)
                    if mode == "recycle":
                        from send2trash import send2trash; send2trash(fp)
                    else: os.remove(fp)
                    ok += 1; freed += sz; w.writerow([time.strftime("%F %T"), mode, fp, sz, "ok"])
                except Exception as e: fail += 1; w.writerow([time.strftime("%F %T"), mode, fp, 0, f"failed: {e}"])
    # forget removed items
    ITEMS["items"] = [it for it in ITEMS["items"] if key(it) in KEPT]; jsave("suggestions.json", ITEMS)
    return {"deleted": ok, "failed": fail, "bytes": freed, "log": str(log)}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
        if p == "/": return self.send(200, (core.APP_DIR / "birdcull" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/state":
            st = scanner.status if scanner else {"state": "idle", "done": 0, "total": 0}
            return self.send(200, {"scan": st, "settings": SET, "info": ITEMS["info"], "n_items": len(ITEMS["items"]), "summary": summary(), "data_dir": str(DD)})
        if p == "/api/ls": d, cur = ls(q.get("path", [""])[0]); return self.send(200, {"dirs": d, "path": cur, "parent": os.path.dirname(cur) if cur else ""})
        if p == "/api/items":
            return self.send(200, {"items": [dict(id=i["id"], folder=i["folder"], name=i["files"][0], files=len(i["files"]), cat=i["cat"], reason=i["reason"], acu=i["acu"],
                                                   kept=key(i) in KEPT, rot=ROT.get(key(i), 0)) for i in ITEMS["items"]]})
        m = re.fullmatch(r"/(thumb|full)/(\d+)", p)
        if m and int(m.group(2)) < len(ITEMS["items"]):
            try: f = pool.submit(make_image, ITEMS["items"][int(m.group(2))], m.group(1) == "full").result()
            except Exception as e: return self.send(500, str(e).encode(), "text/plain")
            if f: return self.send(200, f.read_bytes(), "image/jpeg", {"Cache-Control": "max-age=3600"})
        self.send(404, b"", "text/plain")
    def do_POST(self):
        global scanner
        p = self.path; b = self.body()
        if p == "/api/scan/start":
            roots = [r for r in b.get("roots", []) if os.path.isdir(r)]
            if not roots: return self.send(400, {"error": "Choose a folder first"})
            if scanner and scanner.is_alive(): return self.send(409, {"error": "already running"})
            SET["roots"] = roots; SET["favourites"] = [f for f in b.get("favourites", []) if os.path.isdir(f)]; jsave("settings.json", SET)
            scanner = core.Scanner(roots, DD); scanner.start(); return self.send(200, {"ok": True})
        if p == "/api/scan/stop":
            if scanner: scanner.stop_flag.set()
            return self.send(200, {"ok": True})
        if p == "/api/suggest":
            if scanner and scanner.is_alive() and scanner.status["state"] == "running": return self.send(409, {"error": "scan still running"})
            suggest(); return self.send(200, {"ok": True, "n": len(ITEMS["items"])})
        m = re.fullmatch(r"/api/(keep|unkeep|rot)/(\d+)(?:/(-?\d+))?", p)
        if m and int(m.group(2)) < len(ITEMS["items"]):
            it = ITEMS["items"][int(m.group(2))]
            with lock:
                if m.group(1) == "rot": ROT[key(it)] = (ROT.get(key(it), 0) + int(m.group(3))) % 360; jsave("rotations.json", ROT); return self.send(200, {"ok": True, "rot": ROT[key(it)]})
                (KEPT.add if m.group(1) == "keep" else KEPT.discard)(key(it)); jsave("kept.json", sorted(KEPT))
            return self.send(200, {"ok": True, "summary": summary()})
        if p == "/api/finish":
            if not b.get("confirm"): return self.send(400, {"error": "not confirmed"})
            if b.get("mode") == "permanent" and b.get("typed") != "DELETE": return self.send(400, {"error": "type DELETE to confirm"})
            try: return self.send(200, finish(b.get("mode")))
            except Exception as e: return self.send(500, {"error": str(e)})
        self.send(404, {"error": "unknown"})

def main():
    port = int(os.environ.get("PORT", 8787))
    for _ in range(20):
        try: srv = ThreadingHTTPServer(("127.0.0.1", port), H); break
        except OSError: port += 1
    url = f"http://127.0.0.1:{port}"
    print("BirdCull is running at", url, "— close this window to quit.")
    if not os.environ.get("NO_BROWSER"): threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass

if __name__ == "__main__": main()
