#!/usr/bin/env python3
"""
bird_cull.py - a culling tool made for bird photographers.

For every photo it:
  1. finds the birds with an AI model (YOLO, runs on your NVIDIA GPU)
  2. measures sharpness ON THE BIRD ONLY, at full resolution
     (soft backgrounds, sky and bokeh are ignored)
  3. groups bursts by capture time and keeps the sharpest frames

Then it sorts photos into three folders (inside the folder they came from):
  soft/          a bird is there, but even the sharpest bird is not sharp
  no_bird/       no bird found (empty perch, misfires, flew out of frame)
  burst_extras/  sharp, but not among the best N frames of its burst

Nothing is deleted. RAW + JPEG + .xmp with the same name always move
together. Every subfolder is searched. Scans are saved, so you can stop
any time (Ctrl+C) and continue later, and re-tuning is instant.

Setup (once):
    conda create -n birds python=3.12 -y
    conda activate birds
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install ultralytics rawpy exifread
    python -c "import torch; print(torch.cuda.is_available())"   # should say True

Use:
    python bird_cull.py "/path/to/photos"           # scan + preview page
    python bird_cull.py "/path/to/photos" --move    # move the flagged photos
    python bird_cull.py "/path/to/photos" --undo    # put back the last move

The preview page is  /path/to/photos/.bird_review/review.html
(hidden folder - press Ctrl+H in the file manager to see it).
"""
import argparse
import collections
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAW_EXTS = {".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw",
            ".dng", ".raf", ".orf", ".rw2", ".raw", ".pef", ".srw", ".3fr",
            ".iiq", ".erf", ".kdc", ".mrw", ".x3f"}
IMG_EXTS = {".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".webp", ".bmp"}
SIDECAR_EXTS = {".xmp", ".pp3", ".dop", ".on1"}
OUT_DIRS = ("soft", "no_bird", "burst_extras")
SKIP_DIRS = set(OUT_DIRS) | {"blurry", "duplicates", ".bird_review", ".photo_review"}
CACHE_NAME = ".bird_scan.json"
REVIEW_DIR = ".bird_review"
SCORE_VERSION = 1
DETECT_LONG_SIDE = 1600      # image size handed to the detector
MAX_BIRDS = 6                # birds per frame that get a sharpness check


# ------------------------------------------------------------------ reading

def decode(path, reduce=1):
    """Photo as a BGR array. RAW: the camera's embedded full-size JPEG."""
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
             4: cv2.IMREAD_REDUCED_COLOR_4}[reduce]
    try:
        if path.suffix.lower() in RAW_EXTS:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                try:
                    th = raw.extract_thumb()
                    if th.format == rawpy.ThumbFormat.JPEG:
                        return cv2.imdecode(np.frombuffer(th.data, np.uint8), flags)
                    if th.format == rawpy.ThumbFormat.BITMAP:
                        return cv2.cvtColor(th.data, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.imread(str(path), flags)
    except Exception:
        return None


def capture_time(path):
    """Seconds (with sub-seconds if the camera recorded them), or None."""
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False,
                                         stop_tag="SubSecTimeOriginal")
        v = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if v:
            t = datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S").timestamp()
            sub = tags.get("EXIF SubSecTimeOriginal")
            if sub and str(sub).strip().isdigit():
                s = str(sub).strip()
                t += int(s) / (10 ** len(s))
            return t
    except Exception:
        pass
    m = re.search(r"(\d{8})_(\d{6})", path.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    return None


def prepare(path):
    """Runs in a background thread: read + decode one photo."""
    bgr = decode(path)
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    s = min(1.0, DETECT_LONG_SIDE / max(H, W))
    small = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) \
        if s < 1 else bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {"small": small, "scale": s, "gray": gray, "W": W, "H": H,
            "time": capture_time(path)}


# ------------------------------------------------------------------ scoring

def bird_sharpness(crop):
    """How sharp a bird crop is, independent of bird size and brightness.

    The crop is resized to a fixed size so a big and a small bird are
    judged at the same scale; each small tile's edge energy is divided by
    its contrast, so a dark crow and a white egret score alike; the best
    tile is used, because the head/eye is usually the sharpest part when
    the shot is in focus."""
    h, w = crop.shape
    if h < 8 or w < 8:
        return 0.0
    s = 512 / max(h, w)
    crop = cv2.resize(crop, (max(16, int(w * s)), max(16, int(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    crop = cv2.GaussianBlur(crop, (3, 3), 0).astype(np.float64)
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    h, w = lap.shape
    n = 6
    best, fallback = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            lv, gv = lap[sl].var(), crop[sl].var()
            r = lv / (gv + 10.0)
            fallback = max(fallback, r)
            if gv > 20:          # ignore flat tiles (sky, smooth background)
                best = max(best, r)
    return float(100 * (best or fallback))


def frame_sharpness(gray):
    """Fallback for frames with no bird: best tile of the whole frame."""
    h, w = gray.shape
    s = 1500 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F)
    h, w = lap.shape
    return float(max(lap[i*h//8:(i+1)*h//8, j*w//8:(j+1)*w//8].var()
                     for i in range(8) for j in range(8)))


class Detector:
    def __init__(self, model_name, target, conf, imgsz):
        import torch
        from ultralytics import YOLO
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            print("WARNING: no NVIDIA GPU found by PyTorch - running on CPU (slow).\n"
                  "  Check: python -c \"import torch; print(torch.cuda.is_available())\"")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        self.model = YOLO(model_name)
        ids = [k for k, v in self.model.names.items() if v == target]
        if not ids:
            sys.exit(f"The model has no '{target}' class.")
        self.cls, self.conf, self.imgsz = ids, conf, imgsz

    def __call__(self, images):
        res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf,
                                 classes=self.cls, device=0 if self.cuda else "cpu",
                                 verbose=False)
        out = []
        for r in res:
            b = r.boxes
            out.append(list(zip(b.xyxy.cpu().numpy().tolist(),
                                b.conf.cpu().numpy().tolist())) if b is not None else [])
        return out


def score_shot(prep, boxes):
    s, H, W = prep["scale"], prep["H"], prep["W"]
    birds = []
    for (x1, y1, x2, y2), conf in boxes:
        x1, y1, x2, y2 = x1 / s, y1 / s, x2 / s, y2 / s
        birds.append({"box": [x1, y1, x2, y2], "conf": conf,
                      "area": (x2 - x1) * (y2 - y1) / (W * H)})
    birds.sort(key=lambda b: -b["area"] * b["conf"])
    birds = birds[:MAX_BIRDS]
    for b in birds:
        x1, y1, x2, y2 = b["box"]
        px, py = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        X1, Y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        X2, Y2 = min(W, int(x2 + px)), min(H, int(y2 + py))
        b["sharp"] = bird_sharpness(prep["gray"][Y1:Y2, X1:X2])
        b["box"] = [round(v) for v in b["box"]]
        b["conf"] = round(b["conf"], 3)
        b["area"] = round(b["area"], 5)
        b["sharp"] = round(b["sharp"], 2)
    return {"W": W, "H": H, "time": prep["time"], "birds": birds,
            "frame_sharp": round(frame_sharpness(prep["gray"]), 1)}


# ------------------------------------------------------------------ finding

def find_shots(root):
    """{folder: {stem: [files]}} - same-name RAW/JPG/xmp = one shot."""
    result = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        shots = {}
        for name in filenames:
            stem, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext in RAW_EXTS or ext in IMG_EXTS or ext in SIDECAR_EXTS:
                stem = stem.lower()
                if ext in SIDECAR_EXTS and os.path.splitext(stem)[1] in RAW_EXTS | IMG_EXTS:
                    stem = os.path.splitext(stem)[0]          # photo.NEF.xmp
                shots.setdefault(stem, []).append(Path(dirpath) / name)
        shots = {k: sorted(v) for k, v in shots.items()
                 if any(f.suffix.lower() in RAW_EXTS | IMG_EXTS for f in v)}
        if shots:
            result[Path(dirpath)] = shots
    return result


def primary(files):
    """JPEG if present (fast to read), else the RAW."""
    for f in files:
        if f.suffix.lower() in IMG_EXTS:
            return f
    for f in files:
        if f.suffix.lower() in RAW_EXTS:
            return f


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60}m" if sec >= 3600 else f"{sec // 60}m {sec % 60}s"


def ordered_prefetch(paths, threads):
    """Decode photos in background threads, hand them out in order."""
    with ThreadPoolExecutor(threads) as ex:
        pending = collections.deque()
        it = iter(paths)
        for p in it:
            pending.append((p, ex.submit(prepare, p)))
            if len(pending) >= threads * 2:
                break
        while pending:
            p, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, ex.submit(prepare, nxt)))
            yield p, fut.result()


def scan(root, args):
    folders = find_shots(root)
    tag = {"v": SCORE_VERSION, "model": args.model, "target": args.target}
    caches, todo = {}, []
    for folder, shots in folders.items():
        try:
            old = json.loads((folder / CACHE_NAME).read_text())
        except Exception:
            old = {}
        cache = {}
        for stem, files in shots.items():
            p = primary(files)
            st = p.stat()
            key = {"file": p.name, "size": st.st_size, "mtime": st.st_mtime_ns, **tag}
            c = old.get(stem)
            if not args.rescan and c and all(c.get(k) == v for k, v in key.items()) \
                    and ("birds" in c or c.get("error")):
                cache[stem] = c
            else:
                cache[stem] = key
                todo.append((folder, stem, p))
        caches[folder] = cache

    total = sum(len(s) for s in folders.values())
    print(f"Found {total} photos in {len(folders)} folders: "
          f"{total - len(todo)} already scanned, {len(todo)} to scan.")

    def save(dirty):
        for folder in dirty:
            try:
                (folder / CACHE_NAME).write_text(json.dumps(caches[folder]))
            except OSError as e:
                print(f"\n  (couldn't save scan in {folder}: {e})")

    if not todo:
        return folders, caches

    detector = Detector(args.model, args.target, args.conf, args.imgsz)
    where = {str(p): (folder, stem) for folder, stem, p in todo}
    dirty, done, start = set(), 0, time.time()
    batch = []

    def flush():
        nonlocal done
        boxes = detector([b[1]["small"] for b in batch])
        for (p, prep), bx in zip(batch, boxes):
            folder, stem = where[str(p)]
            caches[folder][stem].update(score_shot(prep, bx))
            dirty.add(folder)
        done += len(batch)
        batch.clear()

    try:
        for p, prep in ordered_prefetch([t[2] for t in todo], args.threads):
            if prep is None:
                folder, stem = where[str(p)]
                caches[folder][stem]["error"] = True
                dirty.add(folder)
                done += 1
            else:
                batch.append((p, prep))
                if len(batch) >= args.batch:
                    flush()
            if done and (done % 25 == 0 or done == len(todo)):
                el = time.time() - start
                print(f"\r  scanned {done}/{len(todo)}  {done / el:.1f}/s  "
                      f"about {fmt(el / done * (len(todo) - done))} left   ",
                      end="", flush=True)
            if len(dirty) and done % 200 == 0:
                save(dirty)
                dirty.clear()
        if batch:
            flush()
    except KeyboardInterrupt:
        print("\nStopped. Saving progress - run the same command again to continue.")
        save(dirty)
        sys.exit(1)
    save(dirty)
    print(f"\r  scanned {done}/{len(todo)} in {fmt(time.time() - start)}            ")
    return folders, caches


# ------------------------------------------------------------------ deciding

def best_bird(c):
    return max(c["birds"], key=lambda b: b["sharp"]) if c.get("birds") else None


def decide(folders, caches, args):
    all_scores = [best_bird(c)["sharp"] for f in caches for c in caches[f].values()
                  if c.get("birds")]
    plan, thresholds = [], {}
    for folder, shots in folders.items():
        cache = caches[folder]
        items = []
        for stem in shots:
            c = cache.get(stem, {})
            if c.get("error"):
                print(f"  skip (can't read): {folder / c['file']}")
            elif "birds" in c:
                items.append((stem, c))

        # threshold for "soft": a number you give, or automatic per folder
        if args.soft is not None:
            thr = args.soft
        else:
            local = [best_bird(c)["sharp"] for _, c in items if c["birds"]]
            ref = local if len(local) >= 15 else all_scores
            thr = args.soft_ratio * float(np.percentile(ref, 90)) if ref else 0
        thresholds[folder] = thr

        soft, no_bird, ok = [], [], []
        for stem, c in items:
            bb = best_bird(c)
            if bb is None:
                if not args.keep_no_bird:
                    no_bird.append(stem)
                else:
                    ok.append((stem, c, 0.0))
            elif bb["sharp"] < thr:
                soft.append(stem)
            else:
                ok.append((stem, c, bb["sharp"]))

        # bursts: consecutive shots taken close together in time
        groups = []
        if args.keep > 0:
            ok.sort(key=lambda x: (x[1].get("time") or 0, x[1]["file"]))
            cur = []
            for x in ok:
                t, pt = x[1].get("time"), cur[-1][1].get("time") if cur else None
                if cur and t is not None and pt is not None and t - pt <= args.burst_gap:
                    cur.append(x)
                else:
                    if len(cur) > args.keep:
                        groups.append(cur)
                    cur = [x]
            if len(cur) > args.keep:
                groups.append(cur)
        bursts = []
        for g in groups:
            ranked = sorted(g, key=lambda x: -x[2])
            bursts.append(([x[0] for x in g],
                           {x[0] for x in ranked[:args.keep]},
                           [x[0] for x in ranked[args.keep:]]))
        if soft or no_bird or bursts:
            plan.append({"folder": folder, "soft": soft, "no_bird": no_bird,
                         "bursts": bursts})
    return plan, thresholds, all_scores


# ------------------------------------------------------------------ report

def make_thumbs(job):
    src, frame_dst, bird_dst, box, W, H = job
    img = decode(Path(src), reduce=1 if box else 4)
    if img is None:
        return
    h, w = img.shape[:2]
    if box:
        sx, sy = w / W, h / H
        x1, y1, x2, y2 = box
        # zoomed bird crop at full resolution
        cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
        half = max(x2 - x1, y2 - y1) * 0.6 * max(sx, sy)
        X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half))
        X2, Y2 = int(min(w, cx + half)), int(min(h, cy + half))
        crop = img[Y1:Y2, X1:X2]
        if crop.size:
            s = 360 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s,
                              interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            cv2.imwrite(bird_dst, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    s = 360 / max(h, w)
    frame = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if box:
        k = s * w / W
        cv2.rectangle(frame, (int(box[0] * k), int(box[1] * k)),
                      (int(box[2] * k), int(box[3] * k)), (0, 255, 255), 2)
    cv2.imwrite(frame_dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])


def write_report(root, folders, caches, plan, thresholds, args):
    rdir = root / REVIEW_DIR
    tdir = rdir / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    jobs, names = [], {}

    def need(folder, stem):
        c = caches[folder][stem]
        bb = best_bird(c)
        p = folder / c["file"]
        h = hashlib.md5(f"{p}|{c.get('mtime')}".encode()).hexdigest()
        fr, bd = f"{h}_f.jpg", (f"{h}_b.jpg" if bb else None)
        names[(folder, stem)] = (fr, bd)
        if not (tdir / fr).exists():
            jobs.append((str(p), str(tdir / fr), str(tdir / bd) if bd else None,
                         bb["box"] if bb else None, c["W"], c["H"]))

    for e in plan:
        for s in e["soft"] + e["no_bird"]:
            need(e["folder"], s)
        for members, _, _ in e["bursts"]:
            for s in members:
                need(e["folder"], s)
    if jobs:
        print(f"Making {len(jobs)} preview images...")
        with ThreadPoolExecutor(args.threads) as ex:
            list(ex.map(make_thumbs, jobs))

    def card(folder, stem, label, color):
        c = caches[folder][stem]
        fr, bd = names[(folder, stem)]
        bb = best_bird(c)
        files = ", ".join(f.name for f in folders[folder][stem])
        info = (f"bird sharpness <b>{bb['sharp']:.0f}</b> · {len(c['birds'])} bird(s) · "
                f"bird = {100 * bb['area']:.1f}% of frame") if bb else "no bird found"
        imgs = f'<img src="thumbs/{fr}" loading="lazy">'
        if bd:
            imgs += f'<img src="thumbs/{bd}" loading="lazy" class="zoom">'
        return (f'<div class="card" style="border-color:{color}"><div class="imgs">{imgs}</div>'
                f'<div class="cap"><b style="color:{color}">{label}</b> · {html.escape(files)}'
                f'<br>{info}</div></div>')

    tot = {k: sum(len(e[k]) for e in plan) for k in ("soft", "no_bird")}
    tot["burst"] = sum(len(d) for e in plan for _, _, d in e["bursts"])
    out = ["<!doctype html><meta charset=utf-8><title>Bird cull review</title><style>"
           "body{font-family:system-ui,sans-serif;background:#181818;color:#ddd;margin:20px}"
           ".card{display:inline-block;margin:5px;border:3px solid;vertical-align:top;"
           "background:#000;width:724px;max-width:95vw}"
           ".imgs{display:flex;gap:4px}.imgs img{width:360px;max-width:47vw;object-fit:contain}"
           ".cap{font-size:13px;padding:5px;word-break:break-all}"
           "h2{margin-top:44px;border-bottom:1px solid #555}h3{color:#aaa}"
           ".row{margin:6px 0 22px}nav a{color:#8cf;margin-right:14px}</style>",
           "<h1>Bird cull review</h1>",
           f"<p>Would move: <b>{tot['soft']}</b> soft · <b>{tot['no_bird']}</b> no bird · "
           f"<b>{tot['burst']}</b> burst extras. Yellow box = the bird that was judged; "
           f"right image = that bird zoomed in.<br>Soft photos are sorted "
           f"<b>closest-to-sharp first</b>, so the ones worth double-checking are at the top "
           f"of each list.</p>"]
    for e in plan:
        folder = e["folder"]
        rel = str(folder.relative_to(root)) if folder != root else "(top folder)"
        out.append(f"<h2>{html.escape(rel)}</h2><p>soft below sharpness "
                   f"{thresholds[folder]:.0f}</p>")
        if e["soft"]:
            out.append(f'<h3>Soft - bird not sharp ({len(e["soft"])})</h3><div class="row">')
            for s in sorted(e["soft"], key=lambda s: -best_bird(caches[folder][s])["sharp"]):
                out.append(card(folder, s, "soft", "#f55"))
            out.append("</div>")
        if e["bursts"]:
            n = sum(len(d) for _, _, d in e["bursts"])
            out.append(f"<h3>Bursts - keeping the {args.keep} sharpest ({n} extras)</h3>")
            for members, kept, _ in e["bursts"]:
                out.append('<div class="row">')
                for s in members:
                    out.append(card(folder, s, "keep", "#4c4") if s in kept
                               else card(folder, s, "burst extra", "#f93"))
                out.append("</div>")
        if e["no_bird"]:
            out.append(f'<h3>No bird found ({len(e["no_bird"])})</h3><div class="row">')
            out += [card(folder, s, "no bird", "#99f") for s in e["no_bird"]]
            out.append("</div>")
    page = rdir / "review.html"
    page.write_text("\n".join(out), encoding="utf-8")
    return page, tot


# ------------------------------------------------------------------ moving

def move(root, folders, plan):
    rdir = root / REVIEW_DIR
    rdir.mkdir(exist_ok=True)
    log = rdir / f"moved_{datetime.now():%Y%m%d_%H%M%S}.csv"
    n = 0
    with open(log, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to"])
        for e in plan:
            folder = e["folder"]
            todo = [("soft", s) for s in e["soft"]] + \
                   [("no_bird", s) for s in e["no_bird"]] + \
                   [("burst_extras", s) for _, _, d in e["bursts"] for s in d]
            for sub, stem in todo:
                dest = folder / sub
                dest.mkdir(exist_ok=True)
                for f in folders[folder][stem]:
                    target = dest / f.name
                    if target.exists():
                        print(f"  skip (already there): {target}")
                        continue
                    shutil.move(str(f), str(target))
                    w.writerow([str(f), str(target)])
                    n += 1
    print(f"\nMoved {n} files (log: {log}).")
    print("Look through soft/, no_bird/ and burst_extras/, then delete what you don't want.")
    print("Changed your mind? Run the same command with --undo")


def undo(root):
    logs = sorted((root / REVIEW_DIR).glob("moved_*.csv"))
    if not logs:
        print("Nothing to undo.")
        return
    n = 0
    with open(logs[-1], newline="") as fh:
        for row in csv.DictReader(fh):
            src, dst = Path(row["to"]), Path(row["from"])
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                n += 1
    logs[-1].rename(logs[-1].with_suffix(".undone"))
    print(f"Put back {n} files.")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--move", action="store_true", help="move flagged photos")
    ap.add_argument("--undo", action="store_true", help="put back the last move")
    g = ap.add_argument_group("tuning (instant to change after the first scan)")
    g.add_argument("--soft", type=float, default=None,
                   help="fixed bird-sharpness cutoff (default: automatic per folder)")
    g.add_argument("--soft-ratio", type=float, default=0.35,
                   help="automatic cutoff = this x the folder's top-10%% sharpness "
                        "(default 0.35; lower = stricter about calling a photo soft)")
    g.add_argument("--keep", type=int, default=3,
                   help="frames to keep per burst (default 3; 0 = keep all)")
    g.add_argument("--burst-gap", type=float, default=1.5,
                   help="seconds between frames that still count as one burst (default 1.5)")
    g.add_argument("--keep-no-bird", action="store_true",
                   help="don't move frames where no bird was found")
    d = ap.add_argument_group("detection (changing these re-scans)")
    d.add_argument("--model", default="yolo26l.pt",
                   help="YOLO model (default yolo26l.pt; yolo26x.pt = slower, a bit better)")
    d.add_argument("--target", default="bird", help=argparse.SUPPRESS)
    d.add_argument("--conf", type=float, default=0.2,
                   help="detection confidence (default 0.2; lower finds more hidden birds)")
    d.add_argument("--imgsz", type=int, default=1280,
                   help="detector resolution (default 1280; 1600 finds smaller birds)")
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--threads", type=int, default=4,
                   help="photos decoded in parallel (use 2 for a slow USB hard drive)")
    d.add_argument("--rescan", action="store_true", help="ignore saved scans")
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    if args.undo:
        return undo(root)

    folders, caches = scan(root, args)
    plan, thresholds, scores = decide(folders, caches, args)
    page, tot = write_report(root, folders, caches, plan, thresholds, args)
    total = sum(len(s) for s in folders.values())
    if scores:
        p = lambda q: np.percentile(scores, q)
        print(f"\nBird sharpness: lowest {min(scores):.0f} · 25% {p(25):.0f} · "
              f"middle {p(50):.0f} · 90% {p(90):.0f} · highest {max(scores):.0f}")
    print(f"{total} photos: {tot['soft']} soft, {tot['no_bird']} no bird, "
          f"{tot['burst']} burst extras.")
    print(f"Review page (open in a browser):\n  {page}")
    if args.move:
        move(root, folders, plan)
    else:
        print("Preview only - nothing moved. Add --move when it looks right.")


if __name__ == "__main__":
    main()
