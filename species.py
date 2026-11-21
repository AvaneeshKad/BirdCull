#!/usr/bin/env python3
"""Stage B: species ID with BioCLIP 2 on full-resolution RAW crops, one representative
(sharpest bird frame) per burst group. Output: scans/species.jsonl (resumable).
Optional --exclude file: scientific names to drop from candidate list (re-ranking)."""
import argparse, collections, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SCAN, OUT = HERE / "scans" / "scan3.jsonl", HERE / "scans" / "species.jsonl"
CROPS = HERE / "scans" / "crops"
BURST_GAP = 1.5


def load_scan():
    d = {}
    for l in open(SCAN):
        r = json.loads(l)
        if r.get("v") == 3 and "error" not in r:
            d[(r["folder"], r["stem"])] = r
    return d


def groups(scan):
    """burst groups of shots that contain a bird: list of list of records"""
    byf = collections.defaultdict(list)
    for r in scan.values():
        if r["birds"] and r.get("time") is not None:
            byf[r["folder"]].append(r)
        elif r["birds"]:
            byf[r["folder"]].append(dict(r, time=0.0))
    out = []
    for f, rs in byf.items():
        rs.sort(key=lambda r: (r["time"], r["stem"]))
        cur = []
        for r in rs:
            if cur and r["time"] - cur[-1]["time"] <= BURST_GAP:
                cur.append(r)
            else:
                if cur:
                    out.append(cur)
                cur = [r]
        if cur:
            out.append(cur)
    return out


def best_bird(r):
    return max(r["birds"], key=lambda b: (b.get("acu") or 0) * min(1, b["conf"] * 1.5))


def crop_full(r, b):
    import rawpy
    with rawpy.imread(str(Path(r["folder"]) / r["raw"])) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)  # FULL size
    H, W = rgb.shape[:2]
    k = W / r["W"]
    x1, y1, x2, y2 = [v * k for v in b["box"]]
    m = 0.12 * max(x2 - x1, y2 - y1)
    c = rgb[int(max(0, y1 - m)):int(min(H, y2 + m)), int(max(0, x1 - m)):int(min(W, x2 + m))]
    im = Image.fromarray(c)
    if max(im.size) > 1024:
        s = 1024 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--redo-topk", action="store_true")
    a = ap.parse_args()
    scan = load_scan()
    gs = groups(scan)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["folder"], r["stem"]))
    todo = []
    for g in gs:
        rep = max(g, key=lambda r: (best_bird(r).get("acu") or 0))
        if (rep["folder"], rep["stem"]) not in done:
            todo.append((rep, len(g)))
    print(f"{len(gs)} groups, {len(todo)} to identify", flush=True)
    if not todo:
        return
    from bioclip import TreeOfLifeClassifier, Rank
    c = TreeOfLifeClassifier(device="cpu")
    c.apply_filter(c.create_taxa_filter(Rank.CLASS, ["Aves"]))
    c.to("cuda")
    c.device = "cuda"
    import torch
    for k, v in list(vars(c).items()):
        if isinstance(v, torch.Tensor):
            setattr(c, k, v.to("cuda"))
    CROPS.mkdir(exist_ok=True)
    out = open(OUT, "a")
    t0 = time.time()

    def prep(item):
        rep, n = item
        b = best_bird(rep)
        try:
            return item, b, crop_full(rep, b)
        except Exception as e:
            return item, b, None

    n = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for (rep, gn), b, im in ex.map(prep, todo):
            rec = {"folder": rep["folder"], "stem": rep["stem"], "group_n": gn,
                   "time": rep.get("time"), "gps": rep.get("gps")}
            if im is None:
                rec["error"] = True
            else:
                res = c.predict([im], Rank.SPECIES, k=5)
                rec["top5"] = [(x["species"], round(float(x["score"]), 4)) for x in res]
                rec["common"] = [x.get("common_name") for x in res]
                rec["crop_px"] = list(im.size)
                im.thumbnail((256, 256))
                nm = hashlib.md5((rep["folder"] + "|" + rep["stem"]).encode()).hexdigest()[:16] + ".jpg"
                im.save(CROPS / nm, quality=85)
                rec["crop"] = nm
            out.write(json.dumps(rec) + "\n")
            n += 1
            if n % 50 == 0:
                out.flush()
                print(f"\r{n}/{len(todo)} {n/(time.time()-t0):.1f}/s", end="", flush=True)
    print("done")


if __name__ == "__main__":
    main()
