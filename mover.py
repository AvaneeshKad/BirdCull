#!/usr/bin/env python3
"""mover.py --move : apply plan.csv (move only, never delete). mover.py --undo LOG.csv : put everything back."""
import csv, os, shutil, sys, time
from pathlib import Path
ROOTS = ["/media/avopti/Extreme SSD/Ashwini_Photos/", "/media/avopti/Extreme SSD/Ashwini'sVASPLaptopJul2025/"]
DEST = {"to_delete": "_to_delete", "culled": "_culled"}
if sys.argv[1] == "--move":
    log = Path(__file__).parent / f"moves_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    n = 0
    with open(log, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["from", "to"])
        for r in csv.DictReader(open(Path(__file__).parent / "plan.csv")):
            if r["action"] not in DEST: continue
            folder = r["folder"]
            assert any(folder.startswith(x) for x in ROOTS), folder
            d = os.path.join(folder, DEST[r["action"]]); os.makedirs(d, exist_ok=True)
            for fn in r["files"].split("|"):
                src, dst = os.path.join(folder, fn), os.path.join(d, fn)
                if not os.path.exists(src) or os.path.exists(dst): continue
                shutil.move(src, dst); w.writerow([src, dst]); n += 1
            fh.flush()
    print("moved", n, "files; log", log)
elif sys.argv[1] == "--undo":
    n = 0
    for r in csv.DictReader(open(sys.argv[2])):
        if os.path.exists(r["to"]) and not os.path.exists(r["from"]):
            shutil.move(r["to"], r["from"]); n += 1
    print("restored", n)
