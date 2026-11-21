#!/usr/bin/env python3
"""Stage C: decide. Writes plan.csv (one row per shot, action + reason). Never moves anything.
python decide.py [--hard 0.14] [--soft 0.18]"""
import argparse, collections, csv, json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import load, folder_stats, approved_folders, best

HERE = Path(__file__).parent
BURST_GAP = 1.5
KEEP = 3
MIN_CONF, MIN_ACUN = 0.4, 3000
RARE_DAYS = 3          # species seen on <= this many days in the whole library = rare
CONFIDENT = 0.50       # BioCLIP top-1 probability needed to call an ID confident


def load_species(exclude):
    sp = {}
    p = HERE / "scans" / "species.jsonl"
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            if "top5" not in r:
                continue
            cand = [(n, s, c) for (n, s), c in zip(r["top5"], r["common"]) if n not in exclude]
            tot = sum(s for _, s, _ in cand) or 1
            r["cand"] = [(n, s / tot, c) for n, s, c in cand]   # renormalised after exclusions
            sp[(r["folder"], r["stem"])] = r
    return sp


def groups_of(shots):
    byf = collections.defaultdict(list)
    for r in shots:
        byf[r["folder"]].append(r)
    out = []
    for f, rs in byf.items():
        rs.sort(key=lambda r: ((r.get("time") or 0), r["stem"]))
        cur = []
        for r in rs:
            if cur and r.get("time") and cur[-1].get("time") and r["time"] - cur[-1]["time"] <= BURST_GAP:
                cur.append(r)
            else:
                if cur:
                    out.append(cur)
                cur = [r]
        if cur:
            out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", type=float, default=0.14)
    ap.add_argument("--soft", type=float, default=0.18)
    ap.add_argument("--exclude", default=str(HERE / "species_exclude.txt"))
    ap.add_argument("--keep-list", default=str(HERE / "never_delete_stems.txt"),
                    help="folder|stem lines I reviewed by eye and judged NOT obviously blurred")
    ap.add_argument("--out", default=str(HERE / "plan.csv"))
    a = ap.parse_args()

    S = load()
    by = folder_stats(S)
    A = approved_folders(by)
    exclude = set(open(a.exclude).read().split("\n")) if os.path.exists(a.exclude) else set()
    exclude.discard("")
    sp = load_species(exclude)
    keep_eye = set(open(a.keep_list).read().split("\n")) if os.path.exists(a.keep_list) else set()

    # cross-copy protection: same filename+size as a file inside an approved folder
    prot_keys = set()
    for f in A:
        for r in by[f]:
            for fn in r["files"]:
                try:
                    prot_keys.add((fn.lower(), os.stat(os.path.join(f, fn)).st_size))
                except OSError:
                    pass

    def cross_copy(r):
        for fn in r["files"]:
            try:
                if (fn.lower(), os.stat(os.path.join(r["folder"], fn)).st_size) in prot_keys:
                    return True
            except OSError:
                pass
        return False

    # species per group -> per shot
    allgroups = groups_of([r for r in S.values() if "error" not in r and r["birds"]])
    shot_sp, group_of = {}, {}
    for gi, g in enumerate(allgroups):
        rep = next((sp[(r["folder"], r["stem"])] for r in g if (r["folder"], r["stem"]) in sp), None)
        for r in g:
            group_of[(r["folder"], r["stem"])] = gi
            shot_sp[(r["folder"], r["stem"])] = rep
    # days seen per confident species
    days = collections.defaultdict(set)
    for gi, g in enumerate(allgroups):
        rep = shot_sp[(g[0]["folder"], g[0]["stem"])]
        if rep and rep["cand"] and rep["cand"][0][1] >= CONFIDENT and rep.get("time"):
            days[rep["cand"][0][0]].add(datetime.fromtimestamp(rep["time"]).strftime("%Y-%m-%d"))

    def rare_info(r):
        rep = shot_sp.get((r["folder"], r["stem"]))
        if rep is None:
            return True, "no species ID"
        if not rep["cand"] or rep["cand"][0][1] < CONFIDENT:
            return True, "species uncertain"
        n = len(days[rep["cand"][0][0]])
        return n <= RARE_DAYS, f"{rep['cand'][0][0]} seen {n} days"

    plan = {}   # (folder, stem) -> (action, reason)
    def put(r, act, why):
        plan[(r["folder"], r["stem"])] = (act, why)

    rescue = {}
    rp = HERE / "scans" / "rescue.jsonl"
    if rp.exists():
        for l in open(rp):
            d = json.loads(l)
            rescue[(d["folder"], d["stem"])] = d
    force = set(open(HERE / "force_delete.txt").read().split("\n")) if (HERE / "force_delete.txt").exists() else set()
    force.discard("")
    cand_shots = []
    for (f, s), r in S.items():
        if "error" in r:
            put(r, "keep", "protected: could not decode RAW")
        elif f in A:
            put(r, "keep", "protected: approved folder")
        elif any(x.lower().endswith((".xmp", ".dng")) for x in r["files"]):
            put(r, "keep", "protected: has .xmp/.dng edit")
        elif r["persons"]:
            put(r, "keep", "protected: person detected")
        elif cross_copy(r):
            put(r, "keep", "protected: same file exists in an approved folder")
        else:
            cand_shots.append(r)

    def usable(b):
        return b["conf"] >= MIN_CONF and b.get("acu") is not None and b.get("acun", 0) >= MIN_ACUN

    keepers = []
    for r in cand_shots:
        b = best(r)
        rare, rwhy = rare_info(r) if r["birds"] else (False, "")
        if not r["birds"]:
            rd = rescue.get((r["folder"], r["stem"]))
            if r.get("animals"):
                put(r, "keep", "protected: non-bird animal detected")
            elif rd is None or "dets" not in rd:
                put(r, "keep", "protected: no-bird verification missing")
            elif any((n == "bird" and c >= 0.05) or (n != "bird" and c >= 0.15) for n, c in rd["dets"]):
                put(r, "keep", "protected: possible bird/animal/person in sensitive 2nd pass")
            else:
                put(r, "culled", "no bird found")
            continue
        if not usable(b):
            put(r, "keep", "bird too small/uncertain to judge")
            continue
        key = f"{r['folder']}|{r['stem']}"
        if key in force:
            put(r, "to_delete", f"obvious blur (reviewed by eye) acu={b['acu']:.3f}")
            continue
        if False and b["acu"] < a.hard and key not in keep_eye:  # automatic hard cut disabled: only eyeballed force_delete.txt goes to _to_delete
            # rare species: only if really blurred (extra margin)
            if not rare or b["acu"] < a.hard * 0.9:
                put(r, "to_delete", f"obvious blur acu={b['acu']:.3f}" + (f" (rare? {rwhy}, but truly blurred)" if rare else ""))
                continue
        if b["acu"] < a.soft:
            put(r, "culled" if not rare or b["acu"] < a.soft * 0.9 else "keep",
                f"soft acu={b['acu']:.3f}" if not rare or b["acu"] < a.soft * 0.9 else f"kept: soft but rare ({rwhy})")
            continue
        keepers.append((r, rare, rwhy))

    # burst culling among remaining good shots
    gid = collections.defaultdict(list)
    for r, rare, rwhy in keepers:
        gid[group_of.get((r["folder"], r["stem"]), id(r))].append((r, rare, rwhy))
    for g, mem in gid.items():
        # protected shots in the same burst count as kept
        allg = [x for x in allgroups[g]] if isinstance(g, int) else []
        nprot = sum(1 for x in allg if plan.get((x["folder"], x["stem"]), ("", ""))[0] == "keep"
                    and plan[(x["folder"], x["stem"])][1].startswith("protected"))
        k = max(1, KEEP - nprot)
        if len(mem) <= k:
            for r, _, _ in mem:
                put(r, "keep", "ok")
            continue
        if any(rare for _, rare, _ in mem):
            for r, _, rwhy in mem:
                put(r, "keep", f"burst kept: rare/uncertain species ({rwhy})")
            continue
        def score(r):
            b = best(r)
            x1, y1, x2, y2 = b["box"]
            edge = x1 <= 2 or y1 <= 2 or x2 >= r["W"] - 2 or y2 >= r["H"] - 2
            return b["acu"] * (0.85 if edge else 1.0) * (1 + min(b["area"], 0.1))
        mem.sort(key=lambda m: -score(m[0]))
        for i, (r, _, _) in enumerate(mem):
            if i < k:
                put(r, "keep", f"burst best {i+1}/{len(mem)}")
            else:
                put(r, "culled", f"burst extra ({len(mem)} frames)")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "stem", "files", "action", "reason"])
        for (f, s), (act, why) in sorted(plan.items()):
            w.writerow([f, s, "|".join(S[(f, s)]["files"]), act, why])
    c = collections.Counter(v[0] for v in plan.values())
    print(dict(c))
    c2 = collections.Counter((v[0], v[1].split(" acu")[0].split(" (")[0][:40]) for v in plan.values() if v[0] != "keep" or v[1].startswith("protected"))
    for k, v in sorted(c2.items()):
        print(k, v)
    # species table
    tab = collections.Counter()
    for gi, g in enumerate(allgroups):
        rep = shot_sp[(g[0]["folder"], g[0]["stem"])]
        if rep and rep["cand"]:
            tab[(rep["cand"][0][0], rep["cand"][0][2])] += 1
    json.dump({"days": {k: sorted(v) for k, v in days.items()},
               "tab": [[k[0], k[1], v] for k, v in tab.most_common()]},
              open(HERE / "scans" / "species_summary.json", "w"))


if __name__ == "__main__":
    main()
