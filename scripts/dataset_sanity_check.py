#!/usr/bin/env python3
"""
Sanity-check reports:
  * image counts per split/class and class balance
  * unexpected file types / extension-vs-format mismatches
  * corrupted / unreadable images
  * exact-duplicate groups (flags cross-split leakage)
  * near-duplicate groups (with --near-dupes)
  * missing data (empty class dirs, splits missing a class)

Exit code: 0 = clean, 1 = warnings, 2 = failures.

All files accepted exited with clean (ie. 0)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile
import imagehash

# surface truncated-file warnings as errors so they count as corruption
warnings.simplefilter("error", Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
EXPECTED_SPLITS = ("train", "val", "test")
BALANCE_TOL = 0.10  # fraction deviation from an even class split before WARN

# per-file worker
def inspect(path_str: str, want_phash: bool):
    """Return dict describing one file. Runs in a worker process."""
    p = Path(path_str)
    out = {
        "path": path_str,
        "ext_ok": p.suffix.lower() in IMG_EXTS,
        "corrupt": False,
        "error": "",
        "fmt": "",
        "fmt_mismatch": False,
        "sha256": "",
        "phash": "",
    }
    try:
        data = p.read_bytes()
        out["sha256"] = hashlib.sha256(data).hexdigest()
    except OSError as e:
        out["corrupt"] = True
        out["error"] = f"read: {e}"
        return out

    try:
        with Image.open(p) as im:
            im.verify()  # structural check
        with Image.open(p) as im:
            im.load()  # force full decode
            rgb = im.convert("RGB")
            out["fmt"] = (im.format or "").upper()
            if want_phash:
                out["phash"] = str(imagehash.phash(rgb))
    except Exception as e:  # noqa: BLE001 - PIL raises many types
        out["corrupt"] = True
        out["error"] = f"decode: {type(e).__name__}: {e}"
        return out

    ext = p.suffix.lower().lstrip(".")
    ext_norm = {"jpg": "JPEG", "jpeg": "JPEG", "tif": "TIFF", "tiff": "TIFF"}.get(ext, ext.upper())
    if out["fmt"] and ext_norm != out["fmt"]:
        out["fmt_mismatch"] = True
    return out


# near-dupe grouping via BK-tree over perceptual hashes
class BKTree:
    def __init__(self):
        self.root = None 

    @staticmethod
    def _d(a, b):
        return bin(a ^ b).count("1")

    def add(self, h):
        if self.root is None:
            self.root = [h, {}]
            return
        node = self.root
        while True:
            dist = self._d(h, node[0])
            child = node[1].get(dist)
            if child is None:
                node[1][dist] = [h, {}]
                return
            node = child

    def query(self, h, tol):
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            dist = self._d(h, node[0])
            if dist <= tol:
                found.append(node[0])
            for edge, child in node[1].items():
                if dist - tol <= edge <= dist + tol:
                    stack.append(child)
        return found


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
         
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset root containing <split>/<class>/ dirs")
    ap.add_argument("--near-dupes", action="store_true", help="also detect near-duplicate images")
    ap.add_argument("--near-dupe-threshold", type=int, default=5,
                    help="max Hamming distance between pHashes to call a near-dupe (default 5)")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = auto)")
    ap.add_argument("--max-list", type=int, default=15, help="max example rows to print per issue")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"FATAL: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    splits = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not splits:
        print(f"FATAL: no split subdirectories under {root}", file=sys.stderr)
        sys.exit(2)

    files: list[tuple[str, str, str]] = []  
    non_image_files: list[str] = []
    classes_by_split: dict[str, set[str]] = defaultdict(set)
    empty_class_dirs: list[str] = []

    for split in splits:
        for cdir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            classes_by_split[split].add(cdir.name)
            n_here = 0
            for f in cdir.iterdir():
                if f.name == ".DS_Store" or f.name.startswith("."):
                    continue
                if not f.is_file():
                    continue
                if f.suffix.lower() in IMG_EXTS:
                    files.append((split, cdir.name, str(f)))
                    n_here += 1
                else:
                    non_image_files.append(str(f))
            if n_here == 0:
                empty_class_dirs.append(f"{split}/{cdir.name}")

    if not files:
        print("FATAL: found no image files at all", file=sys.stderr)
        sys.exit(2)

    print(f"Scanning {len(files):,} files under {root} ...")
    workers = args.workers or None
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(inspect, [f[2] for f in files],
                        [args.near_dupes] * len(files), chunksize=256):
            results.append(r)

    by_path = {r["path"]: r for r in results}

    # counts 
    counts: dict[tuple[str, str], int] = defaultdict(int)
    all_classes = set()
    for split, cls, _ in files:
        counts[(split, cls)] += 1
        all_classes.add(cls)
    all_classes = sorted(all_classes)

    warn, fail = [], []

    print("\n" + "=" * 60)
    print("IMAGE COUNTS")
    print("=" * 60)
    print(f"{'split':<10}{'class':<10}{'count':>12}")
    print("-" * 32)
    grand = 0
    for split in splits:
        sub = 0
        for cls in all_classes:
            c = counts.get((split, cls), 0)
            sub += c
            grand += c
            print(f"{split:<10}{cls:<10}{c:>12,}")
        print(f"{'':<10}{'TOTAL':<10}{sub:>12,}")
    print("-" * 32)
    print(f"{'':<10}{'GRAND':<10}{grand:>12,}")

    # cheking class balance
    print("\n" + "=" * 60)
    print("CLASS BALANCE")
    print("=" * 60)
    for split in splits:
        row = {cls: counts.get((split, cls), 0) for cls in all_classes}
        tot = sum(row.values())
        if tot == 0:
            fail.append(f"split '{split}' has zero images")
            continue
        frac = {cls: n / tot for cls, n in row.items()}
        even = 1 / len(all_classes)
        desc = "  ".join(f"{cls}={n:,} ({frac[cls]*100:.1f}%)" for cls, n in row.items())
        print(f"{split:<8} {desc}")
        worst = max(abs(f - even) for f in frac.values())
        if worst > BALANCE_TOL:
            warn.append(f"'{split}' class imbalance: {desc} (>{BALANCE_TOL*100:.0f}% from even)")

    # checking structural/missing
    print("\n" + "=" * 60)
    print("STRUCTURE")
    print("=" * 60)
    ref_classes = set(all_classes)
    for split in splits:
        missing = ref_classes - classes_by_split[split]
        if missing:
            fail.append(f"split '{split}' missing class dir(s): {sorted(missing)}")
    for d in empty_class_dirs:
        fail.append(f"empty class directory: {d}")
    for split in EXPECTED_SPLITS:
        if split not in splits:
            warn.append(f"expected split '{split}' not present")
    print(f"splits found : {splits}")
    print(f"classes found: {all_classes}")
    print(f"empty class dirs: {empty_class_dirs or 'none'}")

    # file types
    print("\n" + "=" * 60)
    print("FILE TYPES")
    print("=" * 60)
    fmt_counts = defaultdict(int)
    mismatches = []
    for r in results:
        if r["corrupt"]:
            continue
        fmt_counts[r["fmt"] or "?"] += 1
        if r["fmt_mismatch"]:
            mismatches.append(r["path"])
    for fmt, n in sorted(fmt_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:<8} {n:>10,}")
    if non_image_files:
        warn.append(f"{len(non_image_files)} non-image file(s) present, e.g. "
                    + ", ".join(Path(p).name for p in non_image_files[:5]))
        print(f"  non-image files: {len(non_image_files)}")
    if mismatches:
        warn.append(f"{len(mismatches)} file(s) whose extension disagrees with actual format")
        print(f"  extension/format mismatches: {len(mismatches)}")
        for p in mismatches[: args.max_list]:
            rel = Path(p).relative_to(root)
            print(f"    {rel}  -> actual {by_path[p]['fmt']}")

    # corruption 
    print("\n" + "=" * 60)
    print("CORRUPTED / UNREADABLE")
    print("=" * 60)
    corrupt = [r for r in results if r["corrupt"]]
    if not corrupt:
        print("  none ✓")
    else:
        fail.append(f"{len(corrupt)} corrupted/unreadable image(s)")
        for r in corrupt[: args.max_list]:
            rel = Path(r["path"]).relative_to(root)
            print(f"  {rel}  --  {r['error']}")
        if len(corrupt) > args.max_list:
            print(f"  ... and {len(corrupt) - args.max_list} more")

    # exact duplicates 
    print("\n" + "=" * 60)
    print("EXACT DUPLICATES (identical file bytes)")
    print("=" * 60)
    by_hash = defaultdict(list)
    for r in results:
        if r["sha256"] and not r["corrupt"]:
            by_hash[r["sha256"]].append(r["path"])
    dup_groups = [g for g in by_hash.values() if len(g) > 1]
    dup_files = sum(len(g) - 1 for g in dup_groups)
    cross_split = 0
    if not dup_groups:
        print("  none ")
    else:
        for g in dup_groups:
            sps = {Path(p).relative_to(root).parts[0] for p in g}
            if len(sps) > 1:
                cross_split += 1
        print(f"  {len(dup_groups):,} duplicate group(s), {dup_files:,} redundant file(s)")
        if cross_split:
            print(f"  {cross_split:,} group(s) span MULTIPLE SPLITS (train/val/test leakage)")
        for g in sorted(dup_groups, key=len, reverse=True)[: args.max_list]:
            rels = [str(Path(p).relative_to(root)) for p in sorted(g)]
            print(f"  x{len(g)}: " + "  |  ".join(rels[:4]) + (" ..." if len(rels) > 4 else ""))
        warn.append(f"{len(dup_groups)} exact-duplicate group(s) ({dup_files} redundant files)")
        if cross_split:
            fail.append(f"{cross_split} duplicate group(s) leak across splits")

    # check near duplicates 
    if args.near_dupes:
        print("\n" + "=" * 60)
        print(f"NEAR DUPLICATES (pHash Hamming <= {args.near_dupe_threshold})")
        print("=" * 60)
        tol = args.near_dupe_threshold
        phash_to_paths = defaultdict(list)
        for r in results:
            if r["phash"] and not r["corrupt"]:
                phash_to_paths[int(str(r["phash"]), 16)].append(r["path"])

        tree = BKTree()
        for h in phash_to_paths:
            tree.add(h)
        uf = Union()
        for h in phash_to_paths:
            for h2 in tree.query(h, tol):
                if h2 != h:
                    uf.union(h, h2)

        clusters = defaultdict(list)
        for h, paths in phash_to_paths.items():
            clusters[uf.find(h)].extend(paths)
        near_groups = [g for g in clusters.values() if len(g) > 1]
        # subtract exact-dup redundancy already reported
        near_only = [g for g in near_groups
                     if len({by_path[p]["sha256"] for p in g}) > 1]
        near_files = sum(len(g) - 1 for g in near_groups)
        n_cross = 0
        for g in near_groups:
            if len({Path(p).relative_to(root).parts[0] for p in g}) > 1:
                n_cross += 1
        if not near_groups:
            print("  none ✓")
        else:
            print(f"  {len(near_groups):,} near-dupe cluster(s), {near_files:,} extra image(s)")
            print(f"  {len(near_only):,} cluster(s) are NOT already exact duplicates")
            if n_cross:
                print(f"  {n_cross:,} cluster(s) span multiple splits (leakage risk)")
            for g in sorted(near_groups, key=len, reverse=True)[: args.max_list]:
                rels = [str(Path(p).relative_to(root)) for p in sorted(g)]
                print(f"  x{len(g)}: " + "  |  ".join(rels[:4]) + (" ..." if len(rels) > 4 else ""))
            warn.append(f"{len(near_groups)} near-duplicate cluster(s) ({near_files} extra images)")
            if n_cross:
                fail.append(f"{n_cross} near-duplicate cluster(s) span splits")

    # printing verdict 
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if fail:
        print("FAIL:")
        for m in fail:
            print(f"  ✗ {m}")
    if warn:
        print("WARN:")
        for m in warn:
            print(f"  ! {m}")
    if not fail and not warn:
        print("PASS — no issues detected ✓")

    sys.exit(2 if fail else (1 if warn else 0))


if __name__ == "__main__":
    main()
