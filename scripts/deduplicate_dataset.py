#!/usr/bin/env python3
"""
Removes duplicates (identical file bytes -- sha256) and prints reports of near duplicates so that we can make judgement calls of whether to remove it or not.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def _sha256(path_str: str) -> str:
    return hashlib.sha256(Path(path_str).read_bytes()).hexdigest()


def _phash_int(path_str: str) -> int:
    import imagehash
    from PIL import Image

    with Image.open(path_str) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def _map(fn, items):
    if not items:
        return []
    with ProcessPoolExecutor() as ex:
        return list(ex.map(fn, items, chunksize=256))


# --------------------------------------------------------------------------- #
# small data structures for near-dupe grouping
# --------------------------------------------------------------------------- #
class BKTree:
    """Metric tree over 64-bit hashes for fast "within Hamming distance d" queries."""

    def __init__(self) -> None:
        self.root: list | None = None  

    @staticmethod
    def _d(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, h: int) -> None:
        if self.root is None:
            self.root = [h, {}]
            return
        node = self.root
        while True:
            dist = self._d(h, node[0])
            nxt = node[1].get(dist)
            if nxt is None:
                node[1][dist] = [h, {}]
                return
            node = nxt

    def has_within(self, h: int, tol: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = self._d(h, node[0])
            if dist <= tol:
                return True
            for edge, child in node[1].items():
                if dist - tol <= edge <= dist + tol:
                    stack.append(child)
        return False

    def query(self, h: int, tol: int) -> list[int]:
        out: list[int] = []
        if self.root is None:
            return out
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = self._d(h, node[0])
            if dist <= tol:
                out.append(node[0])
            for edge, child in node[1].items():
                if dist - tol <= edge <= dist + tol:
                    stack.append(child)
        return out


class Union:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def discover_classes(split_dir: Path) -> dict[str, str]:
    """{raw_folder_name: lowercased_name} for every class sub-folder."""
    return {d.name: d.name.lower() for d in sorted(split_dir.iterdir()) if d.is_dir()}


def copy_all(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)
       
def clean(
    src: Path,
    dst: Path,
    *,
    seed: int,
    train_fraction: float,
    near_threshold: int | None,
    dedupe_test: bool,
) -> None:
    assert src.is_dir(), f"missing source: {src}"
    assert (src / "train").is_dir() and (src / "test").is_dir(), \
        f"{src} must contain train/ and test/"

    train_classes = discover_classes(src / "train")
    test_classes = discover_classes(src / "test")
    assert set(train_classes) == set(test_classes) and train_classes, \
        f"train classes {sorted(train_classes)} != test classes {sorted(test_classes)}"

    use_near = near_threshold is not None
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    log("=" * 66)
    log("DATASET CLEANING")
    log("=" * 66)
    log(f"src               : {src}")
    log(f"dst               : {dst}")
    log(f"seed              : {seed}")
    log(f"train fraction    : {train_fraction}")
    log(f"near-dupe removal  : "
        + (f"ON (pHash Hamming <= {near_threshold})" if use_near else "OFF (exact only)"))
    log(f"dedupe test set   : {dedupe_test}")
    log("")

    per_class_kept: dict[str, list[Path]] = {}
    test_out: dict[str, list[Path]] = {}
    totals = defaultdict(int)

    for raw_cls, cls in sorted(train_classes.items()):
        test_files = list_images(src / "test" / raw_cls)
        train_files = list_images(src / "train" / raw_cls)

        # hash everything (exact)
        test_sha = _map(_sha256, [str(p) for p in test_files])
        train_sha = _map(_sha256, [str(p) for p in train_files])
        test_sha_set = set(test_sha)

        # optional perceptual hashes 
        test_ph: list[int] = []
        train_ph: list[int] = []
        test_tree = BKTree()
        if use_near:
            test_ph = _map(_phash_int, [str(p) for p in test_files])
            train_ph = _map(_phash_int, [str(p) for p in train_files])
            for h in test_ph:
                test_tree.add(h)

        # drop train images that leak into test 
        drop_exact_leak: set[int] = set()
        drop_near_leak: set[int] = set()
        for i, sha in enumerate(train_sha):
            if sha in test_sha_set:
                drop_exact_leak.add(i)
            elif use_near and test_tree.has_within(train_ph[i], near_threshold):
                drop_near_leak.add(i)

        survivors = [i for i in range(len(train_files))
                     if i not in drop_exact_leak and i not in drop_near_leak]

        # collapse internal exact-duplicate groups
        seen_sha: set[str] = set()
        drop_internal_exact: set[int] = set()
        for i in survivors:
            if train_sha[i] in seen_sha:
                drop_internal_exact.add(i)
            else:
                seen_sha.add(train_sha[i])
        survivors = [i for i in survivors if i not in drop_internal_exact]

        # collapse internal near-duplicate clusters
        drop_internal_near: set[int] = set()
        if use_near:
            uf = Union()
            tree = BKTree()
            idx_by_hash: dict[int, list[int]] = defaultdict(list)
            for i in survivors:
                idx_by_hash[train_ph[i]].append(i)
            for h in idx_by_hash:
                tree.add(h)
            for h in idx_by_hash:
                for h2 in tree.query(h, near_threshold):
                    if h2 != h:
                        uf.union(h, h2)
            # keep the lowest original index in each cluster, drop the rest
            cluster_members: dict = defaultdict(list)
            for i in survivors:
                cluster_members[uf.find(train_ph[i])].append(i)
            for members in cluster_members.values():
                for i in sorted(members)[1:]:
                    drop_internal_near.add(i)
            survivors = [i for i in survivors if i not in drop_internal_near]

        kept = [train_files[i] for i in survivors]
        per_class_kept[cls] = kept

        # test set output 
        if dedupe_test:
            seen: set[str] = set()
            deduped = []
            n_test_drop = 0
            for p, sha in zip(test_files, test_sha):
                if sha in seen:
                    n_test_drop += 1
                else:
                    seen.add(sha)
                    deduped.append(p)
            test_out[cls] = deduped
        else:
            n_test_drop = 0
            test_out[cls] = test_files

        # report
        log(f"[{cls}]")
        log(f"  raw train pool                 : {len(train_files):>7}")
        log(f"  - exact duplicates of a test image : {len(drop_exact_leak):>7}")
        if use_near:
            log(f"  - near duplicates of a test image  : {len(drop_near_leak):>7}")
        log(f"  - internal exact-duplicate copies  : {len(drop_internal_exact):>7}")
        if use_near:
            log(f"  - internal near-duplicate copies   : {len(drop_internal_near):>7}")
        log(f"  = cleaned train pool           : {len(kept):>7}")
        log(f"  raw test set                   : {len(test_files):>7}"
            + (f"   (- {n_test_drop} internal dupes -> {len(test_out[cls])})" if dedupe_test else "   (kept as-is)"))
        log("")

        totals["raw_train"] += len(train_files)
        totals["exact_leak"] += len(drop_exact_leak)
        totals["near_leak"] += len(drop_near_leak)
        totals["internal_exact"] += len(drop_internal_exact)
        totals["internal_near"] += len(drop_internal_near)
        totals["kept"] += len(kept)

    # stratified train/val split of the cleaned pool
    rng = random.Random(seed)
    split_counts: dict[tuple[str, str], int] = {}
    for cls, kept in per_class_kept.items():
        files = list(kept)
        rng.shuffle(files)
        n_train = round(len(files) * train_fraction)
        train_files, val_files = files[:n_train], files[n_train:]
        copy_all(train_files, dst / "train" / cls)
        copy_all(val_files, dst / "val" / cls)
        copy_all(test_out[cls], dst / "test" / cls)
        split_counts[("train", cls)] = len(train_files)
        split_counts[("val", cls)] = len(val_files)
        split_counts[("test", cls)] = len(test_out[cls])

    classes = sorted(per_class_kept)
    log("=" * 66)
    log("FINAL DATASET  (" + str(dst) + ")")
    log("=" * 66)
    log(f"{'split':<8}{'class':<10}{'count':>10}")
    log("-" * 28)
    grand = 0
    for split in ("train", "val", "test"):
        sub = 0
        for cls in classes:
            c = split_counts[(split, cls)]
            sub += c
            grand += c
            log(f"{split:<8}{cls:<10}{c:>10}")
        log(f"{'':<8}{'(total)':<10}{sub:>10}")
    log("-" * 28)
    log(f"{'':<8}{'GRAND':<10}{grand:>10}")
    log("")
    removed = totals["raw_train"] - totals["kept"]
    log(f"removed {removed} images from the training pool "
        f"({totals['exact_leak']} test-leak exact, {totals['near_leak']} test-leak near, "
        f"{totals['internal_exact']} internal exact, {totals['internal_near']} internal near)")
    if len({split_counts[("train", c)] + split_counts[("val", c)] for c in classes}) > 1:
        log("NOTE: classes are no longer perfectly balanced after dedup "
            "(expected -- more duplicates were in one class). Use class weights when training.")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_cleaning_report.txt").write_text("\n".join(report_lines) + "\n")
    log(f"\nwrote {dst / '_cleaning_report.txt'}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, required=True,
                    help="raw download root containing train/ and test/")
    ap.add_argument("--dst", type=Path, default=Path("data/dataset_clean"),
                    help="output ImageFolder root (default: data/dataset_clean)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for the train/val split (default: 42)")
    ap.add_argument("--train-fraction", type=float, default=0.85,
                    help="fraction of the cleaned pool used for train, rest is val (default: 0.85)")
    ap.add_argument("--near-dupe-threshold", type=int, default=None, metavar="N",
                    help="also remove near-duplicates within pHash Hamming distance N "
                         "(omit = exact duplicates only; try 2-3 for 32x32 images)")
    ap.add_argument("--dedupe-test", action="store_true",
                    help="also collapse internal duplicate groups inside the test set "
                         "(default: test is copied unchanged)")
    args = ap.parse_args()

    clean(
        args.src,
        args.dst,
        seed=args.seed,
        train_fraction=args.train_fraction,
        near_threshold=args.near_dupe_threshold,
        dedupe_test=args.dedupe_test,
    )


if __name__ == "__main__":
    main()
