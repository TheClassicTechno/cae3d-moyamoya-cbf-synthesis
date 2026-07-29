#!/usr/bin/env python3
"""
Build combined dataset: existing pre/post (<repo-root>/pre, post) + 2020 pairs + 2024 pairs.
Subject-level split so no subject appears in both train and test.
Output: <repo-root>/combined_subject_split.json (same format as 2020_single_delay_split.json).
2024: discovered from moyamoya_2024_nifti (same path convention as 2020: derived/.../asl_single_delay_pre_diamox, post_diamox).

Standard for "more data" experiments: use --fix-test-set <reference_split.json> so the same 32 test
subjects are kept and new data (e.g. 2024) is added only to train/val. Enables apples-to-apples
comparison when measuring the effect of additional training data.
"""
import os
import sys
import json
import random
import argparse

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_REPO_ROOT, "pyproject.toml")):
    _parent = os.path.dirname(_REPO_ROOT)
    if _parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
    _REPO_ROOT = _parent

DATA_DIR = _REPO_ROOT
PRE_DIR = os.path.join(DATA_DIR, "pre")
POST_DIR = os.path.join(DATA_DIR, "post")
SPLIT_2020 = os.path.join(_REPO_ROOT, "2020_single_delay_split.json")
DATA_2024_ROOT = os.path.join(_REPO_ROOT, "moyamoya_2024_nifti")
CBF_PRE_SUBDIR_2024 = "derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox"
CBF_POST_SUBDIR_2024 = "derived/pre_surgery_yes_diamox/perf/asl_single_delay_post_diamox"
CBF_FILENAME = "CBF_Single_Delay_Pre_Diamox_standard_lin.nii.gz"
SEED = 42
TRAIN_FRAC = 0.75
VAL_FRAC = 0.125


def pre_to_post_path(pre_path):
    base = os.path.basename(pre_path).replace("pre_", "post_")
    return os.path.join(POST_DIR, base)


def subject_id_from_pre_path(pre_path):
    """pre/pre_2021_008.nii.gz -> 2021_008. pre_2022_032.nii.gz -> 2022_032."""
    base = os.path.basename(pre_path).replace(".nii.gz", "").replace("pre_", "")
    return base  # e.g. 2021_008, 2022_032


def subject_id_from_2020(sid):
    """moyamoya_stanford_2020_001 -> 2020_001."""
    if "moyamoya_stanford_2020_" in sid:
        return "2020_" + sid.split("_")[-1]
    return sid


def subject_id_from_2024(sid):
    """moyamoya_stanford_2024_001 -> 2024_001."""
    if "moyamoya_stanford_2024_" in sid:
        return "2024_" + sid.split("_")[-1]
    return sid


def discover_2024_pairs(data_root: str):
    """Discover pre/post pairs from moyamoya_2024_nifti (same layout as 2020). Returns list of (subject_id_short, pre_path, post_path)."""
    out = []
    if not os.path.isdir(data_root):
        return out
    for name in sorted(os.listdir(data_root)):
        if not name.startswith("moyamoya_stanford_2024_"):
            continue
        subj_dir = os.path.join(data_root, name)
        if not os.path.isdir(subj_dir):
            continue
        pre_path = os.path.join(subj_dir, CBF_PRE_SUBDIR_2024, CBF_FILENAME)
        post_path = os.path.join(subj_dir, CBF_POST_SUBDIR_2024, CBF_FILENAME)
        if os.path.isfile(pre_path) and os.path.isfile(post_path):
            sid_short = subject_id_from_2024(name)
            out.append((sid_short, pre_path, post_path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "combined_subject_split.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument(
        "--fix-test-set",
        default="",
        help="Path to reference split JSON. If set, use that file's test_subjects as the fixed "
        "hold-out test set; all other subjects go to train/val only (standard for 'more data' runs).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    pairs_by_subject = {}  # subject_id -> list of (pre_path, post_path)

    # Existing pre/post
    if os.path.isdir(PRE_DIR):
        for f in sorted(os.listdir(PRE_DIR)):
            if not f.startswith("pre_") or not f.endswith(".nii.gz"):
                continue
            pre_path = os.path.join(PRE_DIR, f)
            post_path = pre_to_post_path(pre_path)
            if not os.path.isfile(post_path):
                continue
            sid = subject_id_from_pre_path(pre_path)
            if sid not in pairs_by_subject:
                pairs_by_subject[sid] = []
            pairs_by_subject[sid].append((pre_path, post_path))

    # 2020
    if os.path.isfile(SPLIT_2020):
        with open(SPLIT_2020) as f:
            data = json.load(f)
        for part in ("train", "val", "test"):
            for item in data.get(part, []):
                sid = subject_id_from_2020(item["subject_id"])
                pre_path = item["pre_path"]
                post_path = item["post_path"]
                if os.path.isfile(pre_path) and os.path.isfile(post_path):
                    if sid not in pairs_by_subject:
                        pairs_by_subject[sid] = []
                    pairs_by_subject[sid].append((pre_path, post_path))

    # 2024 (discover from moyamoya_2024_nifti, same format as 2020)
    for sid, pre_path, post_path in discover_2024_pairs(DATA_2024_ROOT):
        if sid not in pairs_by_subject:
            pairs_by_subject[sid] = []
        pairs_by_subject[sid].append((pre_path, post_path))

    # One pair per subject (take first if multiple)
    subject_ids = sorted(pairs_by_subject.keys())
    pairs = []
    for sid in subject_ids:
        pre_path, post_path = pairs_by_subject[sid][0]
        pairs.append((sid, pre_path, post_path))

    n = len(pairs)
    fixed_test_ids = set()
    if args.fix_test_set and os.path.isfile(args.fix_test_set):
        with open(args.fix_test_set) as f:
            ref = json.load(f)
        fixed_test_ids = set(ref.get("test_subjects", []))
        # Only keep IDs that exist in current pool
        pool_ids = {p[0] for p in pairs}
        fixed_test_ids &= pool_ids

    if fixed_test_ids:
        # Fixed test set: same 32 (or fewer if some ref test subjects missing from pool) test subjects
        test_pairs = [p for p in pairs if p[0] in fixed_test_ids]
        remaining = [p for p in pairs if p[0] not in fixed_test_ids]
        random.shuffle(remaining)
        n_remain = len(remaining)
        n_val = int(n_remain * args.val_frac)
        n_train = n_remain - n_val
        train_pairs = remaining[:n_train]
        val_pairs = remaining[n_train:]
        if len(test_pairs) < len(fixed_test_ids):
            print("Note: %d reference test subjects not in current pool; test has %d subjects." % (len(fixed_test_ids) - len(test_pairs), len(test_pairs)))
    else:
        # Original behavior: random split
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        random.shuffle(pairs)
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

    out = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "train_subjects": [p[0] for p in train_pairs],
        "val_subjects": [p[0] for p in val_pairs],
        "test_subjects": [p[0] for p in test_pairs],
        "train": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in train_pairs],
        "val": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in val_pairs],
        "test": [{"subject_id": p[0], "pre_path": p[1], "post_path": p[2]} for p in test_pairs],
    }
    if fixed_test_ids:
        out["fixed_test_from"] = os.path.abspath(args.fix_test_set)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Combined: %d subjects, %d train / %d val / %d test. Written %s" % (n, len(train_pairs), len(val_pairs), len(test_pairs), args.out))
    if n == 0:
        print(
            "WARNING: 0 subjects found -- this split file is empty and unusable. "
            "Check that the raw data directories (pre/post, moyamoya_2020_nifti, moyamoya_2024_nifti) "
            "exist under the repository root; this script does not raise an error on missing data.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
