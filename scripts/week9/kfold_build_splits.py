#!/usr/bin/env python3
"""
Build K-fold split definitions from combined_subject_split.json.

**Default (no --full):** train+val only; the official test list is fixed and never placed
in a fold. Used for nested sensitivity (same 32-subject test every fold); this is
*not* textbook K-fold because the evaluation test set does not rotate.

**--full:** all subjects (train+val+test) participate in rotating folds; each fold has
train, val, and test disjoint subsets (textbook full-cohort K-fold). Used by
run_week11_kfold_full.sh / run_week11_true_kfold_all_models.sh.

Usage:
  python scripts/week9/kfold_build_splits.py --K 5 --out scripts/week9/kfold_splits.json
  python scripts/week9/kfold_build_splits.py --K 5 --full --out scripts/week9/kfold_splits_full.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week7_preprocess import load_combined_split, get_pre_post_pairs, COMBINED_SPLIT_PATH
from week7_data import _subject_id_from_path


def _subject_to_pairs(pairs):
    """Build dict subject_id -> (pre_path, post_path). One pair per subject."""
    out = {}
    for pre_path, post_path in pairs:
        sid = _subject_id_from_path(pre_path)
        if sid in out:
            continue
        out[sid] = (pre_path, post_path)
    return out


def _subject_to_pairs_from_data(data, split_key):
    """Build dict subject_id -> (pre_path, post_path) from data[split_key] using subject_id when present."""
    out = {}
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if not pre or not post:
            continue
        sid = item.get("subject_id") or _subject_id_from_path(pre)
        out[sid] = (pre, post)
    return out


def build_kfold_splits(K: int, seed: int, split_path: str, out_path: str) -> dict:
    """
    Build K-fold splits and write to out_path. Return the written dict.
    Raises ValueError if verification fails.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    data = load_combined_split() if split_path == COMBINED_SPLIT_PATH else _load_json(split_path)
    # Use subject_id from split items when present (consistent IDs across path styles)
    train_s2p = _subject_to_pairs_from_data(data, "train")
    val_s2p = _subject_to_pairs_from_data(data, "val")
    subject_to_pairs = {**train_s2p, **val_s2p}
    all_subjects = list(subject_to_pairs.keys())
    test_subjects = list(data.get("test_subjects", [])) or list(_subject_to_pairs_from_data(data, "test").keys())

    # Verification: no test subject in train+val
    test_set = set(test_subjects)
    trainval_set = set(all_subjects)
    overlap = test_set & trainval_set
    if overlap:
        raise ValueError("Test set must be disjoint from train+val. Overlap: %s" % (list(overlap)[:5],))

    n_trainval = len(all_subjects)
    if n_trainval < K:
        raise ValueError("train+val has %d subjects; K=%d would leave empty folds" % (n_trainval, K))

    # Shuffle with fixed seed for reproducibility
    rng = __import__("random").Random(seed)
    shuffled = list(all_subjects)
    rng.shuffle(shuffled)

    # Split into K folds (fold i = chunk i as val)
    fold_size = n_trainval // K
    remainder = n_trainval % K
    folds = []
    start = 0
    for i in range(K):
        size = fold_size + (1 if i < remainder else 0)
        val_subjects = shuffled[start : start + size]
        train_subjects = [s for s in shuffled if s not in val_subjects]
        start += size
        folds.append({"fold": i, "train_subjects": train_subjects, "val_subjects": val_subjects})

    out = {
        "K": K,
        "seed": seed,
        "n_trainval": n_trainval,
        "n_test": len(test_subjects),
        "test_subjects": test_subjects,
        "folds": folds,
        "subject_to_pairs_source": "train+val from combined split; pairs resolved by subject_id",
    }

    # Verification
    all_val = set()
    for f in folds:
        t_set = set(f["train_subjects"])
        v_set = set(f["val_subjects"])
        if t_set & v_set:
            raise ValueError("Fold %d: train and val must be disjoint" % f["fold"])
        if t_set | v_set != trainval_set:
            raise ValueError("Fold %d: train U val must equal full train+val" % f["fold"])
        all_val |= v_set
    if all_val != trainval_set:
        raise ValueError("Union of val_subjects across folds must equal train+val")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def build_full_kfold_splits(K, seed, split_path, out_path):
    """Full K-fold: all 252 subjects in K folds. Fold i: test=fold i, val=fold (i+1) mod K, train=rest."""
    if K < 3:
        raise ValueError("Full K-fold needs K>=3")
    data = load_combined_split() if split_path == COMBINED_SPLIT_PATH else _load_json(split_path)
    train_s2p = _subject_to_pairs_from_data(data, "train")
    val_s2p = _subject_to_pairs_from_data(data, "val")
    test_s2p = _subject_to_pairs_from_data(data, "test")
    subject_to_pairs = {**train_s2p, **val_s2p, **test_s2p}
    all_subjects = list(subject_to_pairs.keys())
    n_all = len(all_subjects)
    if n_all < K:
        raise ValueError("Total subjects %d < K=%d" % (n_all, K))
    rng = __import__("random").Random(seed)
    shuffled = list(all_subjects)
    rng.shuffle(shuffled)
    fold_size = n_all // K
    remainder = n_all % K
    fold_subjects = []
    start = 0
    for i in range(K):
        size = fold_size + (1 if i < remainder else 0)
        fold_subjects.append(shuffled[start : start + size])
        start += size
    folds = []
    for i in range(K):
        test_subjects = fold_subjects[i]
        val_subjects = fold_subjects[(i + 1) % K]
        train_subjects = []
        for j in range(K):
            if j != i and j != (i + 1) % K:
                train_subjects.extend(fold_subjects[j])
        folds.append({"fold": i, "train_subjects": train_subjects, "val_subjects": val_subjects, "test_subjects": test_subjects})
    out = {"K": K, "seed": seed, "n_total": n_all, "full_kfold": True, "folds": folds, "subject_to_pairs_source": "train+val+test"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _pairs_from_data(data, split_key):
    pairs = []
    for item in data.get(split_key, []):
        pre = item.get("pre_path")
        post = item.get("post_path")
        if pre and post and os.path.isfile(pre) and os.path.isfile(post):
            pairs.append((pre, post))
    return pairs


def get_train_val_pairs_for_fold(kfold_path: str, fold_index: int, split_path: str = None):
    """
    Load kfold_splits.json and return (train_pairs, val_pairs) for the given fold.
    Pairs are (pre_path, post_path) lists. split_path: combined split JSON (default COMBINED_SPLIT_PATH).
    """
    with open(kfold_path) as f:
        kfold = json.load(f)
    split_path = split_path or COMBINED_SPLIT_PATH
    data = load_combined_split() if split_path == COMBINED_SPLIT_PATH else _load_json(split_path)
    train_s2p = _subject_to_pairs_from_data(data, "train")
    val_s2p = _subject_to_pairs_from_data(data, "val")
    subject_to_pairs = {**train_s2p, **val_s2p}

    folds = kfold["folds"]
    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError("fold_index %d out of range [0, %d)" % (fold_index, len(folds)))
    f = folds[fold_index]
    train_pairs = [subject_to_pairs[s] for s in f["train_subjects"] if s in subject_to_pairs]
    val_pairs = [subject_to_pairs[s] for s in f["val_subjects"] if s in subject_to_pairs]
    return train_pairs, val_pairs


def main():
    ap = argparse.ArgumentParser(description="Build K-fold splits (fixed test or full)")
    ap.add_argument("--K", type=int, default=5, help="Number of folds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment")
    ap.add_argument("--split", default=COMBINED_SPLIT_PATH, help="Path to combined_subject_split.json")
    ap.add_argument("--out", default="", help="Output path for kfold_splits.json")
    ap.add_argument("--full", action="store_true", help="Full K-fold: all 252 in folds, no fixed test")
    args = ap.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = args.out or (os.path.join(script_dir, "kfold_splits_full.json") if args.full else os.path.join(script_dir, "kfold_splits.json"))
    if args.full:
        result = build_full_kfold_splits(args.K, args.seed, args.split, out)
        print("Wrote %s (full K-fold K=%d, n_total=%d)" % (out, result["K"], result["n_total"]))
    else:
        result = build_kfold_splits(args.K, args.seed, args.split, out)
        print("Wrote %s (K=%d, n_trainval=%d, n_test=%d)" % (out, result["K"], result["n_trainval"], result["n_test"]))


if __name__ == "__main__":
    main()
