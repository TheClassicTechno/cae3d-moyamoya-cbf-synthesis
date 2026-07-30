#!/usr/bin/env python3
"""Print tab:kfold LaTeX rows from week11_kfold/*_kfold_summary.json (run after K-fold).

Always prints mean$\\pm$std when n_folds >= 2, including std=0.0 (identical folds — see aggregate warning).
"""
import json
import os
import sys

ROOT = os.environ.get("ROOT", "/data1/julih")
KF = os.path.join(ROOT, "week11_kfold")


def row(model_key, path):
    with open(path) as f:
        d = json.load(f)
    mae, ms = d["mae_mean"], d["mae_std"]
    ss, ss_s = d["ssim_mean"], d["ssim_std"]
    ps, ps_s = d["psnr_mean"], d["psnr_std"]
    n_folds = int(d.get("n_folds", 5))

    def fmt(a, s):
        if n_folds < 2:
            return "%.4f" % a
        return "%.4f$\\pm$%.4f" % (a, s)

    return r"%s & %s & %s & %s \\" % (
        model_key.replace("_", r"\_"),
        fmt(mae, ms),
        fmt(ss, ss_s),
        fmt(ps, ps_s),
    )


def main():
    for name in sys.argv[1:] or ["Hybrid_3D", "DDPM_3D", "FNO_3D"]:
        p = os.path.join(KF, name + "_kfold_summary.json")
        if not os.path.isfile(p):
            print("# missing %s" % p, file=sys.stderr)
            continue
        print(row(name, p))


if __name__ == "__main__":
    main()
