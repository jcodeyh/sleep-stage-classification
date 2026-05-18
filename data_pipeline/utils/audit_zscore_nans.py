#!/usr/bin/env python3
"""Audit per-feature NaN rates after per-subject z-scoring.

Reproduces the cross_domain_eval.py pipeline up through zscore_per_subject and
reports, for each feature column and each of the four pools (mesa_train,
mesa_test, stages_train, stages_test):
  - pre-zscore NaN rate (raw missingness)
  - post-zscore NaN rate
  - count of subjects whose entire column is NaN (the trigger for all-NaN
    columns when many such subjects pile up in one pool)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cross_domain_eval import (  # noqa: E402
    DATA_PATH,
    extract_arrays,
    load_and_prepare,
    split_domain_subjects,
    zscore_per_subject,
)


def per_pool_nan_report(X, groups, feature_cols, label):
    n_rows = len(X)
    rates = np.isnan(X).mean(axis=0)
    df_g = pd.DataFrame(X)
    df_g["__g__"] = groups
    n_subj = df_g["__g__"].nunique()
    all_nan_subj_count = []
    for j in range(X.shape[1]):
        col = df_g.groupby("__g__")[j].apply(lambda s: s.isna().all())
        all_nan_subj_count.append(int(col.sum()))
    return pd.DataFrame({
        f"{label}_nan_rate": rates,
        f"{label}_subj_all_nan": all_nan_subj_count,
        f"{label}_n_subj": n_subj,
        f"{label}_n_rows": n_rows,
    }, index=feature_cols)


def main():
    df, feature_cols = load_and_prepare(DATA_PATH, add_site_onehot=False)
    mesa_mask = df["dataset"] == "mesa"
    stages_mask = (df["dataset"] == "stages") & (df["site_tier"] == 1)

    mesa_tr, mesa_te = split_domain_subjects(df, mesa_mask)
    stages_tr, stages_te = split_domain_subjects(df, stages_mask)

    pools = {
        "mesa_train":   extract_arrays(df[mesa_mask],   mesa_tr, feature_cols),
        "mesa_test":    extract_arrays(df[mesa_mask],   mesa_te, feature_cols),
        "stages_train": extract_arrays(df[stages_mask], stages_tr, feature_cols),
        "stages_test":  extract_arrays(df[stages_mask], stages_te, feature_cols),
    }

    print("\n=== PRE-ZSCORE NaN report ===")
    pre_reports = []
    for name, (X, _, g, _) in pools.items():
        pre_reports.append(per_pool_nan_report(X, g, feature_cols, name))
    pre = pd.concat(pre_reports, axis=1)

    print("\n=== POST-ZSCORE NaN report ===")
    post_reports = []
    for name, (X, _, g, _) in pools.items():
        Xz = zscore_per_subject(X, g, feature_cols, keep_absolute=False)
        post_reports.append(per_pool_nan_report(Xz, g, feature_cols, name))
    post = pd.concat(post_reports, axis=1)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    nan_cols = [c for c in pre.columns if c.endswith("_nan_rate")]
    print("\n--- Pre-zscore NaN fraction per pool ---")
    print(pre[nan_cols])

    print("\n--- Post-zscore NaN fraction per pool ---")
    print(post[nan_cols])

    print("\n--- Subjects entirely NaN for the column ---")
    subj_cols = [c for c in post.columns if c.endswith("_subj_all_nan")]
    print(post[subj_cols])

    fully_dead = post[(post[[c for c in post.columns if c.endswith("_nan_rate")]] >= 0.999).any(axis=1)]
    if len(fully_dead):
        print("\n!!! Columns ≥99.9% NaN in some pool (would trigger all-NaN nanmedian):")
        print(fully_dead[nan_cols])

    suspicious = post[(post[[c for c in post.columns if c.endswith("_nan_rate")]] >= 0.10).any(axis=1)]
    if len(suspicious):
        print("\n--- Columns with ≥10% NaN post-zscore in any pool (worth a closer look) ---")
        print(suspicious[nan_cols])


if __name__ == "__main__":
    main()
