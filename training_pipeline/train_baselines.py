#!/usr/bin/env python3
"""
Train and evaluate sleep staging models on MESA clean wearable dataset.

Usage:
    python scripts/train_baselines.py --model logistic --n_classes 5
    python scripts/train_baselines.py --model xgboost --n_classes 5
    python scripts/train_baselines.py --model 1d-cnn --n_classes 5
    python scripts/train_baselines.py --model all --n_classes 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "combined" / "combined_clean_wearable_full.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

STAGE_MAP_5 = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # Wake, N1, N2, N3, REM
STAGE_MAP_3 = {0: 0, 1: 1, 2: 1, 3: 2, 5: 2}  # Wake, NREM, REM
STAGE_MAP_2 = {0: 0, 1: 1, 2: 1, 3: 1, 5: 1}  # Wake, Sleep

STAGE_NAMES = {
    5: ["Wake", "N1", "N2", "N3", "REM"],
    3: ["Wake", "NREM", "REM"],
    2: ["Wake", "Sleep"],
}

META_COLS = ["subject_id", "dataset", "site", "site_tier", "is_high_quality_site",
             "hr_source", "is_clinical_population", "epoch_idx", "clock_time", "sleep_stage", "valid_epoch"]

# Features that leak the label (derived from sleep_stage) — must be excluded
# night_position is fractional position within the night (0–1), correlates with stage distribution
LEAKY_COLS = ["epochs_since_wake", "sleep_pressure", "night_position"]

# PSG-only features not collectible by consumer wearables — excluded from training
PSG_ONLY_COLS = [
    # Body position (requires PSG positional sensor)
    "pos_dominant", "pos_supine", "pos_left", "pos_right", "pos_prone", "pos_upright",
    # Leg movements (requires PSG leg EMG)
    "leg_move_mean", "leg_move_std", "leg_pct_active",
    "leg_move_mean_roll5_mean", "leg_move_mean_roll5_std",
    "leg_move_mean_roll10_mean", "leg_move_mean_roll10_std",
    "leg_move_mean_delta",
    # Spectral HRV (requires clinical-grade ECG derivation)
    "hrv_lf_hf",
    # Respiration (requires PSG respiratory belt)
    "resp_rate",
]

# MESA Actiwatch features — valid for MESA subjects, NaN for STAGES (different device).
# Included in training; NaN imputed to median and treated as a weak signal for STAGES.
MESA_ONLY_COLS = [
    "actig_wake", "actig_white_light",
]

# (demo_is_male is now unified across MESA and STAGES — no unmapped columns remain)

# Physiological clipping ranges
CLIP_RANGES = {
    "hr_mean": (30, 200), "hr_std": (0, 60), "hr_min": (25, 200), "hr_max": (30, 220),
    "hrv_mean_nn": (200, 2000), "hrv_sdnn": (0, 500), "hrv_rmssd": (0, 500),
    "hrv_pnn50": (0, 1), "hrv_lf_hf": (0, 50), "resp_rate": (4, 40),
    "spo2_mean": (50, 100), "spo2_std": (0, 15), "spo2_min": (50, 100),
}

SEED = 42
N3_WEIGHT_BOOST = False  # Set from --n3-weight-boost flag


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(n_classes: int, id_col: str = "subject_id", site_filter: str = "all"):
    """Load parquet, filter usable rows, map stages, return X, y, groups."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Apply site quality tier filter
    if site_filter != "all" and "site_tier" in df.columns:
        max_tier = {"tier1": 1, "tier1_2": 2}[site_filter]
        before = len(df)
        df = df[df["site_tier"] <= max_tier].copy()
        print(f"  Site filter '{site_filter}': kept {len(df):,}/{before:,} epochs "
              f"(tier <= {max_tier}, {df[id_col].nunique()} subjects)")
    elif site_filter != "all":
        print(f"  WARNING: site_tier column not found, --site-filter ignored")

    # Filter to usable rows: valid sleep stage + HR present
    mask = df.sleep_stage.isin([0, 1, 2, 3, 5]) & df.hr_mean.notna()
    df = df[mask].copy()
    print(f"  Usable epochs: {len(df):,} from {df[id_col].nunique()} subjects")

    # Map sleep stages
    stage_map = {5: STAGE_MAP_5, 3: STAGE_MAP_3, 2: STAGE_MAP_2}[n_classes]
    df["label"] = df.sleep_stage.map(stage_map)

    # Feature columns (exclude meta and leaky features)
    # Also exclude the id_col itself if it isn't already in META_COLS (e.g. mesa_id on MESA-only data)
    exclude = set(META_COLS + ["label"] + LEAKY_COLS + PSG_ONLY_COLS + [id_col])
    feature_cols = [c for c in df.columns if c not in exclude]

    # Clip physiological outliers
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in feature_cols:
            df[col] = df[col].clip(lo, hi)

    # Extract arrays — use to_numpy(na_value=np.nan) to correctly handle
    # pandas nullable dtypes (Float64, Int64, boolean) which contain pd.NA
    df_features = df[feature_cols].copy()
    for c in df_features.columns:
        if hasattr(df_features[c], 'to_numpy'):
            df_features[c] = df_features[c].to_numpy(dtype=np.float64, na_value=np.nan)
    X = df_features.values.astype(np.float64)
    y = df["label"].values.astype(np.int32)
    groups = df[id_col].values

    # NaN imputation deferred to after train/val/test split (avoid leakage)

    # Class distribution
    print(f"  Class distribution:")
    names = STAGE_NAMES[n_classes]
    for i, name in enumerate(names):
        count = (y == i).sum()
        print(f"    {name}: {count:,} ({100 * count / len(y):.1f}%)")

    return X, y, groups, feature_cols


def split_data(X, y, groups):
    """Subject-level 70/15/15 split."""
    # First split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    # Second split: 50/50 on temp → 15% val, 15% test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx_rel, test_idx_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    n_train_subj = len(np.unique(groups[train_idx]))
    n_val_subj = len(np.unique(groups[val_idx]))
    n_test_subj = len(np.unique(groups[test_idx]))

    print(f"\n  Split (subject-level, seed={SEED}):")
    print(f"    Train: {len(X_train):,} epochs, {n_train_subj} subjects")
    print(f"    Val:   {len(X_val):,} epochs, {n_val_subj} subjects")
    print(f"    Test:  {len(X_test):,} epochs, {n_test_subj} subjects")

    # Verify no subject leakage
    train_subjs = set(groups[train_idx])
    val_subjs = set(groups[val_idx])
    test_subjs = set(groups[test_idx])
    assert train_subjs.isdisjoint(val_subjs), "Leakage: train/val overlap!"
    assert train_subjs.isdisjoint(test_subjs), "Leakage: train/test overlap!"
    assert val_subjs.isdisjoint(test_subjs), "Leakage: val/test overlap!"
    print("    No subject leakage confirmed.")

    groups_train = groups[train_idx]
    groups_val = groups[val_idx]
    groups_test = groups[test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test, groups_train, groups_val, groups_test


def impute_nan(X_train, X_val, X_test):
    """Impute NaN with column medians computed only on the training set."""
    col_medians = np.nanmedian(X_train, axis=0)
    total_imputed = 0
    for arr, name in [(X_train, "Train"), (X_val, "Val"), (X_test, "Test")]:
        nan_mask = np.isnan(arr)
        n_imputed = nan_mask.sum()
        total_imputed += n_imputed
        for j in range(arr.shape[1]):
            arr[nan_mask[:, j], j] = col_medians[j]
        print(f"    {name}: imputed {n_imputed:,} NaN values")
    print(f"  Total imputed: {total_imputed:,} NaN values (medians from train only)")
    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)


def compute_class_weights(y):
    """Inverse frequency class weights. If N3_WEIGHT_BOOST is True, multiply N3 (class 3) by 2x."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    w = dict(zip(classes, weights))
    if N3_WEIGHT_BOOST and 3 in w:
        w[3] *= 2.0
    return w


def compute_specificity(cm, n_classes):
    """Compute per-class specificity from confusion matrix."""
    spec = {}
    for i in range(n_classes):
        # True negatives: everything not in row i or col i, plus cm[i,i] cancels out
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec[i] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return spec


def compute_per_subject_kappa(y_true, y_pred, groups):
    """Compute Cohen's κ per subject. Returns dict with stats and distribution."""
    subject_kappas = []
    for subj in np.unique(groups):
        mask = groups == subj
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(np.unique(yt)) < 2:
            continue  # κ undefined with single class
        subject_kappas.append(cohen_kappa_score(yt, yp))

    arr = np.array(subject_kappas)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_subjects": len(arr),
    }


def compute_transition_accuracy(y_true, y_pred):
    """Accuracy only on epochs where the true stage differs from the previous epoch."""
    # Find transition points (where true label changes from previous)
    transitions = np.where(y_true[1:] != y_true[:-1])[0] + 1  # +1 because diff shifts
    if len(transitions) == 0:
        return {"accuracy": 0.0, "n_transitions": 0, "total_epochs": len(y_true)}
    trans_correct = (y_pred[transitions] == y_true[transitions]).sum()
    trans_acc = float(trans_correct / len(transitions))
    return {
        "accuracy": trans_acc,
        "n_transitions": int(len(transitions)),
        "n_correct": int(trans_correct),
        "total_epochs": int(len(y_true)),
        "transition_pct": float(100 * len(transitions) / len(y_true)),
    }


def compute_n3_duration_correlation(y_true, y_pred, groups, n_classes):
    """Per-subject predicted vs true N3 minutes. Returns Pearson r stats or None."""
    if n_classes != 5 or groups is None:
        return None
    n3_class = 3  # N3 index in STAGE_MAP_5
    subjects = np.unique(groups)
    if len(subjects) < 3:
        return None
    true_mins, pred_mins = [], []
    for subj in subjects:
        mask = groups == subj
        true_mins.append((y_true[mask] == n3_class).sum() * 0.5)  # 30s epochs
        pred_mins.append((y_pred[mask] == n3_class).sum() * 0.5)
    r, p = pearsonr(true_mins, pred_mins)
    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "n_subjects": len(subjects),
        "true_mean_min": float(np.mean(true_mins)),
        "pred_mean_min": float(np.mean(pred_mins)),
    }


def evaluate(model_name, y_true, y_pred, n_classes, elapsed, groups=None, y_prob=None):
    """Compute and print all evaluation metrics.

    Args:
        y_prob: class probabilities (n_samples, n_classes) for AUROC. Optional.
        groups: subject IDs for per-subject κ. Optional.
    """
    names = STAGE_NAMES[n_classes]
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True)

    # Specificity per class
    specificity = compute_specificity(cm, n_classes)

    # AUROC per class (one-vs-rest)
    auroc = {}
    if y_prob is not None and n_classes > 2:
        try:
            auroc_scores = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None,
                                          labels=list(range(n_classes)))
            for i, name in enumerate(names):
                auroc[name] = float(auroc_scores[i])
            auroc["macro"] = float(np.mean(auroc_scores))
        except ValueError:
            pass  # can fail if a class has no samples
    elif y_prob is not None and n_classes == 2:
        try:
            auroc["macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            auroc[names[0]] = auroc["macro"]
            auroc[names[1]] = auroc["macro"]
        except ValueError:
            pass

    # Per-subject κ
    subject_kappa = None
    if groups is not None:
        subject_kappa = compute_per_subject_kappa(y_true, y_pred, groups)

    # Transition accuracy
    transition = compute_transition_accuracy(y_true, y_pred)

    # ── Print Results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — {n_classes}-class Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Cohen's κ:    {kappa:.4f}")
    print(f"  MCC:          {mcc:.4f}")
    print(f"  Train time:   {elapsed:.1f}s")

    if auroc:
        print(f"  AUROC (macro): {auroc.get('macro', 0):.4f}")

    # N3-specific top-level metrics (5-class only)
    n3_sensitivity = None
    n3_f1 = None
    n3_duration = None
    if n_classes == 5:
        n3_sensitivity = report["N3"]["recall"]
        n3_f1 = report["N3"]["f1-score"]
        print(f"  N3 Sensitivity: {n3_sensitivity:.4f}")
        print(f"  N3 F1:          {n3_f1:.4f}")
        n3_duration = compute_n3_duration_correlation(y_true, y_pred, groups, n_classes)
        if n3_duration:
            print(f"  N3 Duration r:  {n3_duration['pearson_r']:.4f} "
                  f"(p={n3_duration['p_value']:.2e}, "
                  f"true={n3_duration['true_mean_min']:.1f} min, "
                  f"pred={n3_duration['pred_mean_min']:.1f} min)")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'F1':>8}", end="")
    if auroc:
        print(f" {'AUROC':>8}", end="")
    print(f" {'Support':>10}")
    print(f"  {'-' * (52 + (8 if auroc else 0))}")
    for i, name in enumerate(names):
        r = report[name]
        print(f"  {name:>8} {r['precision']:>8.3f} {r['recall']:>8.3f} {specificity[i]:>8.3f} {r['f1-score']:>8.3f}", end="")
        if auroc:
            print(f" {auroc.get(name, 0):>8.3f}", end="")
        print(f" {r['support']:>10,.0f}")

    print(f"\n  Confusion Matrix:")
    header = "  " + " " * 10 + "".join(f"{n:>8}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"  {name:>10}" + "".join(f"{cm[i, j]:>8,}" for j in range(len(names)))
        print(row)

    # Per-subject κ
    if subject_kappa:
        print(f"\n  Per-subject Cohen's κ ({subject_kappa['n_subjects']} subjects):")
        print(f"    Median: {subject_kappa['median']:.4f}")
        print(f"    IQR:    [{subject_kappa['q25']:.4f}, {subject_kappa['q75']:.4f}]")
        print(f"    Range:  [{subject_kappa['min']:.4f}, {subject_kappa['max']:.4f}]")

    # Transition accuracy
    print(f"\n  Transition Accuracy:")
    print(f"    {transition['accuracy']:.4f} ({transition.get('n_correct', 0):,}/{transition['n_transitions']:,} transitions correct)")
    print(f"    Transitions are {transition['transition_pct']:.1f}% of all epochs")

    # Macro and weighted averages
    macro = report["macro avg"]
    weighted = report["weighted avg"]

    results = {
        "model": model_name,
        "n_classes": n_classes,
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "mcc": float(mcc),
        "macro_f1": float(macro["f1-score"]),
        "weighted_f1": float(weighted["f1-score"]),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "train_time_sec": float(elapsed),
        "n_test_samples": int(len(y_true)),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "transition_accuracy": transition,
    }

    # Per-class details with specificity and AUROC
    for i, name in enumerate(names):
        cls_metrics = {k: float(v) for k, v in report[name].items()}
        cls_metrics["specificity"] = specificity[i]
        if name in auroc:
            cls_metrics["auroc"] = auroc[name]
        results["per_class"][name] = cls_metrics

    if auroc:
        results["auroc_macro"] = auroc.get("macro", 0)

    if subject_kappa:
        results["per_subject_kappa"] = subject_kappa

    if n3_sensitivity is not None:
        results["n3_sensitivity"] = float(n3_sensitivity)
        results["n3_f1"] = float(n3_f1)
    if n3_duration:
        results["n3_duration_correlation"] = n3_duration

    return results


def save_results(results, model, scaler, model_name, n_classes, feature_cols):
    """Save model, scaler, metrics, feature importance, and append to experiment log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{model_name}_{n_classes}class_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata to results
    results["timestamp"] = timestamp
    results["data_path"] = str(DATA_PATH)
    results["n_features"] = len(feature_cols)
    results["seed"] = SEED
    results["run_dir"] = str(out_dir)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save scaler
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # Save feature names
    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_cols, f)

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])
        with open(out_dir / "feature_importance.json", "w") as f:
            json.dump(fi, f, indent=2)
        print(f"\n  Top 10 features:")
        for name, imp in fi[:10]:
            print(f"    {name:>25}: {imp:.4f}")
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).mean(axis=0)
        fi = sorted(zip(feature_cols, coefs.tolist()), key=lambda x: -x[1])
        with open(out_dir / "feature_importance.json", "w") as f:
            json.dump(fi, f, indent=2)
        print(f"\n  Top 10 features (mean |coef|):")
        for name, imp in fi[:10]:
            print(f"    {name:>25}: {imp:.4f}")

    # ── Append to experiment log CSV ──────────────────────────────────────
    log_path = RESULTS_DIR / "experiment_log.csv"
    names = STAGE_NAMES[n_classes]

    log_entry = {
        "timestamp": timestamp,
        "model": model_name,
        "n_classes": n_classes,
        "n_features": len(feature_cols),
        "data_file": Path(DATA_PATH).name,
        "accuracy": results["accuracy"],
        "cohen_kappa": results["cohen_kappa"],
        "mcc": results.get("mcc", ""),
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "auroc_macro": results.get("auroc_macro", ""),
        "transition_accuracy": results.get("transition_accuracy", {}).get("accuracy", ""),
        "per_subject_kappa_median": results.get("per_subject_kappa", {}).get("median", ""),
        "per_subject_kappa_iqr_lo": results.get("per_subject_kappa", {}).get("q25", ""),
        "per_subject_kappa_iqr_hi": results.get("per_subject_kappa", {}).get("q75", ""),
        "train_time_sec": results["train_time_sec"],
        "n_test_samples": results["n_test_samples"],
        "n3_sensitivity": results.get("n3_sensitivity", ""),
        "n3_f1": results.get("n3_f1", ""),
        "n3_duration_pearson_r": results.get("n3_duration_correlation", {}).get("pearson_r", ""),
        "n3_weight_boost": N3_WEIGHT_BOOST,
        "seed": SEED,
        "run_dir": str(out_dir.name),
    }
    # Per-class F1 columns
    for name in names:
        log_entry[f"f1_{name}"] = results["per_class"][name]["f1-score"]

    log_df = pd.DataFrame([log_entry])
    if log_path.exists():
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    print(f"\n  Results saved to: {out_dir}")
    print(f"  Appended to experiment log: {log_path}")
    return out_dir


# ── Model Training ────────────────────────────────────────────────────────────

def train_logistic(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, feature_cols,
                    groups_test=None):
    """Train logistic regression with class weights."""
    print("\n" + "=" * 60)
    print("  Training: Logistic Regression")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    weights = compute_class_weights(y_train)
    print(f"  Class weights: { {STAGE_NAMES[n_classes][k]: f'{v:.2f}' for k, v in weights.items()} }")

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weights,
        solver="lbfgs",
        random_state=SEED,
    )

    t0 = time.time()
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    results = evaluate("Logistic Regression", y_test, y_pred, n_classes, elapsed,
                        groups=groups_test, y_prob=y_prob)
    out_dir = save_results(results, model, scaler, "logistic", n_classes, feature_cols)
    return results, out_dir


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, feature_cols,
                   groups_test=None):
    """Train XGBoost with early stopping on validation set."""
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("  Training: XGBoost")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    weights = compute_class_weights(y_train)
    # XGBoost uses sample_weight, not class_weight
    sample_weights = np.array([weights[c] for c in y_train], dtype=np.float32)

    xgb_params = dict(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    if n_classes == 2:
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = n_classes
        xgb_params["eval_metric"] = "mlogloss"

    model = xgb.XGBClassifier(**xgb_params)

    t0 = time.time()
    model.fit(
        X_train_s, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val_s, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    results = evaluate("XGBoost", y_test, y_pred, n_classes, elapsed,
                        groups=groups_test, y_prob=y_prob)
    out_dir = save_results(results, model, scaler, "xgboost", n_classes, feature_cols)
    return results, out_dir


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, feature_cols,
                    groups_test=None):
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    print("\n" + "=" * 60)
    print("  Training: LightGBM")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    weights = compute_class_weights(y_train)
    sample_weights = np.array([weights[c] for c in y_train], dtype=np.float32)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=n_classes,
        objective="multiclass",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    t0 = time.time()
    model.fit(
        X_train_s, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val_s, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    results = evaluate("LightGBM", y_test, y_pred, n_classes, elapsed,
                        groups=groups_test, y_prob=y_prob)
    out_dir = save_results(results, model, scaler, "lightgbm", n_classes, feature_cols)
    return results, out_dir


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, feature_cols,
                         groups_test=None):
    """Train Random Forest."""
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "=" * 60)
    print("  Training: Random Forest")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    weights = compute_class_weights(y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight=weights,
        random_state=SEED,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    results = evaluate("Random Forest", y_test, y_pred, n_classes, elapsed,
                        groups=groups_test, y_prob=y_prob)
    out_dir = save_results(results, model, scaler, "random_forest", n_classes, feature_cols)
    return results, out_dir


def train_1d_cnn(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, feature_cols,
                  groups_test=None, groups_train=None, groups_val=None):
    """Train a lightweight 1D-CNN on sequences of epochs."""
    print("\n" + "=" * 60)
    print("  Training: 1D-CNN")
    print("=" * 60)

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  ERROR: TensorFlow not installed. Skipping 1D-CNN.")
        print("  Install with: pip install tensorflow")
        return None, None

    SEQ_LEN = 20  # 20 epochs = 10 minutes of context
    N_FEATURES = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def make_sequences(X, y, groups, seq_len):
        """Create overlapping sequences per subject — no cross-subject windows.

        For each subject's contiguous block of epochs:
          - If block >= seq_len: sliding window producing (block_len - seq_len + 1) seqs
          - If block < seq_len: left-pad with zeros, producing 1 sequence
        Label = last epoch in window.  Group = that subject's ID.
        Returns shuffled (X_seq, y_seq, groups_seq).
        """
        n_features = X.shape[1]
        # Find subject block boundaries
        boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(X)]])

        X_parts, y_parts, g_parts = [], [], []
        for s, e in zip(starts, ends):
            block_X = X[s:e]
            block_y = y[s:e]
            subj_id = groups[s]
            block_len = e - s

            if block_len >= seq_len:
                n_seq = block_len - seq_len + 1
                X_blk = np.zeros((n_seq, seq_len, n_features), dtype=np.float32)
                for i in range(n_seq):
                    X_blk[i] = block_X[i:i + seq_len]
                y_blk = block_y[seq_len - 1: seq_len - 1 + n_seq]
            else:
                # Left-pad with zeros
                X_blk = np.zeros((1, seq_len, n_features), dtype=np.float32)
                X_blk[0, seq_len - block_len:] = block_X
                y_blk = block_y[-1:]

            X_parts.append(X_blk)
            y_parts.append(y_blk.astype(np.int32))
            g_parts.append(np.full(len(y_blk), subj_id))

        X_seq = np.concatenate(X_parts)
        y_seq = np.concatenate(y_parts)
        groups_seq = np.concatenate(g_parts)

        # Shuffle so mini-batches aren't subject-grouped
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_seq))
        return X_seq[perm], y_seq[perm], groups_seq[perm]

    print(f"  Building per-subject sequences (length={SEQ_LEN})...")
    X_train_seq, y_train_seq, _ = make_sequences(X_train_s, y_train, groups_train, SEQ_LEN)
    X_val_seq, y_val_seq, _ = make_sequences(X_val_s, y_val, groups_val, SEQ_LEN)
    X_test_seq, y_test_seq, groups_test_seq = make_sequences(X_test_s, y_test, groups_test, SEQ_LEN)
    print(f"    Train: {X_train_seq.shape[0]:,} sequences")
    print(f"    Val:   {X_val_seq.shape[0]:,} sequences")
    print(f"    Test:  {X_test_seq.shape[0]:,} sequences")

    # Class weights
    weights = compute_class_weights(y_train_seq)
    print(f"  Class weights: { {STAGE_NAMES[n_classes][k]: f'{v:.2f}' for k, v in weights.items()} }")

    # Build lightweight 1D-CNN
    l2_reg = keras.regularizers.l2(1e-4)
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, N_FEATURES)),
        layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        layers.BatchNormalization(),

        layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.5),

        layers.Dense(64, activation="relu", kernel_regularizer=l2_reg),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax", kernel_regularizer=l2_reg),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    param_count = model.count_params()
    print(f"  Model parameters: {param_count:,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=200,
        batch_size=256,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    y_prob = model.predict(X_test_seq, batch_size=512)
    y_pred = y_prob.argmax(axis=1)
    results = evaluate("1D-CNN", y_test_seq, y_pred, n_classes, elapsed,
                        groups=groups_test_seq, y_prob=y_prob)
    results["n_params"] = int(param_count)
    results["epochs_trained"] = len(history.history["loss"])
    results["seq_len"] = SEQ_LEN

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"1d_cnn_{n_classes}class_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training history for convergence diagnostics
    hist = history.history
    # Convert numpy/float types to plain Python for JSON serialization
    history_data = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    # Save human-readable training summary
    epochs_trained = len(hist["loss"])
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    best_val_loss = float(min(hist["val_loss"]))
    stopped_early = epochs_trained < 200
    with open(out_dir / "training_summary.txt", "w") as f:
        f.write(f"Final epoch:      {epochs_trained} / 200\n")
        f.write(f"Best val_loss:    {best_val_loss:.6f} (epoch {best_epoch})\n")
        f.write(f"Stopping:         {'early stopping (patience=25)' if stopped_early else 'hit max epochs'}\n")
        f.write(f"Final train_loss: {float(hist['loss'][-1]):.6f}\n")
        f.write(f"Final val_loss:   {float(hist['val_loss'][-1]):.6f}\n")
        f.write(f"Final train_acc:  {float(hist['accuracy'][-1]):.6f}\n")
        f.write(f"Final val_acc:    {float(hist['val_accuracy'][-1]):.6f}\n")
        f.write(f"Final LR:         {float(hist['lr'][-1]):.2e}\n")

    model.save(out_dir / "model.keras")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    with open(out_dir / "features.json", "w") as f:
        json.dump(feature_cols, f)

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"  TFLite model: {len(tflite_model) / 1024:.1f} KB")

    print(f"\n  Results saved to: {out_dir}")
    return results, out_dir


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_results(all_results, n_classes):
    """Print comparison table of all trained models."""
    names = STAGE_NAMES[n_classes]
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':>20} {'Accuracy':>10} {'κ':>10} {'Time':>10}")
    print(f"  {'-' * 50}")
    for r in sorted(all_results, key=lambda x: -x["cohen_kappa"]):
        print(f"  {r['model']:>20} {r['accuracy']:>10.4f} {r['cohen_kappa']:>10.4f} {r['train_time_sec']:>8.1f}s")

    # Per-class F1 comparison
    print(f"\n  Per-class F1:")
    print(f"  {'Model':>20}" + "".join(f"{n:>8}" for n in names))
    print(f"  {'-' * (20 + 8 * len(names))}")
    for r in sorted(all_results, key=lambda x: -x["cohen_kappa"]):
        f1s = "".join(f"{r['per_class'][n]['f1-score']:>8.3f}" for n in names)
        print(f"  {r['model']:>20}{f1s}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sleep staging models")
    parser.add_argument("--model", required=True,
                        choices=["logistic", "xgboost", "lightgbm", "random_forest", "1d-cnn", "all"],
                        help="Model to train")
    parser.add_argument("--n_classes", type=int, default=5, choices=[2, 3, 5],
                        help="Number of sleep stage classes")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to parquet file (default: combined_clean_wearable_full.parquet). "
                             "Run scripts/merge_datasets.py first to generate the default file.")
    parser.add_argument("--id-col", type=str, default="subject_id",
                        help="Subject ID column name (default: subject_id). "
                             "Use mesa_id when training on MESA-only parquets.")
    parser.add_argument("--site-filter", type=str, default="all",
                        choices=["all", "tier1", "tier1_2"],
                        help="Filter by site quality tier: all (default), "
                             "tier1 (MESA+STNF+MSTR), tier1_2 (+ BOGN-ECG/MSQW/MSMI)")
    parser.add_argument("--n3-weight-boost", action="store_true", default=False,
                        help="Multiply N3 class weight by 2x on top of inverse-frequency weighting")
    args = parser.parse_args()

    # Set N3 weight boost flag
    global N3_WEIGHT_BOOST
    N3_WEIGHT_BOOST = args.n3_weight_boost
    if N3_WEIGHT_BOOST:
        print("  N3 weight boost: ON (2x multiplier on N3 class weight)")

    # Override data path if provided
    if args.data:
        global DATA_PATH
        DATA_PATH = Path(args.data)

    # Clear error if the data file doesn't exist
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        if "combined" in DATA_PATH.name:
            print("  Generate it by running: python scripts/merge_datasets.py")
            print("  Or pass a specific file with: --data path/to/your.parquet")
        sys.exit(1)

    # Load and split data (shared across all models)
    X, y, groups, feature_cols = load_data(args.n_classes, id_col=args.id_col,
                                           site_filter=args.site_filter)
    X_train, y_train, X_val, y_val, X_test, y_test, groups_train, groups_val, groups_test = split_data(X, y, groups)
    X_train, X_val, X_test = impute_nan(X_train, X_val, X_test)

    trainers = {
        "logistic": train_logistic,
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
        "random_forest": train_random_forest,
        "1d-cnn": train_1d_cnn,
    }

    all_results = []
    if args.model == "all":
        models_to_train = ["logistic", "xgboost", "lightgbm", "random_forest", "1d-cnn"]
    else:
        models_to_train = [args.model]

    for model_name in models_to_train:
        trainer = trainers[model_name]
        extra_kw = {}
        if model_name == "1d-cnn":
            extra_kw = {"groups_train": groups_train, "groups_val": groups_val}
        result, out_dir = trainer(X_train, y_train, X_val, y_val, X_test, y_test,
                                  args.n_classes, feature_cols, groups_test=groups_test,
                                  **extra_kw)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 1:
        compare_results(all_results, args.n_classes)

    print("\nDone!")


if __name__ == "__main__":
    main()
