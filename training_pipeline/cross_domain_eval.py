#!/usr/bin/env python3
"""
Cross-domain evaluation: train on one domain, test on another.

Evaluates generalization across MESA and STAGES (tier1: STNF+MSTR) using
LightGBM and 1D-CNN. Fixed held-out test sets per domain ensure fair
comparison across all 4 conditions.

Usage:
    python scripts/cross_domain_eval.py
    python scripts/cross_domain_eval.py --models lightgbm
    python scripts/cross_domain_eval.py --models 1d-cnn
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
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "combined" / "combined_clean_wearable_full.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_domain"

STAGE_MAP_5 = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # Wake, N1, N2, N3, REM
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES = 5
N3_CLASS = 3  # index in mapped labels

META_COLS = ["subject_id", "dataset", "site", "site_tier", "is_high_quality_site",
             "is_clinical_population", "epoch_idx", "clock_time", "sleep_stage", "valid_epoch",
             "hr_source"]

LEAKY_COLS = ["epochs_since_wake", "sleep_pressure", "night_position"]

PSG_ONLY_COLS = [
    "pos_dominant", "pos_supine", "pos_left", "pos_right", "pos_prone", "pos_upright",
    "leg_move_mean", "leg_move_std", "leg_pct_active",
    "leg_move_mean_roll5_mean", "leg_move_mean_roll5_std",
    "leg_move_mean_roll10_mean", "leg_move_mean_roll10_std",
    "leg_move_mean_delta",
    "hrv_lf_hf", "resp_rate",
]

CLIP_RANGES = {
    "hr_mean": (30, 200), "hr_std": (0, 60), "hr_min": (25, 200), "hr_max": (30, 220),
    "hrv_mean_nn": (200, 2000), "hrv_sdnn": (0, 500), "hrv_rmssd": (0, 500),
    "hrv_pnn50": (0, 1), "hrv_lf_hf": (0, 50), "resp_rate": (4, 40),
    "spo2_mean": (50, 100), "spo2_std": (0, 15), "spo2_min": (50, 100),
}

SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(y):
    """Inverse frequency class weights."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return dict(zip(classes, weights))


def impute_nan_splits(*arrays):
    """Impute NaN/±inf with column medians from the first array (train).

    Falls back to 0 when a train column is entirely NaN (nanmedian returns NaN).
    After per-subject z-scoring, 0 equals the subject mean — the correct
    semantic fill in standardized coordinates. Without this guard, Keras
    models train with NaN loss while LightGBM tolerates the NaN silently.
    """
    train = arrays[0]
    col_medians = np.nanmedian(np.where(np.isinf(train), np.nan, train), axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    result = []
    for arr in arrays:
        arr = arr.copy()
        bad_mask = ~np.isfinite(arr)
        for j in range(arr.shape[1]):
            arr[bad_mask[:, j], j] = col_medians[j]
        result.append(arr.astype(np.float32))
    return result


ABSOLUTE_LEVEL_FEATURES = ("hr_min", "hr_max", "spo2_min")


def zscore_per_subject(X, groups, feature_cols, keep_absolute=False):
    """Z-score time-varying features within each subject.

    Skips constant-per-subject (demo_*, site_*) and cyclic (tod_cos/tod_sin)
    features. When keep_absolute=True, also skips hr_min/hr_max/spo2_min so
    the model retains clinically-meaningful absolute levels (e.g. desaturation
    nadir, resting HR floor) — motivated by B regression under full z-scoring.
    """
    skip_names = set()
    for c in feature_cols:
        if c.startswith("demo_") or c.startswith("site_") or c in ("tod_cos", "tod_sin"):
            skip_names.add(c)
    if keep_absolute:
        skip_names.update(ABSOLUTE_LEVEL_FEATURES)
    skip_idx = {i for i, c in enumerate(feature_cols) if c in skip_names}
    keep_idx = [i for i in range(X.shape[1]) if i not in skip_idx]
    if not keep_idx:
        return X

    df = pd.DataFrame(X[:, keep_idx], copy=False)
    df["__g__"] = groups
    g = df.groupby("__g__", sort=False)
    means = g[df.columns[:-1]].transform("mean").values
    stds = g[df.columns[:-1]].transform("std").values
    stds = np.where((stds < 1e-8) | np.isnan(stds), 1.0, stds)

    X_out = X.copy()
    X_out[:, keep_idx] = (X[:, keep_idx] - means) / stds
    return X_out


def fit_hmm_transitions(y, groups, n_classes=N_CLASSES):
    """Estimate stage transition matrix and initial distribution from training labels.

    Uses Laplace smoothing. Assumes rows are grouped by subject and temporally ordered.
    """
    T = np.ones((n_classes, n_classes))
    pi = np.ones(n_classes)
    boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(groups)]])
    for s, e in zip(starts, ends):
        seq = y[s:e]
        if len(seq) == 0:
            continue
        pi[seq[0]] += 1
        prev = seq[:-1]
        curr = seq[1:]
        np.add.at(T, (prev, curr), 1)
    T = T / T.sum(axis=1, keepdims=True)
    pi = pi / pi.sum()
    return T, pi


def viterbi_smooth(y_prob, groups, T, pi):
    """Per-subject Viterbi decoding. y_prob: (n, n_classes). Returns smoothed y_pred."""
    log_T = np.log(T + 1e-12)
    log_pi = np.log(pi + 1e-12)
    preds = np.empty(len(y_prob), dtype=np.int32)

    boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(groups)]])

    for s, e in zip(starts, ends):
        n = e - s
        if n == 0:
            continue
        log_emit = np.log(y_prob[s:e] + 1e-12)
        k = log_emit.shape[1]
        v = np.empty((n, k))
        bp = np.empty((n, k), dtype=np.int32)
        v[0] = log_pi + log_emit[0]
        for t in range(1, n):
            scores = v[t-1][:, None] + log_T  # (k_prev, k_curr)
            bp[t] = scores.argmax(axis=0)
            v[t] = scores[bp[t], np.arange(k)] + log_emit[t]
        path = np.empty(n, dtype=np.int32)
        path[-1] = v[-1].argmax()
        for t in range(n-2, -1, -1):
            path[t] = bp[t+1, path[t+1]]
        preds[s:e] = path
    return preds


def compute_metrics(y_true, y_pred, y_prob=None, groups=None):
    """Compute full metric suite. Returns dict."""
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    report = classification_report(y_true, y_pred, target_names=STAGE_NAMES,
                                   labels=list(range(N_CLASSES)), output_dict=True,
                                   zero_division=0)

    # N3 sensitivity and F1
    n3_sensitivity = report["N3"]["recall"]
    n3_f1 = report["N3"]["f1-score"]

    # AUROC
    auroc_macro = None
    if y_prob is not None:
        try:
            auroc_scores = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None,
                                         labels=list(range(N_CLASSES)))
            auroc_macro = float(np.mean(auroc_scores))
        except ValueError:
            pass

    # Per-class F1
    per_class_f1 = {name: float(report[name]["f1-score"]) for name in STAGE_NAMES}

    # N3 duration correlation
    n3_duration_r = None
    if groups is not None:
        subjects = np.unique(groups)
        true_mins, pred_mins = [], []
        for subj in subjects:
            mask = groups == subj
            true_mins.append((y_true[mask] == N3_CLASS).sum() * 0.5)
            pred_mins.append((y_pred[mask] == N3_CLASS).sum() * 0.5)
        if len(subjects) >= 3:
            r, p = pearsonr(true_mins, pred_mins)
            n3_duration_r = {"pearson_r": float(r), "p_value": float(p)}

    result = {
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "n3_sensitivity": float(n3_sensitivity),
        "n3_f1": float(n3_f1),
        "auroc_macro": auroc_macro,
        "per_class_f1": per_class_f1,
        "n_test_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }
    if n3_duration_r:
        result["n3_duration_correlation"] = n3_duration_r
    return result


def compute_per_site_kappa(y_true, y_pred, sites):
    """Compute Cohen's kappa per site. Returns dict {site: {kappa, n_epochs}}."""
    site_kappas = {}
    for site in np.unique(sites):
        mask = sites == site
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(np.unique(yt)) < 2:
            continue
        site_kappas[str(site)] = {
            "kappa": float(cohen_kappa_score(yt, yp)),
            "n_epochs": int(mask.sum()),
        }
    return site_kappas


# ── Data Loading & Splitting ─────────────────────────────────────────────────

def load_and_prepare(data_path, add_site_onehot=False):
    """Load parquet, filter, map stages, return prepared DataFrame."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Filter usable rows
    mask = df.sleep_stage.isin([0, 1, 2, 3, 5]) & df.hr_mean.notna()
    df = df[mask].copy()
    df["label"] = df.sleep_stage.map(STAGE_MAP_5)

    # Optional: one-hot encode site as features. Per-subject constant, so
    # zscore_per_subject skips any column starting with "site_".
    if add_site_onehot:
        site_dummies = pd.get_dummies(df["site"], prefix="site", dtype=np.float32)
        df = pd.concat([df, site_dummies], axis=1)
        print(f"  Added site one-hot columns: {list(site_dummies.columns)}")

    # Feature columns
    exclude = set(META_COLS + ["label"] + LEAKY_COLS + PSG_ONLY_COLS + ["subject_id"])
    feature_cols = [c for c in df.columns if c not in exclude]

    # Clip
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in feature_cols:
            df[col] = df[col].clip(lo, hi)

    # Convert nullable dtypes
    for c in feature_cols:
        if hasattr(df[c], 'to_numpy'):
            df[c] = df[c].to_numpy(dtype=np.float64, na_value=np.nan)

    print(f"  Total usable epochs: {len(df):,} from {df['subject_id'].nunique()} subjects")
    return df, feature_cols


def find_cross_domain_dead_features(df, domain_masks, feature_cols, nan_thresh=0.80):
    """Identify features with ≥nan_thresh NaN rate in any domain.

    These columns carry no cross-domain signal — typically dataset-specific
    demographics, race one-hots with non-overlapping categories, or sensors
    absent from one dataset (e.g. actig_white_light not on Huami Arc). Keeping
    them produces NaN train losses for Keras models and dataset-identity
    leakage in tree models. Returns list of (col, {domain: nan_rate}).
    """
    dead = []
    for c in feature_cols:
        rates = {name: float(df.loc[mask, c].isna().mean())
                 for name, mask in domain_masks.items()}
        if max(rates.values()) >= nan_thresh:
            dead.append((c, rates))
    return dead


def split_domain_subjects(df, domain_mask, seed=SEED):
    """80/20 subject-level split within a domain. Returns (train_subjects, test_subjects)."""
    domain_df = df[domain_mask]
    subjects = domain_df["subject_id"].unique()
    # Use GroupShuffleSplit for consistency
    X_dummy = np.zeros(len(subjects))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(X_dummy, groups=subjects))
    return set(subjects[train_idx]), set(subjects[test_idx])


def extract_arrays(df, subject_set, feature_cols):
    """Extract X, y, groups, sites arrays for a set of subjects."""
    mask = df["subject_id"].isin(subject_set)
    sub = df[mask]
    X = sub[feature_cols].values.astype(np.float64)
    y = sub["label"].values.astype(np.int32)
    groups = sub["subject_id"].values
    sites = sub["site"].values if "site" in sub.columns else None
    return X, y, groups, sites


# ── LightGBM ─────────────────────────────────────────────────────────────────

def _fit_lightgbm(X_train, y_train, row_weights=None):
    """Fit LGBM with internal 85/15 val split for early stopping.

    row_weights: optional per-row multiplier applied on top of inverse-frequency
    class weights (used for pseudo-labeled rows in self-training). Pass None or
    all-ones to disable.

    Returns (model, scaler, elapsed_sec).
    """
    import lightgbm as lgb

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    n = len(X_s)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    val_size = int(0.15 * n)
    val_idx = perm[:val_size]
    trn_idx = perm[val_size:]

    class_weights = compute_class_weights(y_train)
    sw_trn = np.array([class_weights[c] for c in y_train[trn_idx]], dtype=np.float32)
    if row_weights is not None:
        sw_trn = sw_trn * row_weights[trn_idx].astype(np.float32)

    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        num_class=N_CLASSES, objective="multiclass",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

    t0 = time.time()
    model.fit(
        X_s[trn_idx], y_train[trn_idx],
        sample_weight=sw_trn,
        eval_set=[(X_s[val_idx], y_train[val_idx])],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )
    return model, scaler, time.time() - t0


def train_eval_lightgbm(X_train, y_train, X_test, y_test, groups_test, sites_test):
    """Train LightGBM, evaluate on test."""
    model, scaler, elapsed = _fit_lightgbm(X_train, y_train)
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)

    metrics = compute_metrics(y_test, y_pred, y_prob=y_prob, groups=groups_test)
    metrics["train_time_sec"] = float(elapsed)
    if sites_test is not None:
        metrics["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred, sites_test)
    return metrics, y_pred, y_prob


# ── Self-training (target-side pseudo-labeling) ───────────────────────────────

def pseudo_label_target(model, scaler, X_tgt, g_tgt, T_hmm, pi_hmm, quantile):
    """HMM-decode target predictions and keep per-class top-quantile by confidence.

    Per-class top-K filtering is the lever against confirmation bias: each class
    contributes its own most-confident epochs even if the model globally
    undershoots that class. Returns (keep_mask, y_pseudo, kept_per_class).
    """
    X_s = scaler.transform(X_tgt)
    y_prob = model.predict_proba(X_s)
    y_pseudo = viterbi_smooth(y_prob, g_tgt, T_hmm, pi_hmm)
    conf = y_prob[np.arange(len(y_prob)), y_pseudo]

    keep_mask = np.zeros(len(y_pseudo), dtype=bool)
    kept_per_class = {}
    for c in range(N_CLASSES):
        cls_mask = y_pseudo == c
        if cls_mask.sum() == 0:
            kept_per_class[c] = 0
            continue
        thresh = np.quantile(conf[cls_mask], 1.0 - quantile)
        kept = cls_mask & (conf >= thresh)
        keep_mask |= kept
        kept_per_class[c] = int(kept.sum())
    return keep_mask, y_pseudo, kept_per_class


def self_train_eval_lightgbm(X_src, y_src, g_src, X_tgt, g_tgt,
                             X_test, y_test, groups_test, sites_test,
                             quantile, pseudo_weight, n_iters):
    """Self-training loop: source-gold + filtered target-pseudo at weight `pseudo_weight`.

    HMM transitions are fit ONCE from source labels and reused across iterations
    (never re-estimated from pseudo-labels — that would amplify confirmation bias).
    Returns (metrics, y_pred, y_prob, T_hmm, pi_hmm).
    """
    T_hmm, pi_hmm = fit_hmm_transitions(y_src, g_src)

    print(f"    [self-train] iter 0 (source-only): fitting initial LGBM on {len(X_src):,} gold epochs")
    model, scaler, elapsed = _fit_lightgbm(X_src, y_src)
    total_elapsed = elapsed

    for it in range(1, n_iters + 1):
        keep_mask, y_pseudo, kept_per_class = pseudo_label_target(
            model, scaler, X_tgt, g_tgt, T_hmm, pi_hmm, quantile)
        kept = int(keep_mask.sum())
        print(f"    [self-train] iter {it}: kept {kept:,}/{len(keep_mask):,} "
              f"pseudo-labels (quantile={quantile})")
        print(f"                 per-class kept: " +
              ", ".join(f"{STAGE_NAMES[c]}={kept_per_class[c]:,}" for c in range(N_CLASSES)))

        X_mix = np.concatenate([X_src, X_tgt[keep_mask]])
        y_mix = np.concatenate([y_src, y_pseudo[keep_mask]])
        w_mix = np.concatenate([np.ones(len(X_src), dtype=np.float32),
                                np.full(kept, pseudo_weight, dtype=np.float32)])
        model, scaler, elapsed = _fit_lightgbm(X_mix, y_mix, row_weights=w_mix)
        total_elapsed += elapsed

    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    metrics = compute_metrics(y_test, y_pred, y_prob=y_prob, groups=groups_test)
    metrics["train_time_sec"] = float(total_elapsed)
    if sites_test is not None:
        metrics["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred, sites_test)
    return metrics, y_pred, y_prob, T_hmm, pi_hmm


# ── 1D-CNN ────────────────────────────────────────────────────────────────────

def make_sequences(X, y, groups, seq_len):
    """Create per-subject sequences with sliding window. Same logic as train_baselines."""
    n_features = X.shape[1]
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
            X_blk = np.zeros((1, seq_len, n_features), dtype=np.float32)
            X_blk[0, seq_len - block_len:] = block_X
            y_blk = block_y[-1:]

        X_parts.append(X_blk)
        y_parts.append(y_blk.astype(np.int32))
        g_parts.append(np.full(len(y_blk), subj_id))

    X_seq = np.concatenate(X_parts)
    y_seq = np.concatenate(y_parts)
    groups_seq = np.concatenate(g_parts)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_seq))
    return X_seq[perm], y_seq[perm], groups_seq[perm]


def make_sequences_blocks(X, y, groups, seq_len, shuffle):
    """Tile each subject into non-overlapping blocks of length seq_len.

    Last block of each subject is zero-padded at the tail. Used by many-to-many
    sequence models (Bi-LSTM). Unlike make_sequences, preserves per-subject
    chronological ordering when shuffle=False so that downstream HMM smoothing
    can operate on predictions in temporal order.

    Returns
    -------
    X_seq: (N, seq_len, n_features)
    y_seq: (N, seq_len) — padded positions are 0 (masked by weight_seq)
    weight_seq: (N, seq_len) — 1 for real epochs, 0 for padding
    groups_seq: (N,) — subject id per block
    flat_order: (N * seq_len,) int indices — for unpadded, chronological
        re-assembly, use flat_order[weight_seq.reshape(-1) == 1] which yields
        a permutation of arange(len(X)) aligned to the original (X, y, groups)
        order if shuffle=False.
    """
    n = len(X)
    n_features = X.shape[1]
    boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])

    X_parts, y_parts, w_parts, g_parts, idx_parts = [], [], [], [], []
    for s, e in zip(starts, ends):
        block_len = e - s
        subj_id = groups[s]
        n_blocks = (block_len + seq_len - 1) // seq_len  # ceil
        for b in range(n_blocks):
            lo = s + b * seq_len
            hi = min(lo + seq_len, e)
            real = hi - lo
            Xb = np.zeros((seq_len, n_features), dtype=np.float32)
            yb = np.zeros(seq_len, dtype=np.int32)
            wb = np.zeros(seq_len, dtype=np.float32)
            ib = np.full(seq_len, -1, dtype=np.int64)
            Xb[:real] = X[lo:hi]
            yb[:real] = y[lo:hi]
            wb[:real] = 1.0
            ib[:real] = np.arange(lo, hi)
            X_parts.append(Xb)
            y_parts.append(yb)
            w_parts.append(wb)
            g_parts.append(subj_id)
            idx_parts.append(ib)

    X_seq = np.stack(X_parts)
    y_seq = np.stack(y_parts)
    weight_seq = np.stack(w_parts)
    groups_seq = np.array(g_parts)
    flat_order = np.stack(idx_parts).reshape(-1)

    if shuffle:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_seq))
        return (X_seq[perm], y_seq[perm], weight_seq[perm],
                groups_seq[perm], flat_order.reshape(len(X_seq), seq_len)[perm].reshape(-1))
    return X_seq, y_seq, weight_seq, groups_seq, flat_order


def tile_flat_to_blocks(flat_arr, groups, seq_len, shuffle, dtype=np.float32):
    """Tile a 1D flat array into (N_blocks, seq_len) using the same per-subject
    block layout as make_sequences_blocks. Same RandomState seed → same shuffle
    permutation, so a tiled aux array can be paired with the X/y output.

    Used to carry per-epoch metadata (e.g. source-vs-pseudo flag) through the
    sequence-tiling step alongside the main feature/label tensors.
    """
    n = len(flat_arr)
    boundaries = np.where(groups[1:] != groups[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    parts = []
    for s, e in zip(starts, ends):
        block_len = e - s
        n_blocks = (block_len + seq_len - 1) // seq_len
        for b in range(n_blocks):
            lo = s + b * seq_len
            hi = min(lo + seq_len, e)
            real = hi - lo
            blk = np.zeros(seq_len, dtype=dtype)
            blk[:real] = flat_arr[lo:hi]
            parts.append(blk)
    out = np.stack(parts)
    if shuffle:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(out))
        out = out[perm]
    return out


def train_eval_bi_lstm(X_train, y_train, groups_train, X_test, y_test, groups_test, sites_test,
                       X_target_train=None, y_target_train=None, groups_target_train=None,
                       target_val_frac=0.0, seq_len=120, lstm_units=128):
    """Train 2-layer Bi-LSTM, evaluate on test.

    Many-to-many: each timestep in a window gets a softmax prediction. Test-time
    predictions are re-assembled in original flat order so downstream HMM
    smoothing (on y_prob + groups_test) works identically to LGBM.

    target_val_frac > 0 with target_train args provided enables cross-domain
    early stopping: a held-out slice of TARGET-domain train subjects supplies
    val_loss instead of within-source val. Source models trained with this
    setting stop when *cross-domain* generalization plateaus rather than
    in-distribution loss — critical when source/target distributions diverge.
    Falls back to source-internal val when target_train is None or frac is 0.

    Returns (metrics, y_pred_flat, y_prob_flat) aligned to (y_test, groups_test).
    """
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  ERROR: TensorFlow not installed. Skipping Bi-LSTM.")
        return None

    SEQ_LEN = seq_len
    N_FEATURES = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Defense in depth: a zero-variance column can still emit NaN/±inf via
    # StandardScaler if upstream guards miss something. Keras has no NaN
    # tolerance, so clamp before training.
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    rng = np.random.RandomState(SEED)
    use_target_val = (target_val_frac > 0.0 and X_target_train is not None
                      and y_target_train is not None and groups_target_train is not None)

    if use_target_val:
        # All source-train goes into the gradient-producing pool; val comes
        # from a held-out slice of target-domain train subjects.
        target_subjects = np.unique(groups_target_train)
        rng.shuffle(target_subjects)
        n_val = max(1, int(target_val_frac * len(target_subjects)))
        val_target_subj = set(target_subjects[:n_val])
        val_mask = np.isin(groups_target_train, list(val_target_subj))

        X_target_s = scaler.transform(X_target_train)
        X_target_s = np.nan_to_num(X_target_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_val_flat = X_target_s[val_mask]
        y_val_flat = y_target_train[val_mask]
        g_val_flat = groups_target_train[val_mask]
        X_trn_flat = X_train_s
        y_trn_flat = y_train
        g_trn_flat = groups_train
        print(f"    Bi-LSTM cross-domain val: {val_mask.sum():,} target-val epochs "
              f"from {n_val} held-out target subjects ({target_val_frac:.0%})")
    else:
        # Within-source val (legacy behavior)
        train_subjects = np.unique(groups_train)
        rng.shuffle(train_subjects)
        val_n = max(1, int(0.15 * len(train_subjects)))
        val_subjects = set(train_subjects[:val_n])
        val_mask = np.isin(groups_train, list(val_subjects))
        trn_mask = ~val_mask
        X_trn_flat = X_train_s[trn_mask]
        y_trn_flat = y_train[trn_mask]
        g_trn_flat = groups_train[trn_mask]
        X_val_flat = X_train_s[val_mask]
        y_val_flat = y_train[val_mask]
        g_val_flat = groups_train[val_mask]
        print(f"    Bi-LSTM internal split: {trn_mask.sum():,} train, {val_mask.sum():,} val epochs")

    # Block-tile train/val (shuffled) and test (preserve order for HMM)
    X_trn_seq, y_trn_seq, w_trn_seq, _, _ = make_sequences_blocks(
        X_trn_flat, y_trn_flat, g_trn_flat, SEQ_LEN, shuffle=True)
    X_val_seq, y_val_seq, w_val_seq, _, _ = make_sequences_blocks(
        X_val_flat, y_val_flat, g_val_flat, SEQ_LEN, shuffle=True)
    X_te_seq, y_te_seq, w_te_seq, _, flat_order_te = make_sequences_blocks(
        X_test_s, y_test, groups_test, SEQ_LEN, shuffle=False)

    print(f"    Sequences: {X_trn_seq.shape[0]:,} train, {X_val_seq.shape[0]:,} val, "
          f"{X_te_seq.shape[0]:,} test blocks (seq_len={SEQ_LEN})")

    # Per-timestep sample weight: class weight × real-epoch mask
    cls_w_trn = compute_class_weights(y_trn_seq[w_trn_seq > 0])
    sw_trn = w_trn_seq * np.vectorize(lambda c: cls_w_trn.get(int(c), 1.0))(y_trn_seq)
    sw_val = w_val_seq * np.vectorize(lambda c: cls_w_trn.get(int(c), 1.0))(y_val_seq)
    sw_trn = sw_trn.astype(np.float32)
    sw_val = sw_val.astype(np.float32)

    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, N_FEATURES)),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Dense(max(64, lstm_units // 2), activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    t0 = time.time()
    model.fit(
        X_trn_seq, y_trn_seq,
        sample_weight=sw_trn,
        validation_data=(X_val_seq, y_val_seq, sw_val),
        epochs=120, batch_size=64,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.time() - t0

    # Predict in order, flatten, drop padding → aligned to (X_test, y_test, groups_test)
    y_prob_seq = model.predict(X_te_seq, batch_size=128)  # (N, T, C)
    y_prob_flat_padded = y_prob_seq.reshape(-1, N_CLASSES)
    mask_flat = w_te_seq.reshape(-1) > 0
    y_prob_raw = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)
    y_prob_raw[flat_order_te[mask_flat]] = y_prob_flat_padded[mask_flat]
    y_pred_raw = y_prob_raw.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred_raw, y_prob=y_prob_raw, groups=groups_test)
    metrics["train_time_sec"] = float(elapsed)
    if sites_test is not None:
        metrics["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred_raw, sites_test)
    return metrics, y_pred_raw, y_prob_raw


def self_train_eval_bi_lstm(X_src, y_src, g_src, X_tgt, y_tgt, g_tgt,
                            X_test, y_test, groups_test, sites_test,
                            target_val_frac=0.10, quantile=0.5,
                            pseudo_weight=0.5, n_iters=1, seq_len=60,
                            lstm_units=128):
    """Bi-LSTM self-training: source-gold + HMM-smoothed target pseudo-labels.

    Mirrors self_train_eval_lightgbm with sequence-level adaptation:
      iter 0: fit Bi-LSTM on source-train sequences only.
      iter k: predict on target-train pool (chronological tiling), HMM-smooth
              using transitions fit ONCE from source labels (never re-estimated
              from pseudo-labels — that would amplify confirmation bias), keep
              top-`quantile` per class by softmax confidence, refit on source
              + kept-pseudo with pseudo_weight on pseudo timesteps.

    A held-out target-val slice (`target_val_frac` of target subjects) supplies
    cross-domain val_loss for early stopping each iteration.

    Returns (metrics, y_pred_flat, y_prob_flat, T_hmm, pi_hmm) — last two
    returned so the caller can reuse them for HMM smoothing without refitting.
    """
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  ERROR: TensorFlow not installed. Skipping Bi-LSTM self-train.")
        return None

    SEQ_LEN = seq_len
    N_FEATURES = X_src.shape[1]

    T_hmm, pi_hmm = fit_hmm_transitions(y_src, g_src)

    rng = np.random.RandomState(SEED)
    target_subjects = np.unique(g_tgt)
    rng.shuffle(target_subjects)
    n_val = max(1, int(target_val_frac * len(target_subjects)))
    val_subj = set(target_subjects[:n_val])
    val_mask = np.isin(g_tgt, list(val_subj))
    pool_mask = ~val_mask

    X_tgt_pool = X_tgt[pool_mask]
    y_tgt_pool_gold = y_tgt[pool_mask]
    g_tgt_pool = g_tgt[pool_mask]
    X_tgt_val = X_tgt[val_mask]
    y_tgt_val = y_tgt[val_mask]
    g_tgt_val = g_tgt[val_mask]

    scaler = StandardScaler()
    def _scale(arr, fit=False):
        s = scaler.fit_transform(arr) if fit else scaler.transform(arr)
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_src_s = _scale(X_src, fit=True)
    X_tgt_pool_s = _scale(X_tgt_pool)
    X_tgt_val_s = _scale(X_tgt_val)
    X_test_s = _scale(X_test)

    cls_w = compute_class_weights(y_src)

    def build_model():
        m = keras.Sequential([
            layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Dense(max(64, lstm_units // 2), activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(N_CLASSES, activation="softmax"),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  weighted_metrics=["accuracy"])
        return m

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    X_val_seq, y_val_seq, w_val_seq, _, _ = make_sequences_blocks(
        X_tgt_val_s, y_tgt_val, g_tgt_val, SEQ_LEN, shuffle=True)
    sw_val_seq = (w_val_seq * np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_val_seq)).astype(np.float32)

    print(f"    [self-train] target-val: {val_mask.sum():,} epochs from {n_val} held-out target subjects")
    print(f"    [self-train] HMM transitions fit from {len(y_src):,} source labels (frozen across iters)")

    # ── Iter 0: source-only ──────────────────────────────────────────────
    print(f"    [self-train] iter 0 (source-only): fitting initial Bi-LSTM on {len(X_src):,} epochs")
    X_trn_seq, y_trn_seq, w_trn_seq, _, _ = make_sequences_blocks(
        X_src_s, y_src, g_src, SEQ_LEN, shuffle=True)
    sw_trn_seq = (w_trn_seq * np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_trn_seq)).astype(np.float32)
    print(f"    [self-train] sequences: {X_trn_seq.shape[0]:,} train, {X_val_seq.shape[0]:,} val (seq_len={SEQ_LEN})")

    model = build_model()
    t0 = time.time()
    model.fit(X_trn_seq, y_trn_seq, sample_weight=sw_trn_seq,
              validation_data=(X_val_seq, y_val_seq, sw_val_seq),
              epochs=120, batch_size=64, callbacks=callbacks, verbose=2)
    total_elapsed = time.time() - t0

    # ── Self-training iterations ─────────────────────────────────────────
    for it in range(1, n_iters + 1):
        print(f"\n    [self-train] iter {it}: predicting on {len(X_tgt_pool):,} target-pool epochs")
        X_pool_seq, _, w_pool_seq, _, flat_order_pool = make_sequences_blocks(
            X_tgt_pool_s, y_tgt_pool_gold, g_tgt_pool, SEQ_LEN, shuffle=False)
        y_prob_seq_pool = model.predict(X_pool_seq, batch_size=128, verbose=0)
        y_prob_padded = y_prob_seq_pool.reshape(-1, N_CLASSES)
        mask_pool = w_pool_seq.reshape(-1) > 0
        y_prob_pool = np.zeros((len(X_tgt_pool), N_CLASSES), dtype=np.float32)
        y_prob_pool[flat_order_pool[mask_pool]] = y_prob_padded[mask_pool]

        y_pseudo_pool = viterbi_smooth(y_prob_pool, g_tgt_pool, T_hmm, pi_hmm)
        conf = y_prob_pool[np.arange(len(y_prob_pool)), y_pseudo_pool]

        keep_mask = np.zeros(len(y_pseudo_pool), dtype=bool)
        kept_per_class = {}
        for c in range(N_CLASSES):
            cls_mask_c = y_pseudo_pool == c
            if cls_mask_c.sum() == 0:
                kept_per_class[c] = 0
                continue
            thresh = np.quantile(conf[cls_mask_c], 1.0 - quantile)
            kept = cls_mask_c & (conf >= thresh)
            keep_mask |= kept
            kept_per_class[c] = int(kept.sum())

        n_kept = int(keep_mask.sum())
        gold_acc = float((y_pseudo_pool[keep_mask] == y_tgt_pool_gold[keep_mask]).mean()) if n_kept else 0.0
        print(f"    [self-train] iter {it}: kept {n_kept:,}/{len(keep_mask):,} pseudo-labels (q={quantile})")
        print(f"                 per-class kept: " +
              ", ".join(f"{STAGE_NAMES[c]}={kept_per_class[c]:,}" for c in range(N_CLASSES)))
        print(f"                 [diag] kept-pseudo accuracy vs gold: {gold_acc:.3f}")

        X_mix_flat = np.concatenate([X_src_s, X_tgt_pool_s[keep_mask]])
        y_mix_flat = np.concatenate([y_src, y_pseudo_pool[keep_mask]])
        g_mix_flat = np.concatenate([g_src, g_tgt_pool[keep_mask]])
        src_flag_flat = np.concatenate([np.ones(len(X_src_s), dtype=np.float32),
                                         np.zeros(n_kept, dtype=np.float32)])

        X_mix_seq, y_mix_seq, w_mix_seq, _, _ = make_sequences_blocks(
            X_mix_flat, y_mix_flat, g_mix_flat, SEQ_LEN, shuffle=True)
        src_flag_seq = tile_flat_to_blocks(src_flag_flat, g_mix_flat, SEQ_LEN, shuffle=True)
        factor_seq = src_flag_seq + (1.0 - src_flag_seq) * pseudo_weight
        sw_mix_seq = (w_mix_seq * factor_seq *
                      np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_mix_seq)).astype(np.float32)

        print(f"    [self-train] iter {it}: refitting on {X_mix_seq.shape[0]:,} mixed sequences")
        model = build_model()
        t0 = time.time()
        model.fit(X_mix_seq, y_mix_seq, sample_weight=sw_mix_seq,
                  validation_data=(X_val_seq, y_val_seq, sw_val_seq),
                  epochs=120, batch_size=64, callbacks=callbacks, verbose=2)
        total_elapsed += time.time() - t0

    # ── Test ─────────────────────────────────────────────────────────────
    X_te_seq, _, w_te_seq, _, flat_order_te = make_sequences_blocks(
        X_test_s, y_test, groups_test, SEQ_LEN, shuffle=False)
    y_prob_seq_te = model.predict(X_te_seq, batch_size=128, verbose=0)
    y_prob_padded_te = y_prob_seq_te.reshape(-1, N_CLASSES)
    mask_te = w_te_seq.reshape(-1) > 0
    y_prob_te = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)
    y_prob_te[flat_order_te[mask_te]] = y_prob_padded_te[mask_te]
    y_pred_te = y_prob_te.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred_te, y_prob=y_prob_te, groups=groups_test)
    metrics["train_time_sec"] = float(total_elapsed)
    if sites_test is not None:
        metrics["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred_te, sites_test)
    return metrics, y_pred_te, y_prob_te, T_hmm, pi_hmm


def train_eval_1d_cnn(X_train, y_train, groups_train, X_test, y_test, groups_test, sites_test):
    """Train 1D-CNN, evaluate on test."""
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  ERROR: TensorFlow not installed. Skipping 1D-CNN.")
        return None

    SEQ_LEN = 20
    N_FEATURES = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Internal val split (15% of train subjects)
    train_subjects = np.unique(groups_train)
    rng = np.random.RandomState(SEED)
    rng.shuffle(train_subjects)
    val_n = max(1, int(0.15 * len(train_subjects)))
    val_subjects = set(train_subjects[:val_n])

    val_mask = np.isin(groups_train, list(val_subjects))
    trn_mask = ~val_mask

    print(f"    1D-CNN internal split: {trn_mask.sum():,} train, {val_mask.sum():,} val epochs")

    X_trn_seq, y_trn_seq, _ = make_sequences(X_train_s[trn_mask], y_train[trn_mask],
                                               groups_train[trn_mask], SEQ_LEN)
    X_val_seq, y_val_seq, _ = make_sequences(X_train_s[val_mask], y_train[val_mask],
                                              groups_train[val_mask], SEQ_LEN)
    X_test_seq, y_test_seq, groups_test_seq = make_sequences(X_test_s, y_test,
                                                              groups_test, SEQ_LEN)

    # Also need sites for test sequences — propagate site from groups
    # Build a group→site mapping from the flat arrays
    sites_test_seq = None
    if sites_test is not None:
        subj_to_site = {}
        for subj, site in zip(groups_test, sites_test):
            subj_to_site[subj] = site
        sites_test_seq = np.array([subj_to_site.get(g, "") for g in groups_test_seq])

    print(f"    Sequences: {X_trn_seq.shape[0]:,} train, {X_val_seq.shape[0]:,} val, "
          f"{X_test_seq.shape[0]:,} test")

    weights = compute_class_weights(y_trn_seq)

    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, N_FEATURES)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]

    t0 = time.time()
    model.fit(
        X_trn_seq, y_trn_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=200, batch_size=256,
        class_weight=weights, callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - t0

    y_prob = model.predict(X_test_seq, batch_size=512)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_test_seq, y_pred, y_prob=y_prob, groups=groups_test_seq)
    metrics["train_time_sec"] = float(elapsed)

    if sites_test_seq is not None:
        metrics["per_site_kappa"] = compute_per_site_kappa(y_test_seq, y_pred, sites_test_seq)

    return metrics


# ── Conditions ────────────────────────────────────────────────────────────────

CONDITIONS = {
    "A_mesa_to_stages": {
        "description": "Train MESA only -> Test STAGES tier1",
        "train_domains": ["mesa"],
        "test_domain": "stages",
    },
    "B_stages_to_mesa": {
        "description": "Train STAGES tier1 only -> Test MESA only",
        "train_domains": ["stages"],
        "test_domain": "mesa",
    },
    "C_combined_to_mesa": {
        "description": "Train combined tier1 -> Test MESA only",
        "train_domains": ["mesa", "stages"],
        "test_domain": "mesa",
    },
    "D_combined_to_stages": {
        "description": "Train combined tier1 -> Test STAGES tier1",
        "train_domains": ["mesa", "stages"],
        "test_domain": "stages",
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-domain sleep staging evaluation")
    parser.add_argument("--models", nargs="+", default=["lightgbm", "1d-cnn"],
                        choices=["lightgbm", "1d-cnn", "bi-lstm"],
                        help="Models to evaluate (default: lightgbm + 1d-cnn)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=list(CONDITIONS.keys()),
                        help="Subset of conditions to run (default: all four A/B/C/D)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to combined parquet (default: combined_clean_wearable_full.parquet)")
    parser.add_argument("--zscore-subject", action="store_true",
                        help="Z-score time-varying features within each subject before training")
    parser.add_argument("--zscore-keep-absolute", action="store_true",
                        help="When z-scoring, keep hr_min/hr_max/spo2_min on absolute scale "
                             "(clinical levels preserved). Requires --zscore-subject.")
    parser.add_argument("--add-site-onehot", action="store_true",
                        help="Add site column as one-hot features")
    parser.add_argument("--hmm-smooth", action="store_true",
                        help="Apply Viterbi smoothing on top of predicted probabilities "
                             "using a transition matrix fit from training labels. LGBM only.")
    parser.add_argument("--self-train", action="store_true",
                        help="Target-side self-training for A/B: pseudo-label target-train "
                             "subjects with source model, HMM-smooth, filter per-class top-K%%, "
                             "retrain on gold+pseudo. LGBM only. C/D: no-op (no unlabeled target).")
    parser.add_argument("--pseudo-quantile", type=float, default=0.5,
                        help="Fraction of most-confident pseudo-labels to keep PER CLASS "
                             "(default: 0.5 = top 50%%). Per-class to prevent N1/N3 starvation.")
    parser.add_argument("--pseudo-weight", type=float, default=0.5,
                        help="Sample-weight multiplier for pseudo-labeled rows (default: 0.5).")
    parser.add_argument("--self-train-iters", type=int, default=1,
                        help="Number of self-training iterations (default: 1).")
    parser.add_argument("--keep-dead-features", action="store_true",
                        help="Disable auto-drop of features with ≥80%% NaN in either domain. "
                             "By default, dataset-specific demographics, non-overlapping race "
                             "one-hots, and sensors absent in one dataset (e.g. actig_white_light) "
                             "are dropped to give an honest cross-domain feature space.")
    parser.add_argument("--dead-nan-thresh", type=float, default=0.80,
                        help="NaN-rate threshold for auto-drop (default: 0.80). A column is "
                             "dropped if its NaN rate is ≥thresh in any domain.")
    parser.add_argument("--bilstm-seq-len", type=int, default=120,
                        help="Bi-LSTM sequence length in 30s epochs (default: 120 = 1h). "
                             "Shorter (e.g. 60) reduces overfitting to source-specific "
                             "long-range temporal autocorrelation.")
    parser.add_argument("--target-val-frac", type=float, default=0.0,
                        help="Fraction of TARGET-train subjects to hold out as cross-domain "
                             "validation set for Bi-LSTM early stopping. 0.0 (default) keeps "
                             "legacy within-source val behavior. A/B only — for C/D, target "
                             "is already in the train pool so this is a no-op.")
    parser.add_argument("--bilstm-units", type=int, default=128,
                        help="Bi-LSTM hidden units per direction (default: 128). Halving to "
                             "64 reduces capacity and may close the cross-domain gap when the "
                             "model overfits source-specific temporal patterns.")
    args = parser.parse_args()

    if args.zscore_keep_absolute and not args.zscore_subject:
        parser.error("--zscore-keep-absolute requires --zscore-subject")
    if args.self_train and not any(m in args.models for m in ("lightgbm", "bi-lstm")):
        parser.error("--self-train requires --models lightgbm or bi-lstm")

    data_path = Path(args.data) if args.data else DATA_PATH
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, feature_cols = load_and_prepare(data_path, add_site_onehot=args.add_site_onehot)

    # ── Domain masks ──────────────────────────────────────────────────────
    mesa_mask = df["dataset"] == "mesa"
    stages_tier1_mask = (df["dataset"] == "stages") & (df["site_tier"] == 1)

    n_mesa = df[mesa_mask]["subject_id"].nunique()
    n_stages = df[stages_tier1_mask]["subject_id"].nunique()
    print(f"\n  MESA subjects: {n_mesa}")
    print(f"  STAGES tier1 subjects (STNF+MSTR): {n_stages}")

    # ── Drop cross-domain dead features ───────────────────────────────────
    if not args.keep_dead_features:
        dead = find_cross_domain_dead_features(
            df, {"mesa": mesa_mask, "stages": stages_tier1_mask},
            feature_cols, nan_thresh=args.dead_nan_thresh)
        if dead:
            print(f"\n  Dropping {len(dead)} cross-domain-dead features "
                  f"(≥{args.dead_nan_thresh:.0%} NaN in one domain):")
            for c, rates in dead:
                print(f"    {c:42s}  mesa={rates['mesa']:.2f}  stages={rates['stages']:.2f}")
            dead_set = {c for c, _ in dead}
            feature_cols = [c for c in feature_cols if c not in dead_set]
            print(f"  Remaining feature count: {len(feature_cols)}")

    # ── Fixed per-domain 80/20 subject splits ─────────────────────────────
    # These held-out test sets are reused across all conditions for fairness.
    mesa_train_subj, mesa_test_subj = split_domain_subjects(df, mesa_mask)
    stages_train_subj, stages_test_subj = split_domain_subjects(df, stages_tier1_mask)

    print(f"\n  MESA split:    {len(mesa_train_subj)} train / {len(mesa_test_subj)} test subjects")
    print(f"  STAGES split:  {len(stages_train_subj)} train / {len(stages_test_subj)} test subjects")

    # Pre-extract test sets (fixed across conditions)
    X_mesa_test, y_mesa_test, g_mesa_test, s_mesa_test = extract_arrays(
        df[mesa_mask], mesa_test_subj, feature_cols)
    X_stages_test, y_stages_test, g_stages_test, s_stages_test = extract_arrays(
        df[stages_tier1_mask], stages_test_subj, feature_cols)

    # Pre-extract train sets per domain
    X_mesa_train, y_mesa_train, g_mesa_train, _ = extract_arrays(
        df[mesa_mask], mesa_train_subj, feature_cols)
    X_stages_train, y_stages_train, g_stages_train, _ = extract_arrays(
        df[stages_tier1_mask], stages_train_subj, feature_cols)

    # Optional: subject-level z-scoring of time-varying features
    if args.zscore_subject:
        if args.zscore_keep_absolute:
            print("\n  Applying per-subject z-scoring "
                  f"(skip demo_*, site_*, tod_*, {', '.join(ABSOLUTE_LEVEL_FEATURES)})...")
        else:
            print("\n  Applying per-subject z-scoring (skip demo_*, site_*, tod_*)...")
        kwargs = {"keep_absolute": args.zscore_keep_absolute}
        X_mesa_train = zscore_per_subject(X_mesa_train, g_mesa_train, feature_cols, **kwargs)
        X_stages_train = zscore_per_subject(X_stages_train, g_stages_train, feature_cols, **kwargs)
        X_mesa_test = zscore_per_subject(X_mesa_test, g_mesa_test, feature_cols, **kwargs)
        X_stages_test = zscore_per_subject(X_stages_test, g_stages_test, feature_cols, **kwargs)

    # Combined train
    X_combined_train = np.concatenate([X_mesa_train, X_stages_train])
    y_combined_train = np.concatenate([y_mesa_train, y_stages_train])
    g_combined_train = np.concatenate([g_mesa_train, g_stages_train])

    # Map condition -> (X_train, y_train, groups_train, X_test, y_test, groups_test, sites_test,
    #                   X_tgt_unlabeled, y_tgt_unlabeled, g_tgt_unlabeled)
    # target-unlabeled is only defined for A and B (single-source conditions); for C/D the
    # target-train subjects are already in the training set, so self-training is a no-op
    # and Bi-LSTM target-val falls back to within-source val.
    condition_data = {
        "A_mesa_to_stages": (X_mesa_train, y_mesa_train, g_mesa_train,
                             X_stages_test, y_stages_test, g_stages_test, s_stages_test,
                             X_stages_train, y_stages_train, g_stages_train),
        "B_stages_to_mesa": (X_stages_train, y_stages_train, g_stages_train,
                             X_mesa_test, y_mesa_test, g_mesa_test, s_mesa_test,
                             X_mesa_train, y_mesa_train, g_mesa_train),
        "C_combined_to_mesa": (X_combined_train, y_combined_train, g_combined_train,
                               X_mesa_test, y_mesa_test, g_mesa_test, s_mesa_test,
                               None, None, None),
        "D_combined_to_stages": (X_combined_train, y_combined_train, g_combined_train,
                                 X_stages_test, y_stages_test, g_stages_test, s_stages_test,
                                 None, None, None),
    }

    # ── Run all conditions ────────────────────────────────────────────────
    all_results = {}
    summary_rows = []

    active_conditions = args.conditions if args.conditions else list(CONDITIONS.keys())

    for cond_name in active_conditions:
        cond_info = CONDITIONS[cond_name]
        X_tr, y_tr, g_tr, X_te, y_te, g_te, s_te, X_tgt_unl, y_tgt_unl, g_tgt_unl = \
            condition_data[cond_name]
        use_selftrain = args.self_train and X_tgt_unl is not None
        use_target_val = (args.target_val_frac > 0.0 and X_tgt_unl is not None
                          and "bi-lstm" in args.models)

        print(f"\n{'=' * 70}")
        print(f"  Condition: {cond_name}")
        print(f"  {cond_info['description']}")
        print(f"  Train: {len(X_tr):,} epochs | Test: {len(X_te):,} epochs")
        if args.self_train and not use_selftrain:
            print(f"  NOTE: --self-train inactive for this condition (no unlabeled target).")
        if use_selftrain:
            print(f"  Self-train: {len(X_tgt_unl):,} target-unlabeled epochs "
                  f"(quantile={args.pseudo_quantile}, weight={args.pseudo_weight}, "
                  f"iters={args.self_train_iters})")
        if use_target_val:
            print(f"  Bi-LSTM target-val: hold out {args.target_val_frac:.0%} of "
                  f"target-train subjects for cross-domain early stopping.")
        if args.target_val_frac > 0.0 and X_tgt_unl is None:
            print(f"  NOTE: --target-val-frac inactive for this condition "
                  f"(target already in train pool).")
        print(f"{'=' * 70}")

        # Impute NaN (medians from train). Target arrays come along whenever
        # they exist so self-train AND Bi-LSTM target-val both have access.
        if X_tgt_unl is not None:
            X_tr_imp, X_tgt_imp, X_te_imp = impute_nan_splits(X_tr, X_tgt_unl, X_te)
        else:
            X_tr_imp, X_te_imp = impute_nan_splits(X_tr, X_te)
            X_tgt_imp = None

        cond_results = {}

        for model_name in args.models:
            print(f"\n  --- {model_name} ---")

            T_hmm_precomputed = None
            pi_hmm_precomputed = None

            if model_name == "lightgbm" and use_selftrain:
                metrics, y_pred_raw, y_prob_raw, T_hmm_precomputed, pi_hmm_precomputed = \
                    self_train_eval_lightgbm(
                        X_tr_imp, y_tr, g_tr, X_tgt_imp, g_tgt_unl,
                        X_te_imp, y_te, g_te, s_te,
                        quantile=args.pseudo_quantile,
                        pseudo_weight=args.pseudo_weight,
                        n_iters=args.self_train_iters)
            elif model_name == "bi-lstm" and use_selftrain:
                result = self_train_eval_bi_lstm(
                    X_tr_imp, y_tr, g_tr, X_tgt_imp, y_tgt_unl, g_tgt_unl,
                    X_te_imp, y_te, g_te, s_te,
                    target_val_frac=args.target_val_frac if args.target_val_frac > 0 else 0.10,
                    quantile=args.pseudo_quantile,
                    pseudo_weight=args.pseudo_weight,
                    n_iters=args.self_train_iters,
                    seq_len=args.bilstm_seq_len,
                    lstm_units=args.bilstm_units)
                if result is None:
                    continue
                metrics, y_pred_raw, y_prob_raw, T_hmm_precomputed, pi_hmm_precomputed = result
            elif model_name == "lightgbm":
                metrics, y_pred_raw, y_prob_raw = train_eval_lightgbm(
                    X_tr_imp, y_tr, X_te_imp, y_te, g_te, s_te)
            elif model_name == "1d-cnn":
                metrics = train_eval_1d_cnn(X_tr_imp, y_tr, g_tr,
                                            X_te_imp, y_te, g_te, s_te)
                y_pred_raw = y_prob_raw = None
            elif model_name == "bi-lstm":
                result = train_eval_bi_lstm(
                    X_tr_imp, y_tr, g_tr,
                    X_te_imp, y_te, g_te, s_te,
                    X_target_train=X_tgt_imp,
                    y_target_train=y_tgt_unl,
                    groups_target_train=g_tgt_unl,
                    target_val_frac=args.target_val_frac,
                    seq_len=args.bilstm_seq_len,
                    lstm_units=args.bilstm_units)
                if result is None:
                    continue
                metrics, y_pred_raw, y_prob_raw = result
            else:
                continue

            if metrics is None:
                continue

            # Optional HMM smoothing (LGBM + Bi-LSTM; CNN test sequences are shuffled so skipped)
            if args.hmm_smooth and model_name in ("lightgbm", "bi-lstm"):
                if T_hmm_precomputed is not None:
                    T_hmm, pi_hmm = T_hmm_precomputed, pi_hmm_precomputed
                    print(f"    Reusing HMM transitions from self-train (fit on {len(y_tr):,} source labels)")
                else:
                    print(f"    Fitting HMM transition matrix from {len(y_tr):,} train labels...")
                    T_hmm, pi_hmm = fit_hmm_transitions(y_tr, g_tr)
                y_pred_hmm = viterbi_smooth(y_prob_raw, g_te, T_hmm, pi_hmm)
                m_hmm = compute_metrics(y_te, y_pred_hmm, y_prob=y_prob_raw, groups=g_te)
                metrics["hmm"] = {
                    "cohen_kappa": m_hmm["cohen_kappa"],
                    "n3_sensitivity": m_hmm["n3_sensitivity"],
                    "n3_f1": m_hmm["n3_f1"],
                    "per_class_f1": m_hmm["per_class_f1"],
                    "confusion_matrix": m_hmm["confusion_matrix"],
                }
                if s_te is not None:
                    metrics["hmm"]["per_site_kappa"] = compute_per_site_kappa(y_te, y_pred_hmm, s_te)
                if "n3_duration_correlation" in m_hmm:
                    metrics["hmm"]["n3_duration_correlation"] = m_hmm["n3_duration_correlation"]

            cond_results[model_name] = metrics

            # Print summary
            print(f"\n  {model_name} results for {cond_name}:")
            print(f"    Cohen's kappa:    {metrics['cohen_kappa']:.4f}")
            print(f"    N3 Sensitivity:   {metrics['n3_sensitivity']:.4f}")
            print(f"    N3 F1:            {metrics['n3_f1']:.4f}")
            if metrics.get("auroc_macro"):
                print(f"    AUROC (macro):    {metrics['auroc_macro']:.4f}")
            print(f"    Per-class F1:     ", end="")
            for name in STAGE_NAMES:
                print(f"{name}={metrics['per_class_f1'][name]:.3f} ", end="")
            print()

            if "per_site_kappa" in metrics:
                print(f"    Per-site kappa:")
                for site, info in sorted(metrics["per_site_kappa"].items()):
                    print(f"      {site}: kappa={info['kappa']:.4f} (n={info['n_epochs']:,})")

            if "n3_duration_correlation" in metrics:
                r_info = metrics["n3_duration_correlation"]
                print(f"    N3 duration r:    {r_info['pearson_r']:.4f} (p={r_info['p_value']:.2e})")

            if "hmm" in metrics:
                h = metrics["hmm"]
                print(f"    HMM-smoothed:     kappa={h['cohen_kappa']:.4f}  "
                      f"N3 F1={h['n3_f1']:.4f}  N3 sens={h['n3_sensitivity']:.4f}")
                print(f"                      per-class F1: " +
                      " ".join(f"{n}={h['per_class_f1'][n]:.3f}" for n in STAGE_NAMES))

            # Summary row
            row = {
                "condition": cond_name,
                "description": cond_info["description"],
                "model": model_name,
                "cohen_kappa": metrics["cohen_kappa"],
                "auroc_macro": metrics.get("auroc_macro", ""),
                "n3_sensitivity": metrics["n3_sensitivity"],
                "n3_f1": metrics["n3_f1"],
                **{f"f1_{name}": metrics["per_class_f1"][name] for name in STAGE_NAMES},
                "n_train": len(X_tr),
                "n_test": metrics["n_test_samples"],
                "train_time_sec": metrics.get("train_time_sec", ""),
            }
            if "hmm" in metrics:
                h = metrics["hmm"]
                row["cohen_kappa_hmm"] = h["cohen_kappa"]
                row["n3_f1_hmm"] = h["n3_f1"]
                row["n3_sensitivity_hmm"] = h["n3_sensitivity"]
                for n in STAGE_NAMES:
                    row[f"f1_{n}_hmm"] = h["per_class_f1"][n]
            summary_rows.append(row)

        all_results[cond_name] = cond_results

    # ── Save results ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    if args.zscore_subject:
        parts.append("zscored")
    if args.zscore_keep_absolute:
        parts.append("keepabs")
    if args.add_site_onehot:
        parts.append("site")
    if args.hmm_smooth:
        parts.append("hmm")
    if args.self_train:
        parts.append(f"st-q{args.pseudo_quantile}-w{args.pseudo_weight}-i{args.self_train_iters}")
    if "bi-lstm" in args.models and args.bilstm_seq_len != 120:
        parts.append(f"sl{args.bilstm_seq_len}")
    if "bi-lstm" in args.models and args.target_val_frac > 0.0:
        parts.append(f"tval{args.target_val_frac:.2f}")
    if "bi-lstm" in args.models and args.bilstm_units != 128:
        parts.append(f"u{args.bilstm_units}")
    suffix = ("_" + "_".join(parts)) if parts else ""

    # Full JSON
    json_path = RESULTS_DIR / f"cross_domain_results{suffix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results: {json_path}")

    # Summary CSV
    csv_path = RESULTS_DIR / f"cross_domain_summary{suffix}_{timestamp}.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  Summary CSV:  {csv_path}")

    # ── Print comparison table ────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  CROSS-DOMAIN COMPARISON")
    print(f"{'=' * 90}")
    print(f"  {'Condition':<25} {'Model':<12} {'kappa':>8} {'N3 Sens':>8} {'N3 F1':>8} {'AUROC':>8}")
    print(f"  {'-' * 75}")
    for row in summary_rows:
        auroc_str = f"{row['auroc_macro']:.4f}" if row['auroc_macro'] != "" else "  N/A"
        print(f"  {row['condition']:<25} {row['model']:<12} "
              f"{row['cohen_kappa']:>8.4f} {row['n3_sensitivity']:>8.4f} "
              f"{row['n3_f1']:>8.4f} {auroc_str:>8}")

    print(f"\n  Per-class F1 breakdown:")
    print(f"  {'Condition':<25} {'Model':<12}" + "".join(f"{n:>8}" for n in STAGE_NAMES))
    print(f"  {'-' * (37 + 8 * len(STAGE_NAMES))}")
    for row in summary_rows:
        f1s = "".join(f"{row[f'f1_{n}']:>8.3f}" for n in STAGE_NAMES)
        print(f"  {row['condition']:<25} {row['model']:<12}{f1s}")

    print("\nDone!")


if __name__ == "__main__":
    main()
