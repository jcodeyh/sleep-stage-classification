#!/usr/bin/env python3
"""Zero-shot evaluation: train MESA + STAGES tier1 -> test DREAMT.

Methodology (Option A, "shared-core"): restrict features to columns that have
<80% NaN in *all three* domains (MESA, STAGES tier1, DREAMT). This produces a
29-feature core (HR x4, HRV x4, actig x3, derived rolling/delta x15, demographics
x3 = age/sex/BMI). SpO2, tod_cos/tod_sin, actig_white_light, and clinical
demographics not in DREAMT are auto-dropped.

Reuses cross_domain_eval.py utilities (z-scoring, HMM smoothing, LGBM trainer,
metrics) so this script stays thin and consistent with the existing pipeline.

Usage:
    python scripts/dreamt_zeroshot_eval.py
    python scripts/dreamt_zeroshot_eval.py --models lightgbm bi-lstm
    python scripts/dreamt_zeroshot_eval.py --models lightgbm --self-train
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cross_domain_eval import (  # noqa: E402
    DATA_PATH,
    SEED,
    N_CLASSES,
    STAGE_NAMES,
    STAGE_MAP_5,
    CLIP_RANGES,
    LEAKY_COLS,
    PSG_ONLY_COLS,
    META_COLS,
    _fit_lightgbm,
    compute_class_weights,
    compute_metrics,
    compute_per_site_kappa,
    extract_arrays,
    fit_hmm_transitions,
    load_and_prepare,
    make_sequences_blocks,
    split_domain_subjects,
    viterbi_smooth,
    zscore_per_subject,
)

DREAMT_PATH = PROJECT_ROOT / "datasets" / "dreamt" / "dreamt_clean_wearable_full.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "dreamt_zeroshot"

DEAD_NAN_THRESH = 0.80


# ── Prior-shift calibration ──────────────────────────────────────────────────

def class_prior(y, n_classes=N_CLASSES, eps=1e-9):
    """Empirical class prior from labels."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return (counts + eps) / (counts.sum() + n_classes * eps)


def calibrate_priors(y_prob, p_source, p_target, eps=1e-9):
    """Saerens-Latinne-Decaestecker prior-shift correction.

    Under the label-shift assumption p_target(x|c) == p_source(x|c), we have
        p_target(c | x) ∝ p_source(c | x) * p_target(c) / p_source(c)
    This is a per-row reweighting + renormalization over y_prob's class axis.

    Returns calibrated y_prob; original is not modified.
    """
    ratio = (p_target + eps) / (p_source + eps)
    y_new = y_prob * ratio[None, :]
    y_new /= y_new.sum(axis=1, keepdims=True)
    return y_new


def split_dreamt_for_sweep(g_dreamt, n_test_subj, seed=SEED):
    """Pre-shuffle DREAMT subjects, last n_test_subj as FIXED test, rest as
    ordered cal pool. Caller takes the first K from cal pool for cohort size K.
    Keeps test composition constant across sweep, so Δκ vs cal-size is a clean
    scaling curve.
    """
    subjects = np.unique(g_dreamt)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(subjects))
    shuffled = subjects[perm]
    cal_pool = shuffled[: len(subjects) - n_test_subj]  # ordered list
    test_subjects = set(shuffled[len(subjects) - n_test_subj:])
    test_mask = np.isin(g_dreamt, list(test_subjects))
    return test_mask, list(cal_pool), test_subjects


def split_dreamt_for_calibration(g_dreamt, n_cal_subj, seed=SEED):
    """Subject-level split of DREAMT for prior calibration. Returns
    (cal_mask, test_mask, cal_subjects, test_subjects).
    """
    subjects = np.unique(g_dreamt)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(subjects))
    cal_subjects = set(subjects[perm[:n_cal_subj]])
    test_subjects = set(subjects[perm[n_cal_subj:]])
    cal_mask = np.isin(g_dreamt, list(cal_subjects))
    test_mask = ~cal_mask
    return cal_mask, test_mask, cal_subjects, test_subjects


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dreamt(feature_cols):
    """Load DREAMT, apply same usability filter as combined, return df with
    `label` column and any combined-only feature_cols filled with NaN.

    DREAMT is treated as one site for per-site metrics; site column synthesized.
    """
    print(f"Loading DREAMT from {DREAMT_PATH}...")
    df = pd.read_parquet(DREAMT_PATH)

    mask = df.sleep_stage.isin([0, 1, 2, 3, 5]) & df.hr_mean.notna()
    df = df[mask].copy()
    df["label"] = df.sleep_stage.map(STAGE_MAP_5)
    df["site"] = "dreamt"

    # Add any combined-only feature columns as all-NaN so extract_arrays
    # produces a column-aligned X. The shared-core filter will drop them.
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Clip physiological outliers (matches combined preprocessing)
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    print(f"  DREAMT usable epochs: {len(df):,} from {df['subject_id'].nunique()} subjects")
    return df


def compute_shared_core(df_combined, df_dreamt, mesa_mask, stages_tier1_mask,
                        feature_cols, nan_thresh=DEAD_NAN_THRESH):
    """Identify the shared-core feature set: columns with <nan_thresh NaN rate
    in MESA, STAGES tier1, AND DREAMT. Returns (kept_cols, dropped_with_rates).
    """
    kept, dropped = [], []
    for c in feature_cols:
        m = float(df_combined.loc[mesa_mask, c].isna().mean())
        s = float(df_combined.loc[stages_tier1_mask, c].isna().mean())
        d = float(df_dreamt[c].isna().mean())
        if max(m, s, d) >= nan_thresh:
            dropped.append((c, m, s, d))
        else:
            kept.append(c)
    return kept, dropped


# ── Evaluation ───────────────────────────────────────────────────────────────

def fit_bilstm_and_predict(X_train, y_train, g_train, test_sets, seq_len=120,
                            lstm_units=128, max_epochs=120, batch_size=64):
    """Train Bi-LSTM on (X_train, y_train) with within-source val for early
    stopping. Predict on each test set. Mirrors cross_domain_eval.train_eval_bi_lstm
    but exposes a single fit followed by N predicts so we don't retrain per
    test set.

    Args:
        test_sets: list of (label, X, y, groups, sites) — predict and assemble
            flat-order y_prob aligned to (X, y, groups) for each.

    Returns:
        dict {label: (y_prob_flat, scaler)} where y_prob_flat is (len(X), C),
        plus a 'fit_seconds' key.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler

    n_features = X_train.shape[1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    rng = np.random.RandomState(SEED)
    train_subjects = np.unique(g_train)
    rng.shuffle(train_subjects)
    val_n = max(1, int(0.15 * len(train_subjects)))
    val_subjects = set(train_subjects[:val_n])
    val_mask = np.isin(g_train, list(val_subjects))
    trn_mask = ~val_mask

    print(f"    Bi-LSTM internal split: {trn_mask.sum():,} train, "
          f"{val_mask.sum():,} val epochs ({val_n}/{len(train_subjects)} subjects)")

    X_trn_seq, y_trn_seq, w_trn_seq, _, _ = make_sequences_blocks(
        X_train_s[trn_mask], y_train[trn_mask], g_train[trn_mask], seq_len, shuffle=True)
    X_val_seq, y_val_seq, w_val_seq, _, _ = make_sequences_blocks(
        X_train_s[val_mask], y_train[val_mask], g_train[val_mask], seq_len, shuffle=True)
    print(f"    Sequences: {X_trn_seq.shape[0]:,} train, {X_val_seq.shape[0]:,} val "
          f"(seq_len={seq_len})")

    cls_w = compute_class_weights(y_trn_seq[w_trn_seq > 0])
    sw_trn = (w_trn_seq * np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_trn_seq)).astype(np.float32)
    sw_val = (w_val_seq * np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_val_seq)).astype(np.float32)

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
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

    print(f"    Bi-LSTM params: {model.count_params():,}")
    t0 = time.time()
    model.fit(
        X_trn_seq, y_trn_seq,
        sample_weight=sw_trn,
        validation_data=(X_val_seq, y_val_seq, sw_val),
        epochs=max_epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )
    fit_elapsed = time.time() - t0

    # Predict on each test set, returning flat-order y_prob aligned to (X, y, groups)
    out = {"fit_seconds": float(fit_elapsed), "model": model, "scaler": scaler,
           "seq_len": seq_len}
    for label, X_te, _y_te, g_te, _s_te in test_sets:
        out[label] = _predict_bilstm_flat(model, scaler, X_te, g_te, seq_len)
    return out


def _predict_bilstm_flat(model, scaler, X_flat, g_flat, seq_len):
    """Apply scaler, tile to blocks, predict, re-assemble flat-order y_prob."""
    X_s = scaler.transform(X_flat)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_seq, _, w_seq, _, flat_order = make_sequences_blocks(
        X_s, np.zeros(len(X_s), dtype=np.int32), g_flat, seq_len, shuffle=False)
    y_prob_seq = model.predict(X_seq, batch_size=128, verbose=0)
    y_prob_padded = y_prob_seq.reshape(-1, N_CLASSES)
    mask_flat = w_seq.reshape(-1) > 0
    y_prob_flat = np.zeros((len(X_flat), N_CLASSES), dtype=np.float32)
    y_prob_flat[flat_order[mask_flat]] = y_prob_padded[mask_flat]
    return y_prob_flat


def fine_tune_bilstm(model, scaler, X_cal, y_cal, g_cal, seq_len,
                     ft_epochs, ft_lr, batch_size=64):
    """Continue training a base Bi-LSTM on a small labeled calibration set.

    Lower LR (1e-4 default) than base training (1e-3). No early stopping —
    too few subjects to safely hold out a val slice.
    """
    from tensorflow import keras

    X_cal_s = scaler.transform(X_cal)
    X_cal_s = np.nan_to_num(X_cal_s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_cal_seq, y_cal_seq, w_cal_seq, _, _ = make_sequences_blocks(
        X_cal_s, y_cal, g_cal, seq_len, shuffle=True)

    cls_w = compute_class_weights(y_cal_seq[w_cal_seq > 0])
    sw_cal = (w_cal_seq * np.vectorize(lambda c: cls_w.get(int(c), 1.0))(y_cal_seq)
              ).astype(np.float32)

    print(f"    Fine-tune sequences: {X_cal_seq.shape[0]:,} (seq_len={seq_len}, "
          f"epochs={ft_epochs}, lr={ft_lr})")
    print(f"    Cal class weights: " +
          " ".join(f"{STAGE_NAMES[c]}={cls_w.get(c, 0):.2f}" for c in range(N_CLASSES)))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ft_lr),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )
    t0 = time.time()
    model.fit(
        X_cal_seq, y_cal_seq, sample_weight=sw_cal,
        epochs=ft_epochs, batch_size=batch_size, verbose=2,
    )
    return model, time.time() - t0


def evaluate_lightgbm(model, scaler, X_test, y_test, groups_test, sites_test,
                      T_hmm=None, pi_hmm=None):
    """Predict, optionally HMM-smooth, return (metrics, y_pred, y_prob)."""
    X_s = scaler.transform(X_test)
    y_prob = model.predict_proba(X_s)
    y_pred_raw = y_prob.argmax(axis=1)

    metrics_raw = compute_metrics(y_test, y_pred_raw, y_prob=y_prob, groups=groups_test)
    if sites_test is not None:
        metrics_raw["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred_raw, sites_test)

    metrics_hmm = None
    y_pred_hmm = None
    if T_hmm is not None:
        y_pred_hmm = viterbi_smooth(y_prob, groups_test, T_hmm, pi_hmm)
        metrics_hmm = compute_metrics(y_test, y_pred_hmm, y_prob=y_prob, groups=groups_test)
        if sites_test is not None:
            metrics_hmm["per_site_kappa"] = compute_per_site_kappa(y_test, y_pred_hmm, sites_test)

    return metrics_raw, metrics_hmm, y_pred_raw, y_pred_hmm, y_prob


def evaluate_with_calibration(y_prob_dr, y_dr, g_dr, s_dr, p_source,
                              n_cal_subj, T_hmm=None, pi_hmm=None):
    """Hold out n_cal_subj DREAMT subjects, estimate target priors from their
    labels, reweight y_prob on the remaining (test) subjects. Evaluate both
    uncalibrated and calibrated on the SAME test subjects for fair comparison.
    """
    cal_mask, test_mask, cal_subj, test_subj = split_dreamt_for_calibration(g_dr, n_cal_subj)
    p_target = class_prior(y_dr[cal_mask])

    y_prob_test = y_prob_dr[test_mask]
    y_test = y_dr[test_mask]
    g_test = g_dr[test_mask]
    s_test = s_dr[test_mask] if s_dr is not None else None

    def _eval(probs):
        pred_raw = probs.argmax(axis=1)
        m_raw = compute_metrics(y_test, pred_raw, y_prob=probs, groups=g_test)
        if s_test is not None:
            m_raw["per_site_kappa"] = compute_per_site_kappa(y_test, pred_raw, s_test)
        m_hmm = None
        if T_hmm is not None:
            pred_hmm = viterbi_smooth(probs, g_test, T_hmm, pi_hmm)
            m_hmm = compute_metrics(y_test, pred_hmm, y_prob=probs, groups=g_test)
            if s_test is not None:
                m_hmm["per_site_kappa"] = compute_per_site_kappa(y_test, pred_hmm, s_test)
        return m_raw, m_hmm

    un_raw, un_hmm = _eval(y_prob_test)
    cal_raw, cal_hmm = _eval(calibrate_priors(y_prob_test, p_source, p_target))

    return {
        "n_cal_subj": int(len(cal_subj)),
        "n_test_subj": int(len(test_subj)),
        "n_cal_epochs": int(cal_mask.sum()),
        "n_test_epochs": int(test_mask.sum()),
        "p_source": p_source.tolist(),
        "p_target": p_target.tolist(),
        "uncalibrated_raw": un_raw,
        "uncalibrated_hmm": un_hmm,
        "calibrated_raw": cal_raw,
        "calibrated_hmm": cal_hmm,
    }


def print_metrics_block(label, metrics):
    """Compact per-condition metrics summary. Matches compute_metrics output schema."""
    pcf1 = metrics["per_class_f1"]
    macro_f1 = float(np.mean(list(pcf1.values())))
    metrics["_macro_f1"] = macro_f1  # cache for downstream summary

    print(f"\n  {label}")
    print(f"    accuracy    = {metrics['accuracy']:.4f}")
    print(f"    cohen_kappa = {metrics['cohen_kappa']:.4f}")
    print(f"    macro_f1    = {macro_f1:.4f}")
    if metrics.get("auroc_macro") is not None:
        print(f"    AUROC macro = {metrics['auroc_macro']:.4f}")
    print(f"    N3 sens={metrics['n3_sensitivity']:.3f}  N3 F1={metrics['n3_f1']:.3f}")
    print(f"    per-class F1: " +
          " ".join(f"{n}={pcf1[n]:.3f}" for n in STAGE_NAMES))
    cm = np.array(metrics["confusion_matrix"])
    print(f"    confusion (rows=true, cols=pred):")
    print(f"      {' ' * 6}" + "".join(f"{n:>8}" for n in STAGE_NAMES))
    for i, n in enumerate(STAGE_NAMES):
        row = "".join(f"{cm[i, j]:>8,}" for j in range(N_CLASSES))
        print(f"      {n:>6}{row}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot DREAMT eval: train MESA+STAGES tier1, test DREAMT")
    parser.add_argument("--models", nargs="+", default=["lightgbm"],
                        choices=["lightgbm", "bi-lstm"],
                        help="Models to evaluate.")
    parser.add_argument("--zscore-keep-absolute", action="store_true",
                        help="When z-scoring, keep hr_min/hr_max on absolute scale.")
    parser.add_argument("--no-hmm", action="store_true",
                        help="Skip HMM Viterbi smoothing (default: smooth).")
    parser.add_argument("--bilstm-seq-len", type=int, default=120,
                        help="Bi-LSTM sequence length in 30s epochs (default: 120 = 1h).")
    parser.add_argument("--bilstm-units", type=int, default=128,
                        help="Bi-LSTM hidden units per direction (default: 128).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enable prior-shift calibration on DREAMT using a held-out "
                             "subset of DREAMT subjects as a labeled calibration slice.")
    parser.add_argument("--n-cal-subj", type=int, default=10,
                        help="DREAMT subjects to hold out as the labeled calibration slice "
                             "(default: 10 of 100). Test set is the remaining subjects.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save y_prob arrays and groups to predictions.npz alongside "
                             "results.json. Lets us iterate on calibration offline.")
    parser.add_argument("--self-train", action="store_true",
                        help="In addition to base Bi-LSTM, run target-side self-training: "
                             "pseudo-label the DREAMT pool, retrain on combined+pseudo. "
                             "Holds out --st-test-subj DREAMT subjects from the pool for "
                             "clean test eval. Per cross_domain_eval precedent.")
    parser.add_argument("--st-pseudo-quantile", type=float, default=0.5,
                        help="Per-class top-K pseudo-label retention (default: 0.5).")
    parser.add_argument("--st-pseudo-weight", type=float, default=0.5,
                        help="Sample-weight multiplier for pseudo-labeled rows (default: 0.5).")
    parser.add_argument("--st-iters", type=int, default=1,
                        help="Self-training iterations (default: 1).")
    parser.add_argument("--st-seq-len", type=int, default=60,
                        help="Self-train Bi-LSTM seq_len (default: 60, matches existing precedent).")
    parser.add_argument("--st-target-val-frac", type=float, default=0.111,
                        help="Internal val fraction applied to the pool+val subjects "
                             "(default: 0.111 → ~10/90).")
    parser.add_argument("--st-test-subj", type=int, default=10,
                        help="DREAMT subjects held out as the final test set (default: 10).")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Few-shot fine-tune the base Bi-LSTM on a small labeled "
                             "DREAMT calibration cohort. Splits DREAMT --ft-cal-subj "
                             "subjects out for fine-tuning, evaluates on the rest.")
    parser.add_argument("--ft-cal-subj", type=int, default=10,
                        help="DREAMT subjects in the labeled calibration cohort "
                             "(default: 10). Test set is the remaining subjects.")
    parser.add_argument("--ft-epochs", type=int, default=20,
                        help="Fine-tune epochs (default: 20). No early stopping — "
                             "too few cal subjects for a safe val split.")
    parser.add_argument("--ft-lr", type=float, default=1e-4,
                        help="Fine-tune learning rate (default: 1e-4 = 10x lower than base).")
    parser.add_argument("--ft-sweep-sizes", type=str, default=None,
                        help="Comma-separated calibration cohort sizes to sweep (e.g. "
                             "'10,20,30,40,50'). Holds test set fixed (--ft-sweep-test-subj) "
                             "so κ vs cal-size is a clean scaling curve. Overrides --fine-tune.")
    parser.add_argument("--ft-sweep-test-subj", type=int, default=50,
                        help="DREAMT subjects to hold out as the FIXED sweep test set "
                             "(default: 50). Cal pool is the remaining 100-N subjects.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Load combined + DREAMT ────────────────────────────────────────────
    df_comb, feature_cols_full = load_and_prepare(DATA_PATH)
    df_dreamt = load_dreamt(feature_cols_full)

    mesa_mask = df_comb["dataset"] == "mesa"
    stages_tier1_mask = (df_comb["dataset"] == "stages") & (df_comb["site_tier"] == 1)

    # ── Shared-core feature filter ────────────────────────────────────────
    kept_cols, dropped = compute_shared_core(
        df_comb, df_dreamt, mesa_mask, stages_tier1_mask,
        feature_cols_full, nan_thresh=DEAD_NAN_THRESH,
    )
    print(f"\n  Shared-core filter (drop ≥{DEAD_NAN_THRESH:.0%} NaN in any of "
          f"mesa/stages_tier1/dreamt):")
    print(f"    kept    = {len(kept_cols)} cols")
    print(f"    dropped = {len(dropped)} cols")
    feature_cols = kept_cols

    # ── Subject splits ────────────────────────────────────────────────────
    mesa_train_subj, mesa_test_subj = split_domain_subjects(df_comb, mesa_mask)
    stages_train_subj, stages_test_subj = split_domain_subjects(df_comb, stages_tier1_mask)

    n_mesa_tr = len(mesa_train_subj)
    n_stages_tr = len(stages_train_subj)
    n_mesa_te = len(mesa_test_subj)
    n_stages_te = len(stages_test_subj)
    print(f"\n  MESA   {n_mesa_tr} train / {n_mesa_te} test subjects")
    print(f"  STAGES {n_stages_tr} train / {n_stages_te} test subjects")
    print(f"  DREAMT {df_dreamt['subject_id'].nunique()} subjects (all zero-shot test)")

    # Extract arrays
    X_mesa_tr, y_mesa_tr, g_mesa_tr, _ = extract_arrays(
        df_comb[mesa_mask], mesa_train_subj, feature_cols)
    X_mesa_te, y_mesa_te, g_mesa_te, s_mesa_te = extract_arrays(
        df_comb[mesa_mask], mesa_test_subj, feature_cols)
    X_stages_tr, y_stages_tr, g_stages_tr, _ = extract_arrays(
        df_comb[stages_tier1_mask], stages_train_subj, feature_cols)
    X_stages_te, y_stages_te, g_stages_te, s_stages_te = extract_arrays(
        df_comb[stages_tier1_mask], stages_test_subj, feature_cols)

    # DREAMT is all-test (no train split)
    X_dr = df_dreamt[feature_cols].values.astype(np.float64)
    y_dr = df_dreamt["label"].values.astype(np.int32)
    g_dr = df_dreamt["subject_id"].values
    s_dr = df_dreamt["site"].values

    # ── Per-subject z-scoring (matches cross_domain_eval convention) ──────
    print("\n  Applying per-subject z-scoring...")
    kw = {"keep_absolute": args.zscore_keep_absolute}
    X_mesa_tr_z = zscore_per_subject(X_mesa_tr, g_mesa_tr, feature_cols, **kw)
    X_mesa_te_z = zscore_per_subject(X_mesa_te, g_mesa_te, feature_cols, **kw)
    X_stages_tr_z = zscore_per_subject(X_stages_tr, g_stages_tr, feature_cols, **kw)
    X_stages_te_z = zscore_per_subject(X_stages_te, g_stages_te, feature_cols, **kw)
    X_dr_z = zscore_per_subject(X_dr, g_dr, feature_cols, **kw)

    # Combined train + in-domain held-out test (concat MESA + STAGES tier1)
    X_train = np.concatenate([X_mesa_tr_z, X_stages_tr_z])
    y_train = np.concatenate([y_mesa_tr, y_stages_tr])
    g_train = np.concatenate([g_mesa_tr, g_stages_tr])

    X_indom = np.concatenate([X_mesa_te_z, X_stages_te_z])
    y_indom = np.concatenate([y_mesa_te, y_stages_te])
    g_indom = np.concatenate([g_mesa_te, g_stages_te])
    s_indom = np.concatenate([s_mesa_te, s_stages_te])

    print(f"\n  Train (combined): {len(X_train):,} epochs, "
          f"{len(np.unique(g_train))} subjects")
    print(f"  In-domain test:   {len(X_indom):,} epochs, "
          f"{len(np.unique(g_indom))} subjects")
    print(f"  DREAMT test:      {len(X_dr):,} epochs, "
          f"{len(np.unique(g_dr))} subjects")

    # ── Aggregate results container ───────────────────────────────────────
    all_results = {
        "config": {
            "data_path": str(DATA_PATH),
            "dreamt_path": str(DREAMT_PATH),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "dropped_features": [
                {"col": c, "mesa_nan": m, "stages_nan": s, "dreamt_nan": d}
                for c, m, s, d in dropped
            ],
            "zscore_keep_absolute": args.zscore_keep_absolute,
            "hmm_smooth": not args.no_hmm,
            "models": args.models,
            "bilstm_seq_len": args.bilstm_seq_len,
            "bilstm_units": args.bilstm_units,
            "seed": SEED,
        },
        "n_train_epochs": int(len(X_train)),
        "n_indom_test_epochs": int(len(X_indom)),
        "n_dreamt_epochs": int(len(X_dr)),
        "n_train_subjects": int(len(np.unique(g_train))),
        "n_indom_test_subjects": int(len(np.unique(g_indom))),
        "n_dreamt_subjects": int(len(np.unique(g_dr))),
    }

    # HMM transitions fit once from training labels, reused across models
    T_hmm = pi_hmm = None
    if not args.no_hmm:
        T_hmm, pi_hmm = fit_hmm_transitions(y_train, g_train)
        print("\n  Fitted HMM transition matrix from train labels.")

    # Source class prior (used by --calibrate)
    p_source = class_prior(y_train)
    print(f"\n  Source class prior (train): " +
          " ".join(f"{n}={p_source[i]:.3f}" for i, n in enumerate(STAGE_NAMES)))

    # Container for saved predictions (filled per-model if --save-predictions)
    saved_preds = {
        "y_indomain": y_indom, "g_indomain": g_indom,
        "y_dreamt": y_dr, "g_dreamt": g_dr,
        "p_source": p_source,
    }

    # ── Train LightGBM once, evaluate on both test sets ────────────────────
    if "lightgbm" in args.models:
        print("\n" + "=" * 70)
        print("  LightGBM (combined train -> in-domain test, DREAMT zero-shot)")
        print("=" * 70)
        t0 = time.time()
        model, scaler, fit_elapsed = _fit_lightgbm(X_train, y_train)
        print(f"  Fit: {fit_elapsed:.1f}s")

        # In-domain
        m_in_raw, m_in_hmm, _, _, y_prob_in = evaluate_lightgbm(
            model, scaler, X_indom, y_indom, g_indom, s_indom,
            T_hmm=T_hmm, pi_hmm=pi_hmm,
        )
        print_metrics_block("In-domain test (combined held-out, raw)", m_in_raw)
        if m_in_hmm is not None:
            print_metrics_block("In-domain test (combined held-out, HMM-smoothed)", m_in_hmm)

        # DREAMT zero-shot
        m_dr_raw, m_dr_hmm, _, _, y_prob_dr = evaluate_lightgbm(
            model, scaler, X_dr_z, y_dr, g_dr, s_dr,
            T_hmm=T_hmm, pi_hmm=pi_hmm,
        )
        print_metrics_block("DREAMT zero-shot (raw)", m_dr_raw)
        if m_dr_hmm is not None:
            print_metrics_block("DREAMT zero-shot (HMM-smoothed)", m_dr_hmm)

        if args.save_predictions:
            saved_preds["y_prob_indomain_lgbm"] = y_prob_in
            saved_preds["y_prob_dreamt_lgbm"] = y_prob_dr

        # ── Domain-shift summary ────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("  Domain-shift summary (combined held-out -> DREAMT)")
        print("-" * 70)

        def shift(get, in_m, dr_m):
            i, d = get(in_m), get(dr_m)
            return f"{i:.4f}  ->  {d:.4f}  (Δ {d - i:+.4f})"

        macro_f1 = lambda m: float(np.mean(list(m["per_class_f1"].values())))
        for label, (im, dm) in (("raw      ", (m_in_raw, m_dr_raw)),
                                  ("HMM-smooth", (m_in_hmm, m_dr_hmm))):
            if im is None or dm is None:
                continue
            print(f"  {label}  κ:        {shift(lambda m: m['cohen_kappa'], im, dm)}")
            print(f"  {label}  acc:      {shift(lambda m: m['accuracy'],    im, dm)}")
            print(f"  {label}  macro_f1: {shift(macro_f1,                    im, dm)}")
            print(f"  {label}  N3 F1:    {shift(lambda m: m['n3_f1'],        im, dm)}")

        lgbm_results = {
            "fit_seconds": float(fit_elapsed),
            "indomain_raw": m_in_raw,
            "indomain_hmm": m_in_hmm,
            "dreamt_raw": m_dr_raw,
            "dreamt_hmm": m_dr_hmm,
        }

        # Prior-shift calibration on DREAMT
        if args.calibrate:
            print("\n" + "-" * 70)
            print(f"  LGBM prior-shift calibration ({args.n_cal_subj} cal / "
                  f"{100 - args.n_cal_subj} test DREAMT subjects)")
            print("-" * 70)
            cal = evaluate_with_calibration(
                y_prob_dr, y_dr, g_dr, s_dr, p_source,
                n_cal_subj=args.n_cal_subj, T_hmm=T_hmm, pi_hmm=pi_hmm,
            )
            p_t = np.array(cal["p_target"])
            print(f"  Cal slice priors: " +
                  " ".join(f"{n}={p_t[i]:.3f}" for i, n in enumerate(STAGE_NAMES)))
            print(f"  Ratio (target/source): " +
                  " ".join(f"{n}={p_t[i]/p_source[i]:.2f}x"
                           for i, n in enumerate(STAGE_NAMES)))
            print_metrics_block("Test-90 (UNcalibrated, raw)", cal["uncalibrated_raw"])
            if cal["uncalibrated_hmm"]:
                print_metrics_block("Test-90 (UNcalibrated, HMM)", cal["uncalibrated_hmm"])
            print_metrics_block("Test-90 (CALIBRATED, raw)", cal["calibrated_raw"])
            if cal["calibrated_hmm"]:
                print_metrics_block("Test-90 (CALIBRATED, HMM)", cal["calibrated_hmm"])

            print("\n  Calibration deltas (test-90):")
            for label, un_m, cal_m in (
                ("raw      ", cal["uncalibrated_raw"], cal["calibrated_raw"]),
                ("HMM-smooth", cal["uncalibrated_hmm"], cal["calibrated_hmm"]),
            ):
                if un_m is None or cal_m is None:
                    continue
                dk = cal_m["cohen_kappa"] - un_m["cohen_kappa"]
                da = cal_m["accuracy"] - un_m["accuracy"]
                df = (np.mean(list(cal_m["per_class_f1"].values())) -
                      np.mean(list(un_m["per_class_f1"].values())))
                print(f"    {label}  Δκ={dk:+.4f}  Δacc={da:+.4f}  Δmacro_f1={df:+.4f}")

            lgbm_results["calibration"] = cal

        all_results["lightgbm"] = lgbm_results
        print(f"\n  Total LGBM time: {time.time() - t0:.1f}s")

    # ── Train Bi-LSTM once, evaluate on both test sets ─────────────────────
    if "bi-lstm" in args.models:
        print("\n" + "=" * 70)
        print("  Bi-LSTM (combined train -> in-domain test, DREAMT zero-shot)")
        print("=" * 70)
        t0 = time.time()
        bi_out = fit_bilstm_and_predict(
            X_train, y_train, g_train,
            test_sets=[
                ("indomain", X_indom, y_indom, g_indom, s_indom),
                ("dreamt", X_dr_z, y_dr, g_dr, s_dr),
            ],
            seq_len=args.bilstm_seq_len,
            lstm_units=args.bilstm_units,
        )
        print(f"  Fit: {bi_out['fit_seconds']:.1f}s")

        def _eval_with_optional_hmm(y_prob, y_te, g_te, s_te):
            y_pred_raw = y_prob.argmax(axis=1)
            m_raw = compute_metrics(y_te, y_pred_raw, y_prob=y_prob, groups=g_te)
            if s_te is not None:
                m_raw["per_site_kappa"] = compute_per_site_kappa(y_te, y_pred_raw, s_te)
            m_hmm = None
            if T_hmm is not None:
                y_pred_hmm = viterbi_smooth(y_prob, g_te, T_hmm, pi_hmm)
                m_hmm = compute_metrics(y_te, y_pred_hmm, y_prob=y_prob, groups=g_te)
                if s_te is not None:
                    m_hmm["per_site_kappa"] = compute_per_site_kappa(y_te, y_pred_hmm, s_te)
            return m_raw, m_hmm

        bi_in_raw, bi_in_hmm = _eval_with_optional_hmm(
            bi_out["indomain"], y_indom, g_indom, s_indom)
        bi_dr_raw, bi_dr_hmm = _eval_with_optional_hmm(
            bi_out["dreamt"], y_dr, g_dr, s_dr)

        print_metrics_block("Bi-LSTM in-domain (raw)", bi_in_raw)
        if bi_in_hmm is not None:
            print_metrics_block("Bi-LSTM in-domain (HMM-smoothed)", bi_in_hmm)
        print_metrics_block("Bi-LSTM DREAMT zero-shot (raw)", bi_dr_raw)
        if bi_dr_hmm is not None:
            print_metrics_block("Bi-LSTM DREAMT zero-shot (HMM-smoothed)", bi_dr_hmm)

        print("\n" + "-" * 70)
        print("  Bi-LSTM domain-shift summary (combined held-out -> DREAMT)")
        print("-" * 70)

        def shift(get, in_m, dr_m):
            i, d = get(in_m), get(dr_m)
            return f"{i:.4f}  ->  {d:.4f}  (Δ {d - i:+.4f})"

        macro_f1 = lambda m: float(np.mean(list(m["per_class_f1"].values())))
        for label, (im, dm) in (("raw      ", (bi_in_raw, bi_dr_raw)),
                                  ("HMM-smooth", (bi_in_hmm, bi_dr_hmm))):
            if im is None or dm is None:
                continue
            print(f"  {label}  κ:        {shift(lambda m: m['cohen_kappa'], im, dm)}")
            print(f"  {label}  acc:      {shift(lambda m: m['accuracy'],    im, dm)}")
            print(f"  {label}  macro_f1: {shift(macro_f1,                    im, dm)}")
            print(f"  {label}  N3 F1:    {shift(lambda m: m['n3_f1'],        im, dm)}")

        bi_results = {
            "fit_seconds": bi_out["fit_seconds"],
            "indomain_raw": bi_in_raw,
            "indomain_hmm": bi_in_hmm,
            "dreamt_raw": bi_dr_raw,
            "dreamt_hmm": bi_dr_hmm,
        }

        if args.save_predictions:
            saved_preds["y_prob_indomain_bilstm"] = bi_out["indomain"]
            saved_preds["y_prob_dreamt_bilstm"] = bi_out["dreamt"]

        if args.calibrate:
            print("\n" + "-" * 70)
            print(f"  Bi-LSTM prior-shift calibration ({args.n_cal_subj} cal / "
                  f"{100 - args.n_cal_subj} test DREAMT subjects)")
            print("-" * 70)
            cal = evaluate_with_calibration(
                bi_out["dreamt"], y_dr, g_dr, s_dr, p_source,
                n_cal_subj=args.n_cal_subj, T_hmm=T_hmm, pi_hmm=pi_hmm,
            )
            p_t = np.array(cal["p_target"])
            print(f"  Cal slice priors: " +
                  " ".join(f"{n}={p_t[i]:.3f}" for i, n in enumerate(STAGE_NAMES)))
            print(f"  Ratio (target/source): " +
                  " ".join(f"{n}={p_t[i]/p_source[i]:.2f}x"
                           for i, n in enumerate(STAGE_NAMES)))
            print_metrics_block("Test-90 (UNcalibrated, raw)", cal["uncalibrated_raw"])
            if cal["uncalibrated_hmm"]:
                print_metrics_block("Test-90 (UNcalibrated, HMM)", cal["uncalibrated_hmm"])
            print_metrics_block("Test-90 (CALIBRATED, raw)", cal["calibrated_raw"])
            if cal["calibrated_hmm"]:
                print_metrics_block("Test-90 (CALIBRATED, HMM)", cal["calibrated_hmm"])

            print("\n  Calibration deltas (test-90):")
            for label, un_m, cal_m in (
                ("raw      ", cal["uncalibrated_raw"], cal["calibrated_raw"]),
                ("HMM-smooth", cal["uncalibrated_hmm"], cal["calibrated_hmm"]),
            ):
                if un_m is None or cal_m is None:
                    continue
                dk = cal_m["cohen_kappa"] - un_m["cohen_kappa"]
                da = cal_m["accuracy"] - un_m["accuracy"]
                df = (np.mean(list(cal_m["per_class_f1"].values())) -
                      np.mean(list(un_m["per_class_f1"].values())))
                print(f"    {label}  Δκ={dk:+.4f}  Δacc={da:+.4f}  Δmacro_f1={df:+.4f}")

            bi_results["calibration"] = cal

        all_results["bi_lstm"] = bi_results
        print(f"\n  Total Bi-LSTM wall time: {time.time() - t0:.1f}s")

        # ── Calibration cohort size sweep ───────────────────────────────────
        if args.ft_sweep_sizes:
            from tensorflow import keras as _keras

            sweep_sizes = sorted(int(s) for s in args.ft_sweep_sizes.split(","))
            n_test = args.ft_sweep_test_subj
            cal_pool_max = 100 - n_test
            if max(sweep_sizes) > cal_pool_max:
                raise ValueError(f"max sweep size {max(sweep_sizes)} > cal pool "
                                 f"({cal_pool_max} = 100 - {n_test} test)")

            print("\n" + "=" * 70)
            print(f"  CALIBRATION COHORT SIZE SWEEP")
            print(f"  Fixed test set: {n_test} DREAMT subjects (last {n_test} of "
                  f"SEED-shuffled order)")
            print(f"  Cal pool: {cal_pool_max} subjects, sweep sizes: {sweep_sizes}")
            print("=" * 70)

            test_mask, cal_pool, test_subjects = split_dreamt_for_sweep(
                g_dr, n_test, seed=SEED)
            X_te_dr = X_dr_z[test_mask]
            y_te_dr = y_dr[test_mask]
            g_te_dr = g_dr[test_mask]
            s_te_dr = s_dr[test_mask] if s_dr is not None else None
            print(f"  Test epochs: {len(X_te_dr):,}")

            # Base predictions on the fixed test set (slice from full-DREAMT preds)
            base_test_prob = bi_out["dreamt"][test_mask]
            base_raw, base_hmm = _eval_with_optional_hmm(
                base_test_prob, y_te_dr, g_te_dr, s_te_dr)
            print_metrics_block(
                f"BASE (zero-shot) on test-{n_test} (raw)", base_raw)
            if base_hmm is not None:
                print_metrics_block(
                    f"BASE (zero-shot) on test-{n_test} (HMM)", base_hmm)

            base_model = bi_out["model"]
            scaler = bi_out["scaler"]
            seq_len = bi_out["seq_len"]
            base_weights = base_model.get_weights()

            sweep_results = []
            for cal_size in sweep_sizes:
                cal_subj = set(cal_pool[:cal_size])
                cal_mask = np.isin(g_dr, list(cal_subj))
                X_cal = X_dr_z[cal_mask]
                y_cal = y_dr[cal_mask]
                g_cal = g_dr[cal_mask]

                print(f"\n  --- cal_size={cal_size} "
                      f"(cal epochs: {len(X_cal):,}) ---")

                # Fresh model from base weights (don't accumulate fine-tunes)
                ft_model = _keras.models.clone_model(base_model)
                ft_model.set_weights(base_weights)

                ft_model, ft_elapsed = fine_tune_bilstm(
                    ft_model, scaler, X_cal, y_cal, g_cal,
                    seq_len=seq_len,
                    ft_epochs=args.ft_epochs, ft_lr=args.ft_lr,
                )
                ft_test_prob = _predict_bilstm_flat(
                    ft_model, scaler, X_te_dr, g_te_dr, seq_len)
                ft_raw, ft_hmm = _eval_with_optional_hmm(
                    ft_test_prob, y_te_dr, g_te_dr, s_te_dr)

                sweep_results.append({
                    "cal_size": cal_size,
                    "ft_seconds": float(ft_elapsed),
                    "test_raw": ft_raw,
                    "test_hmm": ft_hmm,
                    "cal_subjects": sorted(map(str, cal_subj)),
                })

                print(f"    cal={cal_size:3d}  "
                      f"κ_raw={ft_raw['cohen_kappa']:.4f}  "
                      f"κ_HMM={ft_hmm['cohen_kappa']:.4f}  "
                      f"acc_HMM={ft_hmm['accuracy']:.4f}  "
                      f"N3_F1={ft_hmm['n3_f1']:.3f}  "
                      f"AUROC={ft_hmm['auroc_macro']:.3f}")

            # ── Scaling-curve summary table ────────────────────────────────
            print("\n" + "-" * 70)
            print(f"  SCALING CURVE (test-{n_test} fixed, base κ_HMM={base_hmm['cohen_kappa']:.4f})")
            print("-" * 70)
            print(f"  {'cal_size':>8}  {'κ_raw':>8}  {'κ_HMM':>8}  {'Δκ_HMM':>8}  "
                  f"{'acc_HMM':>8}  {'macroF1':>8}  {'N3_F1':>7}  {'AUROC':>7}")
            for r in sweep_results:
                d_kappa = r["test_hmm"]["cohen_kappa"] - base_hmm["cohen_kappa"]
                mf1 = float(np.mean(list(r["test_hmm"]["per_class_f1"].values())))
                print(f"  {r['cal_size']:>8d}  "
                      f"{r['test_raw']['cohen_kappa']:>8.4f}  "
                      f"{r['test_hmm']['cohen_kappa']:>8.4f}  "
                      f"{d_kappa:>+8.4f}  "
                      f"{r['test_hmm']['accuracy']:>8.4f}  "
                      f"{mf1:>8.4f}  "
                      f"{r['test_hmm']['n3_f1']:>7.3f}  "
                      f"{r['test_hmm']['auroc_macro']:>7.3f}")

            if args.save_predictions:
                saved_preds["sweep_test_mask"] = test_mask

            all_results["bi_lstm_finetune_sweep"] = {
                "config": {
                    "sweep_sizes": sweep_sizes,
                    "n_test_subj": int(len(test_subjects)),
                    "ft_epochs": args.ft_epochs,
                    "ft_lr": args.ft_lr,
                    "seq_len": seq_len,
                },
                "base_test_raw": base_raw,
                "base_test_hmm": base_hmm,
                "sweep": sweep_results,
            }

        # ── Few-shot fine-tune (calibration cohort) ─────────────────────────
        if args.fine_tune and not args.ft_sweep_sizes:
            print("\n" + "=" * 70)
            print(f"  Bi-LSTM FINE-TUNE on {args.ft_cal_subj} labeled DREAMT subjects")
            print(f"  Test set: remaining {100 - args.ft_cal_subj} DREAMT subjects.")
            print("=" * 70)

            cal_mask, test_mask, cal_subj, test_subj = split_dreamt_for_calibration(
                g_dr, args.ft_cal_subj, seed=SEED)
            print(f"  Cal subjects ({len(cal_subj)}): {sorted(cal_subj)[:5]}... "
                  f"Test subjects: {len(test_subj)}")

            X_cal_dr = X_dr_z[cal_mask]
            y_cal_dr = y_dr[cal_mask]
            g_cal_dr = g_dr[cal_mask]
            X_te_dr = X_dr_z[test_mask]
            y_te_dr = y_dr[test_mask]
            g_te_dr = g_dr[test_mask]
            s_te_dr = s_dr[test_mask] if s_dr is not None else None
            print(f"  Cal epochs: {len(X_cal_dr):,}  Test epochs: {len(X_te_dr):,}")

            # Base predictions on the same 90 test subjects (already have these
            # from full-DREAMT prediction — slice and re-evaluate)
            base_test_prob = bi_out["dreamt"][test_mask]
            base_test_raw, base_test_hmm = _eval_with_optional_hmm(
                base_test_prob, y_te_dr, g_te_dr, s_te_dr)
            print_metrics_block(
                f"BASE (zero-shot) on test-{len(test_subj)} (raw)", base_test_raw)
            if base_test_hmm is not None:
                print_metrics_block(
                    f"BASE (zero-shot) on test-{len(test_subj)} (HMM)", base_test_hmm)

            # Fine-tune the trained Bi-LSTM on cal subjects (model is mutated)
            ft_model = bi_out["model"]
            ft_scaler = bi_out["scaler"]
            ft_seq_len = bi_out["seq_len"]
            ft_model, ft_elapsed = fine_tune_bilstm(
                ft_model, ft_scaler,
                X_cal_dr, y_cal_dr, g_cal_dr,
                seq_len=ft_seq_len,
                ft_epochs=args.ft_epochs, ft_lr=args.ft_lr,
            )
            print(f"  Fine-tune: {ft_elapsed:.1f}s")

            # Predict on test subjects with fine-tuned model
            ft_test_prob = _predict_bilstm_flat(
                ft_model, ft_scaler, X_te_dr, g_te_dr, ft_seq_len)
            ft_test_raw, ft_test_hmm = _eval_with_optional_hmm(
                ft_test_prob, y_te_dr, g_te_dr, s_te_dr)
            print_metrics_block(
                f"FINE-TUNED on test-{len(test_subj)} (raw)", ft_test_raw)
            if ft_test_hmm is not None:
                print_metrics_block(
                    f"FINE-TUNED on test-{len(test_subj)} (HMM)", ft_test_hmm)

            # Comparison
            print(f"\n  Few-shot Δ vs base (same {len(test_subj)} test subjects):")

            def _ft_shift(label, base_m, ft_m):
                if base_m is None or ft_m is None:
                    return
                d_kappa = ft_m["cohen_kappa"] - base_m["cohen_kappa"]
                d_acc = ft_m["accuracy"] - base_m["accuracy"]
                bf1 = float(np.mean(list(base_m["per_class_f1"].values())))
                ff1 = float(np.mean(list(ft_m["per_class_f1"].values())))
                d_n3 = ft_m["n3_f1"] - base_m["n3_f1"]
                print(f"    {label}  Δκ={d_kappa:+.4f}  Δacc={d_acc:+.4f}  "
                      f"Δmacro_f1={ff1 - bf1:+.4f}  ΔN3_F1={d_n3:+.4f}")

            _ft_shift("raw      ", base_test_raw, ft_test_raw)
            _ft_shift("HMM-smooth", base_test_hmm, ft_test_hmm)

            if args.save_predictions:
                saved_preds["y_prob_dreamt_test_base_bilstm_ft"] = base_test_prob
                saved_preds["y_prob_dreamt_test_finetuned_bilstm"] = ft_test_prob
                saved_preds["test_mask_finetune"] = test_mask
                saved_preds["cal_mask_finetune"] = cal_mask

            all_results["bi_lstm_fine_tune"] = {
                "config": {
                    "ft_cal_subj": int(len(cal_subj)),
                    "ft_test_subj": int(len(test_subj)),
                    "ft_epochs": args.ft_epochs,
                    "ft_lr": args.ft_lr,
                    "seq_len": ft_seq_len,
                    "cal_subjects": sorted(map(str, cal_subj)),
                },
                "fine_tune_seconds": float(ft_elapsed),
                "base_test_raw": base_test_raw,
                "base_test_hmm": base_test_hmm,
                "finetuned_test_raw": ft_test_raw,
                "finetuned_test_hmm": ft_test_hmm,
            }

        # ── Self-training Bi-LSTM (transductive) ────────────────────────────
        if args.self_train:
            from cross_domain_eval import self_train_eval_bi_lstm

            print("\n" + "=" * 70)
            print(f"  Bi-LSTM SELF-TRAIN (combined + DREAMT pseudo-labels)")
            print(f"  Holds out {args.st_test_subj} DREAMT subjects as clean test set.")
            print("=" * 70)

            # Deterministic 90/10 subject-level split of DREAMT
            dreamt_subj = np.unique(g_dr)
            rng = np.random.RandomState(SEED + 1)  # different seed than calibration
            perm = rng.permutation(len(dreamt_subj))
            test_subj = set(dreamt_subj[perm[: args.st_test_subj]])
            poolval_subj = set(dreamt_subj[perm[args.st_test_subj:]])
            test_mask = np.isin(g_dr, list(test_subj))
            poolval_mask = ~test_mask
            print(f"  DREAMT split: {len(test_subj)} test, {len(poolval_subj)} pool+val "
                  f"(internal val_frac={args.st_target_val_frac:.3f} → "
                  f"~{int(args.st_target_val_frac * len(poolval_subj))} val / "
                  f"~{len(poolval_subj) - int(args.st_target_val_frac * len(poolval_subj))} pool)")

            # Same-test-set base: slice the seq_len=120 base Bi-LSTM probs.
            # Note seq_len mismatch with self-train (60); flagged in the printout.
            base_test10_prob = bi_out["dreamt"][test_mask]
            y_test10 = y_dr[test_mask]
            g_test10 = g_dr[test_mask]
            s_test10 = s_dr[test_mask]
            base_test10_raw, base_test10_hmm = _eval_with_optional_hmm(
                base_test10_prob, y_test10, g_test10, s_test10)
            print_metrics_block(
                f"Base Bi-LSTM, test-10 (seq_len={args.bilstm_seq_len}, raw)",
                base_test10_raw,
            )
            if base_test10_hmm is not None:
                print_metrics_block(
                    f"Base Bi-LSTM, test-10 (seq_len={args.bilstm_seq_len}, HMM)",
                    base_test10_hmm,
                )

            # Run self-train (internally fits iter-0 source-only, then iter-1 with pseudos).
            X_pv = X_dr_z[poolval_mask]
            y_pv = y_dr[poolval_mask]
            g_pv = g_dr[poolval_mask]
            X_te10 = X_dr_z[test_mask]

            t1 = time.time()
            st_metrics_raw, _, st_y_prob, T_st, pi_st = self_train_eval_bi_lstm(
                X_train, y_train, g_train,
                X_pv, y_pv, g_pv,
                X_te10, y_test10, g_test10, s_test10,
                target_val_frac=args.st_target_val_frac,
                quantile=args.st_pseudo_quantile,
                pseudo_weight=args.st_pseudo_weight,
                n_iters=args.st_iters,
                seq_len=args.st_seq_len,
                lstm_units=args.bilstm_units,
            )
            st_elapsed = time.time() - t1

            # HMM-smooth self-train predictions using its returned T_hmm/pi_hmm
            st_pred_hmm = viterbi_smooth(st_y_prob, g_test10, T_st, pi_st)
            st_metrics_hmm = compute_metrics(
                y_test10, st_pred_hmm, y_prob=st_y_prob, groups=g_test10)
            if s_test10 is not None:
                st_metrics_hmm["per_site_kappa"] = compute_per_site_kappa(
                    y_test10, st_pred_hmm, s_test10)

            print_metrics_block(
                f"Self-train Bi-LSTM, test-10 (seq_len={args.st_seq_len}, raw)",
                st_metrics_raw,
            )
            print_metrics_block(
                f"Self-train Bi-LSTM, test-10 (seq_len={args.st_seq_len}, HMM)",
                st_metrics_hmm,
            )

            # ── Side-by-side comparison ──────────────────────────────────────
            print("\n  Self-train Δ vs base on same 10 test subjects:")
            print(f"  (NOTE: base used seq_len={args.bilstm_seq_len}, "
                  f"self-train used seq_len={args.st_seq_len} — different temporal context)")

            def _shift_pair(label, base_m, st_m):
                if base_m is None or st_m is None:
                    return
                d_kappa = st_m["cohen_kappa"] - base_m["cohen_kappa"]
                d_acc = st_m["accuracy"] - base_m["accuracy"]
                bf1 = float(np.mean(list(base_m["per_class_f1"].values())))
                sf1 = float(np.mean(list(st_m["per_class_f1"].values())))
                d_n3 = st_m["n3_f1"] - base_m["n3_f1"]
                print(f"    {label}  Δκ={d_kappa:+.4f}  Δacc={d_acc:+.4f}  "
                      f"Δmacro_f1={sf1 - bf1:+.4f}  ΔN3_F1={d_n3:+.4f}")

            _shift_pair("raw      ", base_test10_raw, st_metrics_raw)
            _shift_pair("HMM-smooth", base_test10_hmm, st_metrics_hmm)

            if args.save_predictions:
                saved_preds["y_prob_dreamt_test10_base_bilstm"] = base_test10_prob
                saved_preds["y_prob_dreamt_test10_selftrain_bilstm"] = st_y_prob
                saved_preds["test_mask_selftrain"] = test_mask

            all_results["bi_lstm_selftrain"] = {
                "config": {
                    "pseudo_quantile": args.st_pseudo_quantile,
                    "pseudo_weight": args.st_pseudo_weight,
                    "iters": args.st_iters,
                    "seq_len": args.st_seq_len,
                    "target_val_frac": args.st_target_val_frac,
                    "n_test_subj": int(len(test_subj)),
                    "n_poolval_subj": int(len(poolval_subj)),
                    "test_subjects": sorted(map(str, test_subj)),
                },
                "fit_seconds": float(st_elapsed),
                "base_test10_raw": base_test10_raw,
                "base_test10_hmm": base_test10_hmm,
                "selftrain_test10_raw": st_metrics_raw,
                "selftrain_test10_hmm": st_metrics_hmm,
            }
            print(f"\n  Total self-train wall time: {st_elapsed:.1f}s")

    # ── Persist combined results ───────────────────────────────────────────
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Results: {out_path}")

    if args.save_predictions:
        npz_path = run_dir / "predictions.npz"
        np.savez_compressed(npz_path, **saved_preds)
        print(f"  Predictions: {npz_path}")


if __name__ == "__main__":
    main()
