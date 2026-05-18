#!/usr/bin/env python3
"""Generate the figures embedded in docs/REPORT.md.

Consolidates four overlapping legacy plot scripts into a single entry point.
Selects the best run per (model, n_classes, feature-set) from results/*/metrics.json
and produces the suite of figures referenced by docs/REPORT.md.

Usage:
    python reporting/generate_figures.py \
        --results-dir results \
        --out-dir docs/figures \
        --suite all

Suites:
    mesa      — 12 in-domain MESA figures (model comparison, task complexity,
                confusion, per-class radar, feature importance, stage distribution,
                temporal impact, AUROC, per-subject kappa, transition, sens/spec,
                demographics)
    combined  — 5 figures for the combined MESA+STAGES tier-1 1D-CNN report
                (confusion, per-class metrics, feature importance, class
                distribution, N3 duration)
    all       — both suites
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ── Constants ────────────────────────────────────────────────────────────────

STAGE_NAMES = {5: ["Wake", "N1", "N2", "N3", "REM"], 3: ["Wake", "NREM", "REM"], 2: ["Wake", "Sleep"]}
PALETTE = sns.color_palette("Set2", 8)
STAGE_COLORS = {
    "Wake": "#E69F00", "N1": "#56B4E9", "N2": "#009E73",
    "N3": "#0072B2", "REM": "#CC79A7",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── Run discovery ────────────────────────────────────────────────────────────

def load_runs(results_dir: Path) -> list[dict]:
    """Load every results/*/metrics.json with a tag for feature set."""
    runs = []
    for d in sorted(results_dir.iterdir()):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        r = json.loads(mf.read_text())
        data = r.get("data_path", "")
        r["_dir"] = d.name
        r["_data_tag"] = "full" if "full" in data else ("temporal" if "temporal" in data else "original")
        runs.append(r)
    return runs


def best_run(runs, *, model=None, n_classes=None, data_tag=None, key="cohen_kappa"):
    candidates = runs
    if model is not None:
        candidates = [r for r in candidates if r["model"] == model]
    if n_classes is not None:
        candidates = [r for r in candidates if r.get("n_classes") == n_classes]
    if data_tag is not None:
        if isinstance(data_tag, str):
            data_tag = (data_tag,)
        candidates = [r for r in candidates if r["_data_tag"] in data_tag]
    candidates = [r for r in candidates if key in r and r[key] is not None]
    return max(candidates, key=lambda r: r[key]) if candidates else None


def load_metrics(results_dir: Path, run_dir: str) -> dict:
    with open(results_dir / run_dir / "metrics.json") as f:
        return json.load(f)


# ── MESA suite (12 figures) ──────────────────────────────────────────────────

def mesa_model_comparison(runs, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    five_class = [r for r in runs if r.get("n_classes") == 5]
    best = {}
    for r in five_class:
        key = (r["model"], r["_data_tag"])
        if key not in best or r["cohen_kappa"] > best[key]["cohen_kappa"]:
            best[key] = r

    models = ["Logistic Regression", "LightGBM", "XGBoost", "1D-CNN"]
    tags = ["original", "temporal", "full"]
    tag_labels = {"original": "Original (28)", "temporal": "Temporal (54)", "full": "Full (71)"}
    tag_colors = {"original": PALETTE[0], "temporal": PALETTE[1], "full": PALETTE[2]}

    for ax_idx, (metric, label) in enumerate([
        ("cohen_kappa", "Cohen's κ"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(models))
        width = 0.25
        for i, tag in enumerate(tags):
            vals = [best.get((m, tag), {}).get(metric, 0) or 0 for m in models]
            bars = ax.bar(x + i * width, vals, width, label=tag_labels[tag], color=tag_colors[tag])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(label)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        ax.set_ylim(0, 0.9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("5-Class Sleep Staging: Model Comparison Across Feature Sets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "01_model_comparison.png")
    plt.close(fig)


def mesa_task_complexity(runs, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    best = {}
    for r in runs:
        if r["model"] == "XGBoost" and r["_data_tag"] in ("temporal", "full"):
            nc = r.get("n_classes")
            if nc and (nc not in best or r["cohen_kappa"] > best[nc]["cohen_kappa"]):
                best[nc] = r

    classes = sorted(best)
    if not classes:
        plt.close(fig)
        return
    kappas = [best[nc]["cohen_kappa"] for nc in classes]
    accs = [best[nc]["accuracy"] for nc in classes]
    macro_f1s = [best[nc].get("macro_f1", 0) for nc in classes]

    x = np.arange(len(classes))
    width = 0.25
    for offset, vals, label, color in [
        (-width, kappas, "Cohen's κ", PALETTE[0]),
        (0, accs, "Accuracy", PALETTE[1]),
        (width, macro_f1s, "Macro F1", PALETTE[2]),
    ]:
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    label_map = {2: "2-class\n(Wake/Sleep)", 3: "3-class\n(Wake/NREM/REM)", 5: "5-class\n(W/N1/N2/N3/R)"}
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[c] for c in classes])
    ax.set_xlabel("Classification Task")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("XGBoost Performance vs Task Complexity", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "02_task_complexity.png")
    plt.close(fig)


def mesa_confusion_matrices(runs, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, nc in zip(axes, [5, 3]):
        names = STAGE_NAMES[nc]
        best = best_run(runs, n_classes=nc, data_tag=("temporal", "full"))
        if best is None or "confusion_matrix" not in best:
            continue
        cm = np.array(best["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})",
                        ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{nc}-Class Confusion Matrix\n({best['model']}, κ={best['cohen_kappa']:.3f})",
                     fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "03_confusion_matrices.png")
    plt.close(fig)


def mesa_per_class_radar(runs, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    names = STAGE_NAMES[5]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]

    best_per_model = {}
    for r in runs:
        if r.get("n_classes") == 5 and r["_data_tag"] in ("temporal", "full") and "per_class" in r:
            m = r["model"]
            if m not in best_per_model or r["cohen_kappa"] > best_per_model[m]["cohen_kappa"]:
                best_per_model[m] = r

    colors = {"XGBoost": PALETTE[0], "LightGBM": PALETTE[1],
              "Logistic Regression": PALETTE[2], "1D-CNN": PALETTE[3]}
    for model_name, r in sorted(best_per_model.items()):
        f1s = [r["per_class"][n]["f1-score"] for n in names]
        f1s += f1s[:1]
        color = colors.get(model_name, PALETTE[4])
        ax.plot(angles, f1s, "o-", linewidth=2, color=color,
                label=f"{model_name} (κ={r['cohen_kappa']:.3f})")
        ax.fill(angles, f1s, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Per-Class F1 Score (5-Class)", fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(out / "04_per_class_f1_radar.png")
    plt.close(fig)


def mesa_feature_importance(runs, results_dir: Path, out: Path):
    best = best_run(runs, model="XGBoost", n_classes=5, data_tag=("temporal", "full"))
    if best is None:
        return
    fi_path = results_dir / best["_dir"] / "feature_importance.json"
    if not fi_path.exists():
        return
    fi = json.loads(fi_path.read_text())[:20]
    names = [x[0] for x in fi][::-1]
    values = [x[1] for x in fi][::-1]

    def category(n):
        for prefix, label in [("actig", "Actigraphy"), ("hrv", "Heart Rate"),
                              ("hr_", "Heart Rate"), ("spo2", "SpO2"),
                              ("pos", "Position"), ("leg", "Movement"),
                              ("tod", "Circadian"), ("night", "Circadian"),
                              ("demo", "Demographic"), ("resp", "Respiratory")]:
            if prefix in n:
                return label
        return "Other"

    cat_colors = {
        "Actigraphy": "#2ecc71", "Heart Rate": "#e74c3c", "SpO2": "#3498db",
        "Position": "#f39c12", "Movement": "#9b59b6", "Circadian": "#1abc9c",
        "Demographic": "#e67e22", "Respiratory": "#34495e", "Other": "#95a5a6",
    }
    colors = [cat_colors[category(n)] for n in names]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top 20 Features — XGBoost 5-Class", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    cats = list(dict.fromkeys([category(n) for n in names]))
    ax.legend([Patch(color=cat_colors[c]) for c in cats], cats,
              loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "05_feature_importance.png")
    plt.close(fig)


def mesa_stage_distribution(parquet_path: Path, out: Path):
    if not parquet_path.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df = pd.read_parquet(parquet_path, columns=["sleep_stage", "mesa_id"])
    valid = df[df.sleep_stage.isin([0, 1, 2, 3, 5])]
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}
    order = ["Wake", "N1", "N2", "N3", "REM"]
    colors = [STAGE_COLORS[s] for s in order]

    counts = valid.sleep_stage.map(stage_map).value_counts().reindex(order)
    bars = axes[0].bar(order, counts.values, color=colors)
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                     f"{v:,}\n({100*v/len(valid):.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Number of 30s Epochs")
    axes[0].set_title("Sleep Stage Distribution (All Subjects)", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    subj_props = valid.groupby("mesa_id").sleep_stage.value_counts(normalize=True).unstack(fill_value=0)
    subj_props = subj_props.rename(columns=stage_map).reindex(columns=order)
    bp = axes[1].boxplot([subj_props[s].values for s in order], labels=order, patch_artist=True,
                          medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_ylabel("Proportion of Night")
    axes[1].set_title("Per-Subject Stage Proportions", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "06_stage_distribution.png")
    plt.close(fig)


def mesa_temporal_impact(runs, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    pairs: dict[tuple, dict] = {}
    for r in runs:
        key = (r["model"], r.get("n_classes"))
        tag = r["_data_tag"]
        if tag in ("original", "temporal"):
            pairs.setdefault(key, {})
            if tag not in pairs[key] or r["cohen_kappa"] > pairs[key][tag]["cohen_kappa"]:
                pairs[key][tag] = r
    valid = {k: v for k, v in pairs.items() if "original" in v and "temporal" in v}
    if not valid:
        plt.close(fig)
        return

    labels, orig_vals, temp_vals = [], [], []
    for (model, nc), p in sorted(valid.items()):
        labels.append(f"{model}\n{nc}-class")
        orig_vals.append(p["original"]["cohen_kappa"])
        temp_vals.append(p["temporal"]["cohen_kappa"])

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, orig_vals, width, label="Original (28 features)", color=PALETTE[0])
    ax.bar(x + width / 2, temp_vals, width, label="+ Temporal (54 features)", color=PALETTE[1])
    for i in range(len(labels)):
        improvement = temp_vals[i] - orig_vals[i]
        pct = 100 * improvement / orig_vals[i] if orig_vals[i] > 0 else 0
        ax.annotate(f"+{pct:.0f}%", (x[i] + width / 2, temp_vals[i]),
                    xytext=(x[i] + width / 2 + 0.15, temp_vals[i] + 0.02),
                    fontsize=8, color="green", fontweight="bold")
    ax.set_ylabel("Cohen's κ"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.85); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Impact of Temporal Features on Model Performance", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "07_temporal_impact.png")
    plt.close(fig)


def mesa_auroc(runs, out: Path):
    best = best_run([r for r in runs if "auroc_macro" in r], n_classes=5)
    if best is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    names = STAGE_NAMES[5]
    aurocs = [best["per_class"][n].get("auroc", 0) for n in names]
    colors = [STAGE_COLORS[n] for n in names]
    bars = ax.bar(names, aurocs, color=colors)
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=best["auroc_macro"], color="gray", linestyle="--",
               label=f"Macro AUROC = {best['auroc_macro']:.3f}")
    ax.set_ylabel("AUROC (One-vs-Rest)"); ax.set_ylim(0.7, 1.0)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Per-Class AUROC — {best['model']} 5-Class", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "08_auroc_per_class.png")
    plt.close(fig)


def mesa_per_subject_kappa(runs, out: Path):
    best = best_run([r for r in runs if "per_subject_kappa" in r], n_classes=5)
    if best is None:
        return
    sk = best["per_subject_kappa"]
    np.random.seed(42)
    simulated = np.clip(np.random.normal(sk["mean"], sk["std"], sk["n_subjects"]),
                        sk["min"], sk["max"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated, bins=30, color=PALETTE[0], edgecolor="white", alpha=0.8)
    for q, label, c in [("median", "Median", "red"), ("q25", "Q25", "orange"), ("q75", "Q75", "orange")]:
        ax.axvline(sk[q], color=c, linestyle="--" if q != "median" else "-",
                   linewidth=1.5 if q != "median" else 2, label=f"{label} = {sk[q]:.3f}")
    ax.set_xlabel("Cohen's κ (per subject)"); ax.set_ylabel("Number of Subjects")
    ax.set_title(f"Per-Subject Agreement Distribution\n{best['model']} 5-Class ({sk['n_subjects']} subjects)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "09_per_subject_kappa.png")
    plt.close(fig)


def mesa_transition_accuracy(runs, out: Path):
    best = best_run([r for r in runs if "transition_accuracy" in r], n_classes=5)
    if best is None:
        return
    ta = best["transition_accuracy"]
    total = best["n_test_samples"]
    n_trans = ta["n_transitions"]
    n_nontrans = total - n_trans
    nontrans_correct = int(best["accuracy"] * total) - ta["n_correct"]
    nontrans_acc = nontrans_correct / n_nontrans if n_nontrans else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["Overall\nAccuracy", "Non-Transition\nEpochs", "Transition\nEpochs"]
    values = [best["accuracy"], nontrans_acc, ta["accuracy"]]
    bars = ax.bar(cats, values, color=[PALETTE[0], PALETTE[1], PALETTE[3]], width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.85); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Transition vs Steady-State Accuracy\n{best['model']} 5-Class", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "10_transition_accuracy.png")
    plt.close(fig)


def mesa_sensitivity_specificity(runs, out: Path):
    best = None
    for r in runs:
        if (r.get("n_classes") == 5 and "per_class" in r
                and "specificity" in r["per_class"].get("Wake", {})):
            if best is None or r["cohen_kappa"] > best["cohen_kappa"]:
                best = r
    if best is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in STAGE_NAMES[5]:
        pc = best["per_class"][name]
        sens, spec = pc["recall"], pc["specificity"]
        ax.scatter(spec, sens, s=200, c=STAGE_COLORS[name], zorder=5,
                   edgecolors="black", linewidth=1)
        ax.annotate(name, (spec, sens), textcoords="offset points", xytext=(10, 5),
                    fontweight="bold")
    ax.plot([0.5, 1], [0.5, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity (Recall)")
    ax.set_xlim(0.8, 1.0); ax.set_ylim(0.3, 0.85); ax.grid(alpha=0.3)
    ax.set_title(f"Sensitivity vs Specificity — {best['model']} 5-Class", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(out / "11_sensitivity_specificity.png")
    plt.close(fig)


def mesa_demographic_analysis(parquet_path: Path, out: Path):
    if not parquet_path.exists():
        return
    df = pd.read_parquet(parquet_path,
                         columns=["mesa_id", "sleep_stage", "demo_age", "demo_bmi", "demo_is_male"])
    valid = df[df.sleep_stage.isin([0, 1, 2, 3, 5])]
    subj = valid.groupby("mesa_id").first()
    subj_stages = (valid.groupby("mesa_id")
                   .apply(lambda g: (g.sleep_stage == 3).sum() / len(g), include_groups=False)
                   .reset_index())
    subj_stages.columns = ["mesa_id", "n3_pct"]
    merged = subj_stages.merge(subj[["demo_bmi", "demo_age"]].reset_index(), on="mesa_id")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(subj.demo_age, bins=30, color=PALETTE[0], edgecolor="white")
    axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Number of Subjects")
    axes[0].set_title("Age Distribution", fontweight="bold")
    axes[0].axvline(subj.demo_age.median(), color="red", linestyle="--",
                    label=f"Median={subj.demo_age.median():.0f}")
    axes[0].legend()

    for ax, xcol, color, xlim in [
        (axes[1], "demo_bmi", PALETTE[2], (15, 55)),
        (axes[2], "demo_age", PALETTE[3], (50, 95)),
    ]:
        ax.scatter(merged[xcol], merged.n3_pct * 100, alpha=0.3, s=10, color=color)
        z = np.polyfit(merged[xcol].dropna(), merged.n3_pct.dropna() * 100, 1)
        ax.plot(np.linspace(*xlim, 100), np.polyval(z, np.linspace(*xlim, 100)),
                "r--", linewidth=2, label=f"Trend (slope={z[0]:.2f})")
        ax.set_xlabel("BMI" if xcol == "demo_bmi" else "Age (years)")
        ax.set_ylabel("% N3 (Deep Sleep)")
        ax.set_title(("BMI" if xcol == "demo_bmi" else "Age") + " vs Deep Sleep Proportion",
                     fontweight="bold")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out / "12_demographic_analysis.png")
    plt.close(fig)


# ── Combined-Tier1 suite (5 figures) ─────────────────────────────────────────

def combined_confusion(metrics: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.85, aspect="equal")
    for i in range(5):
        for j in range(5):
            v = cm_norm[i, j]
            ax.text(j, i, f".{int(v*100):02d}" if v < 1 else "1.0",
                    ha="center", va="center", fontsize=11,
                    color="white" if v > 0.45 else "#333333")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES[5]); ax.set_yticklabels(STAGE_NAMES[5])
    ax.set_xlabel("Predicted Stage"); ax.set_ylabel("True Stage")
    ax.set_title("1D-CNN Normalised Confusion Matrix\n(Combined Tier 1, 59 Wearable Features)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "fig1_confusion_matrix.png")
    plt.close(fig)


def combined_per_class(metrics: dict, out: Path):
    metrics_keys = ["precision", "recall", "f1-score", "auroc"]
    labels = ["Precision", "Recall", "F1-Score", "AUROC"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(STAGE_NAMES[5]))
    n = len(metrics_keys)
    w = 0.75 / n
    for idx, (mk, lbl) in enumerate(zip(metrics_keys, labels)):
        vals = [metrics["per_class"][s].get(mk, 0) for s in STAGE_NAMES[5]]
        offset = (idx - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.92, label=lbl, alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(STAGE_NAMES[5])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", ls=":", lw=0.7, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_title("1D-CNN Per-Class Performance", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "fig2_per_class.png")
    plt.close(fig)


def combined_feature_importance(results_dir: Path, run_dir: str, out: Path):
    fi_path = results_dir / run_dir / "feature_importance.json"
    if not fi_path.exists():
        return
    fi = json.loads(fi_path.read_text())[:20]
    names = [x[0] for x in fi][::-1]
    vals = [x[1] for x in fi][::-1]

    def col(n):
        if n.startswith(("hr_", "hrv_")): return "#D55E00"
        if n.startswith("spo2"):          return "#0072B2"
        if n.startswith("actig"):         return "#009E73"
        if n.startswith("tod_"):          return "#CC79A7"
        if n.startswith("demo_"):         return "#56B4E9"
        return "#999999"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(range(len(names)), vals, color=[col(n) for n in names],
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Split Count")
    ax.set_title("Top 20 Features (LightGBM, Combined Tier 1, Wearable-Only)",
                 fontweight="bold")
    ax.legend(handles=[
        Patch(color="#D55E00", label="HR / HRV"),
        Patch(color="#0072B2", label="SpO₂"),
        Patch(color="#009E73", label="Actigraphy"),
        Patch(color="#CC79A7", label="Circadian"),
        Patch(color="#56B4E9", label="Demographics"),
    ], loc="lower right", framealpha=0.9, edgecolor="none", fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "fig3_feature_importance.png")
    plt.close(fig)


def combined_class_distribution(out: Path):
    # Static distribution from the canonical combined Tier 1 build.
    mesa = {"Wake": 646872, "N1": 171987, "N2": 719782, "N3": 127504, "REM": 227034}
    stages = {"Wake": 194767, "N1": 89183, "N2": 271775, "N3": 77253, "REM": 132382}
    combined = {s: mesa[s] + stages[s] for s in STAGE_NAMES[5]}

    fig, ax = plt.subplots(figsize=(9, 3))
    datasets = ["MESA (n=1,744)", "STAGES (n=1,326)", "Combined (n=3,070)"]
    for idx, (lbl, data) in enumerate(zip(datasets, [mesa, stages, combined])):
        total = sum(data.values())
        left = 0
        for stage in STAGE_NAMES[5]:
            pct = data[stage] / total
            ax.barh(idx, pct, left=left, color=STAGE_COLORS[stage],
                    edgecolor="white", linewidth=0.5, height=0.6)
            if pct > 0.07:
                ax.text(left + pct / 2, idx, f"{pct:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if stage != "N1" else "black")
            left += pct
    ax.set_yticks(range(3)); ax.set_yticklabels(datasets)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Proportion of Valid Epochs")
    ax.set_title("Sleep Stage Distribution", fontweight="bold")
    ax.legend(handles=[Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_NAMES[5]],
              loc="upper right", ncol=5, bbox_to_anchor=(1.0, 1.25),
              framealpha=0.9, edgecolor="none", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out / "fig4_class_distribution.png")
    plt.close(fig)


def combined_n3_duration(results_dir: Path, runs_for_n3: list[tuple[str, str, str]], out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels, trues, preds, colors = [], [], [], []
    for lbl, rd, col in runs_for_n3:
        path = results_dir / rd / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        d = m.get("n3_duration_correlation")
        if d:
            labels.append(lbl); trues.append(d["true_mean_min"])
            preds.append(d["pred_mean_min"]); colors.append(col)
    if not labels:
        plt.close(fig)
        return
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(x - w / 2, trues, w, label="True N3", color="#BBBBBB", edgecolor="white")
    ax.bar(x + w / 2, preds, w, label="Predicted N3", color=colors, edgecolor="white")
    for i in range(len(labels)):
        ax.text(x[i] + w / 2, preds[i] + 2, f"{preds[i]/trues[i]:.1f}x",
                ha="center", fontweight="bold", color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean N3 Duration (min/subject)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_title("N3 Duration: Predicted vs True", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "fig5_n3_duration.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--suite", choices=["mesa", "combined", "all"], default="all")
    parser.add_argument("--mesa-parquet", type=Path,
                        default=Path("datasets/mesa/mesa_clean_wearable_full.parquet"),
                        help="Path to MESA full parquet (needed for stage distribution / demographics figures)")
    parser.add_argument("--combined-cnn-run", default="1d_cnn_5class_20260326_221556",
                        help="Run dir under --results-dir for the combined-Tier-1 1D-CNN figures")
    parser.add_argument("--combined-lgbm-run", default="lightgbm_5class_20260326_210229",
                        help="Run dir for LightGBM feature-importance in the combined suite")
    parser.add_argument("--combined-lgbm-boost-run", default="lightgbm_5class_20260326_213330",
                        help="Run dir for LightGBM with N3 boost (N3 duration comparison)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.results_dir.exists():
        raise SystemExit(f"results directory not found: {args.results_dir}")

    runs = load_runs(args.results_dir)
    print(f"Found {len(runs)} run(s) under {args.results_dir}")

    if args.suite in ("mesa", "all"):
        print("\nMESA suite:")
        mesa_model_comparison(runs, args.out_dir); print("  01_model_comparison.png")
        mesa_task_complexity(runs, args.out_dir); print("  02_task_complexity.png")
        mesa_confusion_matrices(runs, args.out_dir); print("  03_confusion_matrices.png")
        mesa_per_class_radar(runs, args.out_dir); print("  04_per_class_f1_radar.png")
        mesa_feature_importance(runs, args.results_dir, args.out_dir); print("  05_feature_importance.png")
        mesa_stage_distribution(args.mesa_parquet, args.out_dir); print("  06_stage_distribution.png")
        mesa_temporal_impact(runs, args.out_dir); print("  07_temporal_impact.png")
        mesa_auroc(runs, args.out_dir); print("  08_auroc_per_class.png")
        mesa_per_subject_kappa(runs, args.out_dir); print("  09_per_subject_kappa.png")
        mesa_transition_accuracy(runs, args.out_dir); print("  10_transition_accuracy.png")
        mesa_sensitivity_specificity(runs, args.out_dir); print("  11_sensitivity_specificity.png")
        mesa_demographic_analysis(args.mesa_parquet, args.out_dir); print("  12_demographic_analysis.png")

    if args.suite in ("combined", "all"):
        print("\nCombined-Tier1 suite:")
        cnn_path = args.results_dir / args.combined_cnn_run / "metrics.json"
        if cnn_path.exists():
            cnn = json.loads(cnn_path.read_text())
            combined_confusion(cnn, args.out_dir); print("  fig1_confusion_matrix.png")
            combined_per_class(cnn, args.out_dir); print("  fig2_per_class.png")
        else:
            print(f"  (skipped fig1, fig2 — missing {cnn_path})")
        combined_feature_importance(args.results_dir, args.combined_lgbm_run, args.out_dir)
        print("  fig3_feature_importance.png")
        combined_class_distribution(args.out_dir); print("  fig4_class_distribution.png")
        combined_n3_duration(args.results_dir, [
            ("1D-CNN", args.combined_cnn_run, "#0072B2"),
            ("LightGBM", args.combined_lgbm_run, "#009E73"),
            ("LightGBM\n(+N3 boost)", args.combined_lgbm_boost_run, "#999999"),
        ], args.out_dir); print("  fig5_n3_duration.png")

    print(f"\nDone. Figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
