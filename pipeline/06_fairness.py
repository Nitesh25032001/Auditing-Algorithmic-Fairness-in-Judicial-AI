"""
STEP 6: TEMPORAL FAIRNESS AUDIT — THE CORE CONTRIBUTION
=========================================================
PURPOSE:
    This is the HEART of our research paper. We formally measure whether
    the ML models are "temporally fair" — i.e., do they treat cases from
    different time periods equally?

FAIRNESS METRICS COMPUTED:

1. DEMOGRAPHIC PARITY DIFFERENCE (DPD)
   - Checks if the model PREDICTS "Allowed" at the same rate across cohorts.
   - Formula: max(P(ŷ=1|cohort=A), ...) − min(P(ŷ=1|cohort=A), ...)
   - Ideal value: 0.0 (same prediction rate regardless of era)
   - Threshold for concern: > 0.10

2. EQUALIZED ODDS DIFFERENCE (EOD)
   - Checks if True Positive Rate (sensitivity) is equal across cohorts.
   - A model may be accurate overall but good at catching "Allowed" only
     for newer cases, while being poor for older ones.
   - Ideal value: 0.0

3. PREDICTIVE PARITY
   - Among cases predicted as "Allowed", are they actually "Allowed"
     at the same rate in each cohort?
   - Checks precision consistency across eras.

4. EXPECTED CALIBRATION ERROR (ECE)
   - Measures whether the model's CONFIDENCE matches actual accuracy.
   - If a model says "90% sure" but is right only 60% of the time
     for old cases, it's poorly calibrated for that cohort.

5. STATISTICAL PARITY GAP
   - Overall disparity index: combines DPD and EOD.

6. TEMPORAL DRIFT (KS-TEST)
   - Kolmogorov-Smirnov test to check if prediction distributions
     significantly differ across cohorts.
   - p < 0.05 means statistically significant drift.

7. POPULATION STABILITY INDEX (PSI)
   - Measures how much the prediction distribution has shifted.
   - PSI < 0.1: Stable, PSI 0.1-0.25: Moderate drift, PSI > 0.25: Major drift

OUTPUT:
    results/fairness_metrics.json    (all computed metrics)
    results/fairness_summary.csv     (table for paper)
    figures/fairness_*.png           (plots for paper)
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = r"c:\Users\nites\OneDrive\Desktop\data"
FEATURE_DIR = os.path.join(BASE_DIR, "data_processed", "features")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

COHORT_LABELS = {
    "A_1990_2000": "1990–2000",
    "B_2001_2010": "2001–2010",
    "C_2011_2020": "2011–2020",
    "D_2021_2025": "2021–2025",
}
COHORT_ORDER = sorted(COHORT_LABELS.keys())

# ─────────────────────────────────────────────
# FAIRNESS METRIC FUNCTIONS
# ─────────────────────────────────────────────

def demographic_parity_difference(y_pred, cohorts):
    """
    DPD = max prediction rate − min prediction rate across cohorts.
    We use label=1 (Allowed) as the positive class.
    """
    rates = {}
    for cohort in COHORT_ORDER:
        mask = cohorts == cohort
        if mask.sum() == 0:
            continue
        rates[cohort] = float(np.mean(y_pred[mask] == 1))
    if len(rates) < 2:
        return 0.0, rates
    dpd = max(rates.values()) - min(rates.values())
    return round(dpd, 4), rates

def equalized_odds_difference(y_true, y_pred, cohorts):
    """
    EOD = max TPR difference across cohorts (for positive class = Allowed).
    TPR = True Positive Rate = among actual Allowed cases, how many did we predict Allowed?
    """
    tprs = {}
    for cohort in COHORT_ORDER:
        mask = cohorts == cohort
        y_t = y_true[mask]
        y_p = y_pred[mask]
        positives = y_t == 1
        if positives.sum() == 0:
            tprs[cohort] = 0.0
            continue
        tprs[cohort] = float(np.sum((y_p == 1) & (y_t == 1)) / positives.sum())

    if len(tprs) < 2:
        return 0.0, tprs
    eod = max(tprs.values()) - min(tprs.values())
    return round(eod, 4), tprs

def predictive_parity(y_true, y_pred, cohorts):
    """
    Predictive Parity: Among predicted Allowed, what fraction is actually Allowed?
    Measures precision consistency across cohorts.
    """
    ppv = {}
    for cohort in COHORT_ORDER:
        mask = cohorts == cohort
        y_t = y_true[mask]
        y_p = y_pred[mask]
        predicted_pos = y_p == 1
        if predicted_pos.sum() == 0:
            ppv[cohort] = 0.0
            continue
        ppv[cohort] = float(np.sum((y_t == 1) & (y_p == 1)) / predicted_pos.sum())

    if len(ppv) < 2:
        return 0.0, ppv
    pp_diff = max(ppv.values()) - min(ppv.values())
    return round(pp_diff, 4), ppv

def expected_calibration_error(y_true, y_prob, cohort_mask, n_bins=10):
    """
    ECE measures whether confidence scores match actual accuracy.
    Bins predictions by confidence, computes weighted mean |confidence - accuracy|.
    """
    y_t = y_true[cohort_mask]
    prob = y_prob[cohort_mask]

    if len(y_t) == 0:
        return 0.0

    # Use max probability as confidence
    confidence = np.max(prob, axis=1)
    predicted  = np.argmax(prob, axis=1)
    correct    = (predicted == y_t).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i+1])
        if in_bin.sum() == 0:
            continue
        acc_in_bin = correct[in_bin].mean()
        conf_in_bin = confidence[in_bin].mean()
        ece += (in_bin.sum() / len(y_t)) * abs(acc_in_bin - conf_in_bin)

    return round(float(ece), 4)

def population_stability_index(pred_dist_ref, pred_dist_new, epsilon=1e-6):
    """
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    Uses the first cohort (1990-2000) as the reference distribution.
    """
    p = np.array(pred_dist_ref) + epsilon
    q = np.array(pred_dist_new) + epsilon
    p /= p.sum()
    q /= q.sum()
    psi = np.sum((q - p) * np.log(q / p))
    return round(float(psi), 4)

def ks_test_across_cohorts(y_prob_by_cohort):
    """
    KS-test: Is the distribution of prediction probabilities significantly
    different between the first cohort and all others?
    """
    ref_probs = y_prob_by_cohort.get("A_1990_2000", np.array([]))
    results = {}
    for cohort, probs in y_prob_by_cohort.items():
        if cohort == "A_1990_2000" or len(probs) == 0 or len(ref_probs) == 0:
            continue
        stat, pval = stats.ks_2samp(ref_probs, probs)
        results[cohort] = {
            "ks_statistic": round(float(stat), 4),
            "p_value":      round(float(pval), 6),
            "significant":  bool(pval < 0.05)
        }
    return results

# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────

def plot_accuracy_by_cohort(cohort_acc_by_model, save_path):
    """Bar chart: accuracy per cohort, per model."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    cohort_display = [COHORT_LABELS.get(c, c) for c in COHORT_ORDER]
    x = np.arange(len(COHORT_ORDER))
    width = 0.25

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for i, (model_name, accs) in enumerate(cohort_acc_by_model.items()):
        vals = [accs.get(c, 0) for c in COHORT_ORDER]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=colors[i], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Temporal Cohort", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Across Temporal Cohorts\n(Indian Supreme Court, 1990–2025)",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(cohort_display, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='Random Baseline')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_fairness_metrics(fairness_data, save_path):
    """Grouped bar chart of all fairness metrics across models."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["dpd", "eod", "pp_diff"]
    metric_labels = [
        "Demographic\nParity Diff.",
        "Equalized\nOdds Diff.",
        "Predictive\nParity Diff."
    ]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    model_names = list(fairness_data.keys())

    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [fairness_data[m].get(metric, 0) for m in model_names]
        bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor='white')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel("Difference (lower = fairer)", fontsize=10)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Threshold (0.1)')
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(vals) * 1.4, 0.2))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("Temporal Fairness Metrics by Model", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_ece_by_cohort(ece_by_model, save_path):
    """Heatmap of ECE (calibration error) across cohorts and models."""
    plt.style.use('seaborn-v0_8-whitegrid')

    data = {}
    for model, cohort_ece in ece_by_model.items():
        data[model] = {COHORT_LABELS.get(c, c): v for c, v in cohort_ece.items()}

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "ECE (lower = better)"})
    ax.set_title("Expected Calibration Error by Model & Temporal Cohort",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Temporal Cohort", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ─────────────────────────────────────────────
# MAIN AUDIT FUNCTION
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 6: Temporal Fairness Audit")
    print("=" * 60)

    # Load features and labels
    print("\nLoading data...")
    X = sp.load_npz(os.path.join(FEATURE_DIR, "tfidf_matrix.npz"))
    y = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
    cohorts = np.load(os.path.join(FEATURE_DIR, "cohorts.npy"), allow_pickle=True)

    # Load models
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
    }

    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  Loaded: {name}")
        else:
            print(f"  [WARN] Model not found: {fname} — skipping")

    if not models:
        print("ERROR: No models found. Run 05_train.py first.")
        return

    # ── Run Fairness Audit for Each Model ──
    fairness_results = {}
    cohort_acc_by_model = {}
    ece_by_model = {}

    for model_name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"Auditing: {model_name}")
        print(f"{'─'*50}")

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        # 1. Demographic Parity Difference
        dpd, pred_rates = demographic_parity_difference(y_pred, cohorts)
        print(f"\n  [1] Demographic Parity Difference (DPD) = {dpd}")
        for c, r in pred_rates.items():
            print(f"      {COHORT_LABELS.get(c,c)}: Prediction(Allowed) = {r:.4f}")

        # 2. Equalized Odds Difference
        eod, tprs = equalized_odds_difference(y, y_pred, cohorts)
        print(f"\n  [2] Equalized Odds Difference (EOD) = {eod}")
        for c, t in tprs.items():
            print(f"      {COHORT_LABELS.get(c,c)}: TPR = {t:.4f}")

        # 3. Predictive Parity
        pp_diff, ppv = predictive_parity(y, y_pred, cohorts)
        print(f"\n  [3] Predictive Parity Difference = {pp_diff}")
        for c, p in ppv.items():
            print(f"      {COHORT_LABELS.get(c,c)}: Precision(Allowed) = {p:.4f}")

        # 4. ECE by cohort
        ece_by_cohort = {}
        if y_prob is not None:
            print(f"\n  [4] Expected Calibration Error (ECE) by cohort:")
            for cohort in COHORT_ORDER:
                mask = cohorts == cohort
                if mask.sum() == 0:
                    continue
                ece_val = expected_calibration_error(y, y_prob, mask)
                ece_by_cohort[cohort] = ece_val
                print(f"      {COHORT_LABELS.get(cohort,cohort)}: ECE = {ece_val:.4f}")

        # 5. PSI (vs reference cohort A_1990_2000)
        print(f"\n  [5] Population Stability Index (PSI vs 1990-2000):")
        ref_mask = cohorts == "A_1990_2000"
        ref_dist  = np.bincount(y_pred[ref_mask], minlength=3) / ref_mask.sum()
        psi_vals = {}
        for cohort in COHORT_ORDER[1:]:
            mask = cohorts == cohort
            if mask.sum() == 0:
                continue
            cur_dist = np.bincount(y_pred[mask], minlength=3) / mask.sum()
            psi = population_stability_index(ref_dist, cur_dist)
            severity = "Stable" if psi < 0.1 else ("Moderate drift" if psi < 0.25 else "MAJOR DRIFT")
            psi_vals[cohort] = psi
            print(f"      {COHORT_LABELS.get(cohort,cohort)}: PSI = {psi:.4f} [{severity}]")

        # 6. KS Test
        print(f"\n  [6] KS-Test (distribution shift vs 1990-2000):")
        prob_by_cohort = {}
        if y_prob is not None:
            for cohort in COHORT_ORDER:
                mask = cohorts == cohort
                if mask.sum() > 0:
                    prob_by_cohort[cohort] = y_prob[mask, 1]  # P(Allowed) per case
            ks_results = ks_test_across_cohorts(prob_by_cohort)
            for cohort, res in ks_results.items():
                sig = "***SIGNIFICANT***" if res["significant"] else "not significant"
                print(f"      {COHORT_LABELS.get(cohort,cohort)}: KS={res['ks_statistic']:.4f}, p={res['p_value']:.4f} [{sig}]")
        else:
            ks_results = {}

        # Per-cohort accuracy (for plot)
        cohort_acc = {}
        for cohort in COHORT_ORDER:
            mask = cohorts == cohort
            if mask.sum() == 0:
                continue
            from sklearn.metrics import accuracy_score
            cohort_acc[cohort] = round(accuracy_score(y[mask], y_pred[mask]), 4)
        cohort_acc_by_model[model_name] = cohort_acc

        # Store all results
        fairness_results[model_name] = {
            "dpd":            dpd,
            "eod":            eod,
            "pp_diff":        pp_diff,
            "pred_rates":     pred_rates,
            "tprs":           tprs,
            "ppv":            ppv,
            "ece_by_cohort":  ece_by_cohort,
            "psi":            psi_vals,
            "ks_test":        ks_results,
            "cohort_accuracy":cohort_acc,
        }
        ece_by_model[model_name] = ece_by_cohort

    # ── Save All Results ──
    with open(os.path.join(RESULTS_DIR, "fairness_metrics.json"), "w") as f:
        json.dump(fairness_results, f, indent=2)

    # ── Summary Table (for paper) ──
    rows = []
    for model_name, res in fairness_results.items():
        rows.append({
            "Model":                model_name,
            "DPD":                  res["dpd"],
            "EOD":                  res["eod"],
            "Predictive Parity":    res["pp_diff"],
            "Temporal Fairness Gap":max(res["cohort_accuracy"].values()) - min(res["cohort_accuracy"].values())
                                    if res["cohort_accuracy"] else 0,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "fairness_summary.csv"), index=False)

    # ── Generate Plots ──
    print("\n" + "="*50)
    print("Generating figures for paper...")
    plot_accuracy_by_cohort(cohort_acc_by_model,
                             os.path.join(FIGURE_DIR, "fairness_accuracy_by_cohort.png"))
    plot_fairness_metrics(fairness_results,
                           os.path.join(FIGURE_DIR, "fairness_metrics_comparison.png"))
    if ece_by_model:
        plot_ece_by_cohort(ece_by_model,
                            os.path.join(FIGURE_DIR, "fairness_ece_heatmap.png"))

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("FAIRNESS AUDIT COMPLETE")
    print("=" * 60)
    print(f"\n{'Model':<25} {'DPD':>8} {'EOD':>8} {'PP':>8} {'Fair.Gap':>10}")
    print("-" * 62)
    for row in rows:
        print(f"{row['Model']:<25} {row['DPD']:>8.4f} {row['EOD']:>8.4f} "
              f"{row['Predictive Parity']:>8.4f} {row['Temporal Fairness Gap']:>10.4f}")

    print(f"\nKey: DPD=Demographic Parity Diff, EOD=Equalized Odds Diff")
    print(f"     PP=Predictive Parity Diff, Fair.Gap=Max-Min Accuracy across cohorts")
    print(f"\n{'>'*3} Values closer to 0 = MORE FAIR across time periods")
    print(f"\nAll results: {RESULTS_DIR}")
    print(f"All figures: {FIGURE_DIR}")
    print("Next step: Run pipeline/07_xai.py")

if __name__ == "__main__":
    main()
