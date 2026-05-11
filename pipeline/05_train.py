"""
STEP 5: ML MODEL TRAINING (5-FOLD CROSS-VALIDATION)
===================================================
PURPOSE:
    Train multiple ML models to classify case outcomes (Allowed/Dismissed/Partial).
    We train 3 models to compare their performance and fairness profiles.

EVALUATION METRICS:
    - Accuracy: Overall % correct
    - Macro F1: Average F1 across all 3 classes (handles class imbalance)
    - Cohen's Kappa: Agreement score beyond chance (used in legal AI papers)
    - Per-cohort accuracy: KEY for our fairness analysis

TRAIN/TEST SPLIT:
    - 5-Fold Stratified Cross-Validation
    - We generate Out-Of-Fold (OOF) predictions to evaluate performance and
      fairness robustly without overfitting.

OUTPUT:
    models/  (saved model files from the best fold)
    results/training_results.json  (all metrics)
    results/per_cohort_results.json (cohort-level metrics)
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              cohen_kappa_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import joblib
import warnings
import copy
from scipy.stats import f_oneway
warnings.filterwarnings("ignore")

BASE_DIR    = r"c:\Users\nites\OneDrive\Desktop\data"
FEATURE_DIR = os.path.join(BASE_DIR, "data_processed", "features")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

def load_features():
    """Load TF-IDF matrix, labels, and cohort information."""
    print("Loading features...")
    X = sp.load_npz(os.path.join(FEATURE_DIR, "tfidf_matrix.npz"))
    y = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
    cohorts = np.load(os.path.join(FEATURE_DIR, "cohorts.npy"), allow_pickle=True)
    years   = np.load(os.path.join(FEATURE_DIR, "years.npy"))
    print(f"  Features: {X.shape[0]:,} docs × {X.shape[1]:,} features")
    print(f"  Labels  : {len(y):,}")
    return X, y, cohorts, years

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics for a model based on OOF predictions.
    """
    metrics = {
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":     round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "cohen_kappa":  round(cohen_kappa_score(y_true, y_pred), 4),
    }

    if y_prob is not None:
        try:
            classes = sorted(set(y_true))
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] > 1:
                auc = roc_auc_score(y_bin, y_prob[:, :y_bin.shape[1]],
                                     multi_class="ovr", average="macro")
                metrics["auc_roc"] = round(auc, 4)
        except Exception:
            metrics["auc_roc"] = None

    report = classification_report(y_true, y_pred,
                                    labels=[0, 1, 2],
                                    target_names=["Dismissed","Allowed","Partial"],
                                    output_dict=True, zero_division=0)
    for cls in ["Dismissed", "Allowed", "Partial"]:
        if cls in report:
            metrics[f"f1_{cls.lower()}"] = round(report[cls]["f1-score"], 4)

    return metrics

def evaluate_per_cohort(y_true, y_pred, cohorts, model_name):
    """
    Evaluates OOF model performance separately for each temporal cohort.
    """
    cohort_metrics = {}
    unique_cohorts = sorted(set(cohorts))

    print(f"\n  Per-Cohort Evaluation for {model_name} (5-Fold CV):")
    for cohort in unique_cohorts:
        mask = cohorts == cohort
        if mask.sum() < 10:
            continue

        y_c = y_true[mask]
        y_pred_c = y_pred[mask]

        acc  = accuracy_score(y_c, y_pred_c)
        f1   = f1_score(y_c, y_pred_c, average="macro", zero_division=0)
        kappa= cohen_kappa_score(y_c, y_pred_c) if len(set(y_c)) > 1 else 0.0

        cohort_metrics[cohort] = {
            "n_samples":   int(mask.sum()),
            "accuracy":    round(acc, 4),
            "macro_f1":    round(f1, 4),
            "cohen_kappa": round(kappa, 4),
        }
        print(f"    {cohort}: n={mask.sum():,}  Acc={acc:.4f}  F1={f1:.4f}  Kappa={kappa:.4f}")

    return cohort_metrics

def train_model_cv(name, base_model, X, y, cohorts):
    """
    Train model using 5-Fold Stratified CV, collect OOF predictions,
    and save the best model.
    """
    print(f"\n{'='*50}")
    print(f"Training: {name} (5-Fold CV)")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), len(set(y))))
    
    best_acc = 0
    best_model = None
    
    fold_metrics = {"accuracy": [], "macro_f1": [], "cohen_kappa": [], "auc_roc": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model to ensure fresh weights
        from sklearn.base import clone
        if name == "XGBoost":
            # XGBoost handles cloning weirdly sometimes, create fresh instance
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=3,
                random_state=RANDOM_STATE + fold, n_jobs=-1,
                eval_metric="mlogloss", verbosity=0
            )
        else:
            model = clone(base_model)
            
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            oof_probs[val_idx] = probs
            
        fold_acc = accuracy_score(y_val, preds)
        fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_kappa = cohen_kappa_score(y_val, preds)
        
        fold_auc = None
        if probs is not None:
            try:
                classes = sorted(set(y_val))
                y_bin = label_binarize(y_val, classes=classes)
                if y_bin.shape[1] > 1:
                    fold_auc = roc_auc_score(y_bin, probs[:, :y_bin.shape[1]], multi_class="ovr", average="macro")
            except Exception:
                pass
                
        fold_metrics["accuracy"].append(fold_acc)
        fold_metrics["macro_f1"].append(fold_f1)
        fold_metrics["cohen_kappa"].append(fold_kappa)
        if fold_auc is not None:
            fold_metrics["auc_roc"].append(fold_auc)
            
        # Track best model
        if fold_acc > best_acc:
            best_acc = fold_acc
            best_model = copy.deepcopy(model)
            
        print(f"  Fold {fold+1}/{N_SPLITS} complete. Acc: {fold_acc:.4f} (F1: {fold_f1:.4f})")

    # Overall metrics on OOF predictions
    metrics = compute_metrics(y, oof_preds, oof_probs)
    
    # Calculate 95% CIs
    ci_95 = {}
    for k, v in fold_metrics.items():
        if len(v) > 0:
            std_val = np.std(v, ddof=1)
            ci = 1.96 * (std_val / np.sqrt(len(v)))
            ci_95[k] = ci
            
    metrics["ci_95_accuracy"] = round(ci_95.get("accuracy", 0), 4)
    metrics["ci_95_macro_f1"] = round(ci_95.get("macro_f1", 0), 4)
    metrics["ci_95_kappa"] = round(ci_95.get("cohen_kappa", 0), 4)
    metrics["ci_95_auc_roc"] = round(ci_95.get("auc_roc", 0), 4)
    
    print(f"\n  OVERALL METRICS (OOF) with 95% CI:")
    print(f"    Accuracy            : {metrics['accuracy']} ± {metrics['ci_95_accuracy']}")
    print(f"    Macro F1            : {metrics['macro_f1']} ± {metrics['ci_95_macro_f1']}")
    print(f"    Cohen's Kappa       : {metrics['cohen_kappa']} ± {metrics['ci_95_kappa']}")
    if "auc_roc" in metrics and metrics.get("auc_roc") is not None:
        print(f"    AUC-ROC             : {metrics['auc_roc']} ± {metrics['ci_95_auc_roc']}")

    # Per-cohort metrics
    cohort_metrics = evaluate_per_cohort(y, oof_preds, cohorts, name)

    # Save best model
    model_path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  Best Model saved: {model_path} (Acc: {best_acc:.4f})")

    return metrics, cohort_metrics, fold_metrics["accuracy"]

def compute_fairness_gap(cohort_metrics):
    accs = [v["accuracy"] for v in cohort_metrics.values()]
    if len(accs) < 2:
        return 0.0
    return round(max(accs) - min(accs), 4)

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 5: ML Model Training (5-Fold CV)")
    print("=" * 60)

    X, y, cohorts, years = load_features()

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            multi_class="multinomial", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", verbosity=0
        ),
    }

    all_results = {}
    all_cohort_results = {}
    all_fold_accs = {}

    for model_name, base_model in models.items():
        metrics, cohort_metrics, fold_accs = train_model_cv(
            model_name, base_model, X, y, cohorts
        )
        fairness_gap = compute_fairness_gap(cohort_metrics)
        metrics["temporal_fairness_gap"] = fairness_gap
        print(f"\n  Temporal Fairness Gap: {fairness_gap:.4f}")

        all_results[model_name] = metrics
        all_cohort_results[model_name] = cohort_metrics
        all_fold_accs[model_name] = fold_accs

    with open(os.path.join(RESULTS_DIR, "training_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "per_cohort_results.json"), "w") as f:
        json.dump(all_cohort_results, f, indent=2)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (5-FOLD CV with 95% CI)")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>15} {'Macro F1':>15} {'Kappa':>15}")
    print("-" * 68)
    for name, m in all_results.items():
        acc_str = f"{m['accuracy']:.4f} ± {m['ci_95_accuracy']:.4f}"
        f1_str  = f"{m['macro_f1']:.4f} ± {m['ci_95_macro_f1']:.4f}"
        kap_str = f"{m['cohen_kappa']:.4f} ± {m['ci_95_kappa']:.4f}"
        print(f"{name:<20} {acc_str:>15} {f1_str:>15} {kap_str:>15}")

    if "Logistic Regression" in all_fold_accs and "Random Forest" in all_fold_accs and "XGBoost" in all_fold_accs:
        f_stat, p_val = f_oneway(
            all_fold_accs["Logistic Regression"],
            all_fold_accs["Random Forest"],
            all_fold_accs["XGBoost"]
        )
        print("\n" + "=" * 60)
        print("STATISTICAL SIGNIFICANCE (ANOVA on Fold Accuracies)")
        print("=" * 60)
        print(f"F-Statistic: {f_stat:.4f}")
        print(f"p-value:     {p_val:.4e}")
        if p_val < 0.05:
            print("Conclusion:  The differences in accuracy across models are statistically significant (p < 0.05).")
        else:
            print("Conclusion:  No statistically significant difference in accuracy between models (p >= 0.05).")

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Next step: Run pipeline/06_fairness.py")

if __name__ == "__main__":
    main()
