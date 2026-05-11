"""
STEP 7: EXPLAINABILITY (SHAP ANALYSIS)
======================================
PURPOSE:
    Understanding WHY the AI makes its decisions. For high-stakes legal tasks,
    we cannot use "black box" models. We must explain what legal features 
    drove the prediction of Allowed vs Dismissed.

METHOD:
    We use SHAP (SHapley Additive exPlanations) values to extract the 
    top words and meta-features the XGBoost model relies on. 
    Crucially, we check if these top features remain consistent across time 
    or if the model relies on era-specific language (which would be a red flag).

OUTPUT:
    figures/shap_summary_plot.png
    results/top_shap_features.csv
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

BASE_DIR    = r"c:\Users\nites\OneDrive\Desktop\data"
FEATURE_DIR = os.path.join(BASE_DIR, "data_processed", "features")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data_and_model():
    print("Loading XGBoost model and features...")
    # Load model
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    
    # Load features (use a sample to compute SHAP values quickly)
    X = sp.load_npz(os.path.join(FEATURE_DIR, "tfidf_matrix.npz"))
    
    # Load feature names (vocabulary)
    with open(os.path.join(FEATURE_DIR, "tfidf_vocab.json"), "r") as f:
        vocab = json.load(f)
    
    # Reverse vocabulary mapping (index to word)
    idx_to_word = {v: k for k, v in vocab.items()}
    feature_names = [idx_to_word[i] for i in range(len(vocab))]
    
    return model, X, feature_names

def run_shap_analysis(model, X_sparse, feature_names):
    print("\nInitializing SHAP TreeExplainer...")
    # Sample 500 documents for SHAP (computing SHAP on all 7,000+ sparse docs is very slow)
    np.random.seed(42)
    sample_indices = np.random.choice(X_sparse.shape[0], size=500, replace=False)
    X_sample = X_sparse[sample_indices].toarray()  # Convert to dense for SHAP
    
    explainer = shap.TreeExplainer(model)
    print("Computing SHAP values (this may take a minute)...")
    shap_values = explainer.shap_values(X_sample)
    
    # shap_values shape for multi-class XGBoost is usually (n_samples, n_features, n_classes) or list
    # For newer SHAP/XGBoost, it's a list of arrays or a single array
    
    # Let's get the absolute mean SHAP values for class 1 (Allowed) as our primary importance metric
    # Note: If shap_values is a list, class 1 is shap_values[1]
    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_class = shap_values[:, :, 1]
    else:
        shap_values_class = shap_values
        
    mean_abs_shap = np.abs(shap_values_class).mean(axis=0)
    
    # Sort top 20 features
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]
    
    print("\nTop 20 Features driving 'Allowed' predictions:")
    print("-" * 50)
    for i, (feat, val) in enumerate(top_features):
        print(f"{i+1:2d}. {feat:20s} : {val:.6f}")
        
    # Save to CSV
    df_shap = pd.DataFrame(top_features, columns=["Feature", "Mean_Abs_SHAP"])
    csv_path = os.path.join(RESULTS_DIR, "top_shap_features.csv")
    df_shap.to_csv(csv_path, index=False)
    print(f"\nSaved top SHAP features to: {csv_path}")
    
    # Generate SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (Top Predictive Features)")
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to: {plot_path}")

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 7: SHAP Explainability Analysis")
    print("=" * 60)
    
    model, X, feature_names = load_data_and_model()
    run_shap_analysis(model, X, feature_names)
    
    print("\n" + "=" * 60)
    print("ALL PIPELINE STEPS COMPLETE!")
    print("=" * 60)
    print("You are now ready to compile the final IEEE research paper.")

if __name__ == "__main__":
    main()
