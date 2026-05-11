import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import scipy.sparse as sp

BASE_DIR = r"c:\Users\nites\OneDrive\Desktop\data"
FEATURE_DIR = os.path.join(BASE_DIR, "data_processed", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading data...")
X = sp.load_npz(os.path.join(FEATURE_DIR, "tfidf_matrix.npz"))
y = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
years = np.load(os.path.join(FEATURE_DIR, "years.npy"))

# 1. Dataset Timeline Distribution
print("Generating Timeline Distribution...")
plt.figure(figsize=(10, 5))
unique_years, counts = np.unique(years, return_counts=True)
sns.barplot(x=unique_years, y=counts, palette="viridis")
plt.title("Supreme Court Cases per Year (1990-2025)", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Judgments", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dataset_timeline.png"), dpi=300)
plt.close()

# 2. Confusion Matrix for XGBoost
print("Generating Confusion Matrix...")
model_path = os.path.join(MODEL_DIR, "xgboost.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    
    cm = confusion_matrix(y, y_pred)
    # Classes: 0: Dismissed, 1: Allowed, 2: Partial
    labels = ["Dismissed", "Allowed", "Partial"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title("XGBoost Decision Extraction\nConfusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Outcome", fontsize=12)
    plt.ylabel("Actual Outcome", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "xgboost_confusion_matrix.png"), dpi=300)
    plt.close()
    print("Figures generated successfully!")
else:
    print(f"Model not found at {model_path}")
