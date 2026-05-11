"""
STEP 4: FEATURE ENGINEERING
=============================
PURPOSE:
    Machine Learning models cannot understand raw text.
    We need to convert text into NUMBERS (feature vectors).

TWO APPROACHES WE USE:

1. TF-IDF (Term Frequency - Inverse Document Frequency)
   - Classic NLP approach. Fast, works with all ML models.
   - Assigns a score to each word based on how important it is
     across all documents. Common words (the, is, and) get low
     scores. Rare legal terms (quashed, mandamus, certiorari)
     get high scores.
   - Output: a matrix of 50,000 features per document

2. Sentence-level Statistics (Handcrafted Legal Features)
   - word_count, sentence_count, avg_sentence_length
   - legal_term_density: how many legal terms per 100 words
   - These capture writing style differences across eras

WHY TWO APPROACHES?
   - TF-IDF features → used for LR, RF, XGBoost (fast models)
   - Sentence stats → used alongside TF-IDF for richer features
   - BERT embeddings → created during training (Step 5)

OUTPUT:
    data_processed/features/tfidf_matrix.npz  (sparse matrix)
    data_processed/features/tfidf_vocab.json  (vocabulary)
    data_processed/features/meta_features.csv (word counts etc)
    data_processed/features/labels.npy        (outcome array)
    data_processed/features/cohorts.npy       (cohort array)
    data_processed/features/years.npy         (year array)
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

BASE_DIR     = r"c:\Users\nites\OneDrive\Desktop\data"
INPUT_FILE   = os.path.join(BASE_DIR, "data_processed", "labeled.csv")
FEATURE_DIR  = os.path.join(BASE_DIR, "data_processed", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

# Legal terms specific to Indian Supreme Court — used for legal density feature
LEGAL_TERMS = [
    "appellant", "respondent", "petitioner", "writ", "mandamus", "certiorari",
    "prohibition", "quo warranto", "habeas corpus", "affidavit", "amicus",
    "ex parte", "prima facie", "locus standi", "ultra vires", "suo motu",
    "quash", "quashed", "impugned", "remand", "remanded", "stay", "interim",
    "injunction", "decree", "judgment", "order", "appeal", "revision",
    "acquittal", "conviction", "sentence", "bail", "custody", "detenu",
    "constitutional", "fundamental rights", "article 14", "article 19",
    "article 21", "article 32", "article 226", "schedule", "amendment",
]

def compute_legal_density(text):
    """
    Counts how many legal terms appear per 100 words.
    Higher density = more legally technical document.
    This may vary across eras (older docs use more archaic language).
    """
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    term_count = sum(1 for term in LEGAL_TERMS if term in text_lower)
    return round((term_count / word_count) * 100, 4)

def compute_meta_features(text):
    """
    Computes basic statistical features from the judgment text.
    """
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    words = text.split()
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    return {
        "word_count":        len(words),
        "sentence_count":    len(sentences),
        "avg_sentence_len":  round(avg_sent_len, 2),
        "legal_density":     compute_legal_density(text),
        "char_count":        len(text),
        "unique_words":      len(set(w.lower() for w in words)),
        "lexical_richness":  round(len(set(words)) / len(words), 4) if words else 0,
    }

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 4: Feature Engineering")
    print("=" * 60)

    print(f"\nReading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"Total records: {len(df):,}")

    # Keep only labeled documents (exclude unknown outcomes)
    df_labeled = df[df["outcome"].isin([0, 1, 2])].copy()
    print(f"Records with valid labels: {len(df_labeled):,}")
    print(f"Removed unknown outcomes : {len(df) - len(df_labeled):,}")

    texts  = df_labeled["clean_text"].fillna("").tolist()
    labels = df_labeled["outcome"].values
    years  = df_labeled["year"].values
    cohorts= df_labeled["cohort"].values

    # ─────────────────────────────────────────────
    # TF-IDF VECTORIZATION
    # max_features=50000: keep top 50,000 most important words
    # ngram_range=(1,2):  include single words AND two-word phrases
    #                     e.g., "appeal dismissed", "set aside"
    # min_df=3:           ignore words appearing in fewer than 3 docs
    # sublinear_tf=True:  apply log scaling to term frequencies
    # ─────────────────────────────────────────────
    print("\n[1/3] Computing TF-IDF features...")
    print("      (50,000 features, unigrams + bigrams — this takes a few minutes)")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-z][a-z]+\b',  # only alphabetic tokens
    )

    tfidf_matrix = vectorizer.fit_transform(tqdm(texts, desc="TF-IDF vectorizing"))

    print(f"      TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Save TF-IDF matrix (sparse format saves space)
    sp.save_npz(os.path.join(FEATURE_DIR, "tfidf_matrix.npz"), tfidf_matrix)

    # Save vocabulary
    vocab = {word: int(idx) for word, idx in vectorizer.vocabulary_.items()}
    with open(os.path.join(FEATURE_DIR, "tfidf_vocab.json"), "w") as f:
        json.dump(vocab, f)

    print("      Saved: tfidf_matrix.npz, tfidf_vocab.json")

    # ─────────────────────────────────────────────
    # HANDCRAFTED META FEATURES
    # ─────────────────────────────────────────────
    print("\n[2/3] Computing meta features (legal density, sentence stats)...")
    meta_rows = []
    for text in tqdm(texts, desc="Meta features"):
        meta_rows.append(compute_meta_features(text))

    meta_df = pd.DataFrame(meta_rows)
    meta_df["year"]    = years
    meta_df["cohort"]  = cohorts
    meta_df["outcome"] = labels
    meta_df.to_csv(os.path.join(FEATURE_DIR, "meta_features.csv"), index=False)
    print("      Saved: meta_features.csv")

    # ─────────────────────────────────────────────
    # SAVE LABELS AND COHORTS
    # ─────────────────────────────────────────────
    print("\n[3/3] Saving labels, cohorts, years...")
    np.save(os.path.join(FEATURE_DIR, "labels.npy"),  labels)
    np.save(os.path.join(FEATURE_DIR, "cohorts.npy"), cohorts)
    np.save(os.path.join(FEATURE_DIR, "years.npy"),   years)

    # Save the filtered dataset too (needed in training step)
    df_labeled.drop(columns=["clean_text"]).to_csv(
        os.path.join(BASE_DIR, "data_processed", "dataset.csv"), index=False
    )

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"  TF-IDF matrix : {tfidf_matrix.shape[0]:,} docs × {tfidf_matrix.shape[1]:,} features")
    print(f"  Meta features : {len(meta_df.columns)} columns")

    print(f"\nLabel distribution:")
    for label, name in [(0,"Dismissed"),(1,"Allowed"),(2,"Partial")]:
        cnt = int(np.sum(labels == label))
        pct = cnt / len(labels) * 100
        print(f"    {name:12}: {cnt:,}  ({pct:.1f}%)")

    print(f"\nCohort distribution (for fairness audit):")
    for cohort in sorted(set(cohorts)):
        cnt = int(np.sum(cohorts == cohort))
        pct = cnt / len(cohorts) * 100
        print(f"    {cohort}: {cnt:,}  ({pct:.1f}%)")

    print(f"\nAll features saved to: {FEATURE_DIR}")
    print("Next step: Run pipeline/05_train.py")

if __name__ == "__main__":
    main()
