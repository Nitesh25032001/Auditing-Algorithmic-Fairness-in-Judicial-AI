"""
STEP 2: TEXT CLEANING
======================
PURPOSE:
    Takes the raw extracted text and removes all noise that would
    confuse the ML model. Court documents have lots of boilerplate
    that is NOT useful for predicting outcomes.

WHAT WE REMOVE:
    - Page numbers ("Page 1 of 12", "1", "2", etc.)
    - Headers/footers (repeated court names, case numbers)
    - Citation noise ("AIR 1990 SC 123", "SCC 456")
    - Extra whitespace, special characters
    - Very short documents (likely extraction failures)

WHY THIS MATTERS:
    If we feed noisy text to the ML model, it learns from noise
    instead of actual legal reasoning. Cleaning = better accuracy.

OUTPUT:
    data_processed/cleaned.json
    Each record: {id, filename, year, cohort, clean_text, word_count, status}
"""

import json
import re
import os
from tqdm import tqdm

BASE_DIR    = r"c:\Users\nites\OneDrive\Desktop\data"
INPUT_FILE  = os.path.join(BASE_DIR, "data_processed", "extracted.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data_processed", "cleaned.json")

# Minimum word count to keep a document
MIN_WORDS = 100

def clean_text(text):
    """
    Cleans raw legal text by removing noise patterns.
    """
    if not text:
        return ""

    # 1. Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # 2. Remove common header/footer patterns in Indian court docs
    text = re.sub(r'IN THE SUPREME COURT OF INDIA\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'CIVIL APPELLATE JURISDICTION\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'CRIMINAL APPELLATE JURISDICTION\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'REPORTABLE\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'NOT REPORTABLE\s*', ' ', text, flags=re.IGNORECASE)

    # 3. Remove legal citation noise (AIR 1990 SC 123, (2010) 5 SCC 123)
    text = re.sub(r'AIR\s+\d{4}\s+SC\s+\d+', ' ', text)
    text = re.sub(r'\(\d{4}\)\s+\d+\s+SCC\s+\d+', ' ', text)
    text = re.sub(r'\d{4}\s+\(\d+\)\s+SCC\s+\d+', ' ', text)

    # 4. Remove writ petition / appeal numbers
    text = re.sub(r'(Civil Appeal|Criminal Appeal|Writ Petition|SLP)[\s\(]*No[\.\s]*\d+[\s\)]*of\s+\d{4}', ' ', text, flags=re.IGNORECASE)

    # 5. Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)      # max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)            # max 1 consecutive space
    text = re.sub(r'\s+\.', '.', text)             # fix ". " → "."

    # 6. Remove non-ASCII characters (OCR artefacts)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 2: Text Cleaning")
    print("=" * 60)

    print(f"\nReading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Total records loaded: {len(records):,}")

    cleaned_records = []
    skipped = 0

    for record in tqdm(records, desc="Cleaning text", unit="doc"):
        # Skip previously failed extractions
        if record.get("status") != "ok":
            skipped += 1
            continue

        clean = clean_text(record["raw_text"])
        word_count = len(clean.split())

        # Skip documents that are too short after cleaning
        if word_count < MIN_WORDS:
            skipped += 1
            continue

        cleaned_records.append({
            "id":         record["id"],
            "filename":   record["filename"],
            "case_name":  record.get("case_name", ""),
            "year":       record["year"],
            "cohort":     record["cohort"],
            "clean_text": clean,
            "word_count": word_count,
        })

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  Input records   : {len(records):,}")
    print(f"  Cleaned & kept  : {len(cleaned_records):,}")
    print(f"  Skipped/removed : {skipped:,}")

    # Cohort distribution
    cohort_counts = {}
    for r in cleaned_records:
        c = r["cohort"]
        cohort_counts[c] = cohort_counts.get(c, 0) + 1
    print(f"\nCohort distribution:")
    for cohort, count in sorted(cohort_counts.items()):
        print(f"    {cohort}: {count:,} cases")

    avg_words = int(sum(r["word_count"] for r in cleaned_records) / len(cleaned_records)) if cleaned_records else 0
    print(f"\nAverage word count after cleaning: {avg_words:,}")
    print(f"\nOutput saved: {OUTPUT_FILE}")
    print("Next step: Run pipeline/03_label.py")

if __name__ == "__main__":
    main()
