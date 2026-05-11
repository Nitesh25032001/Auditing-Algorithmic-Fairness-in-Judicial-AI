"""
STEP 1: PDF TEXT EXTRACTION
============================
PURPOSE:
    Reads every PDF file from the year folders (1990-2025) and extracts the
    raw judgment text. Saves output as a JSON file for the next pipeline step.

WHY THIS APPROACH:
    - pdfplumber is the most reliable tool for court document PDFs
    - We track year from folder name (used later for temporal fairness)
    - We track filename as case name (metadata for paper)
    - We save progress in batches so if it crashes, we don't lose everything

OUTPUT:
    data_processed/extracted.json
    Each record: {id, filename, year, raw_text, word_count, status}
"""

import os
import json
import pdfplumber
from tqdm import tqdm
import logging

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR   = r"c:\Users\nites\OneDrive\Desktop\data"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "extracted.json")
YEARS = list(range(1990, 2026))   # 1990 to 2025
BATCH_SAVE = 500                   # Save progress every 500 files

# ─────────────────────────────────────────────
# LOGGING SETUP
# (Shows info messages + saves errors to a log file)
# ─────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "extraction_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

def extract_text_from_pdf(filepath):
    """
    Opens a PDF and extracts all text from all pages.
    Returns the text as a single string.
    If the PDF fails (corrupted, image-only), returns empty string.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n".join(pages_text)
    except Exception as e:
        logging.error(f"Failed: {filepath} | Error: {e}")
        return ""

def get_case_name(filename):
    """
    Converts filename to a readable case name.
    Example:
        'A_K_Bhatnagar_vs_Union_Of_India_on_9_November_1990_1.PDF'
        → 'A K Bhatnagar vs Union Of India on 9 November 1990'
    """
    name = filename.replace(".PDF", "").replace(".pdf", "")
    name = name.replace("_", " ").strip()
    # Remove trailing '_1', '_2' artifact numbers
    if name.endswith(" 1") or name.endswith(" 2"):
        name = name[:-2].strip()
    return name

def assign_cohort(year):
    """
    Assigns a temporal cohort label to each judgment year.
    This is our 'protected group' for the fairness audit.

    Cohort A: 1990-2000  (Pre-IT era)
    Cohort B: 2001-2010  (Digital transition era)
    Cohort C: 2011-2020  (Modern era)
    Cohort D: 2021-2025  (AI era)
    """
    if 1990 <= year <= 2000:
        return "A_1990_2000"
    elif 2001 <= year <= 2010:
        return "B_2001_2010"
    elif 2011 <= year <= 2020:
        return "C_2011_2020"
    else:
        return "D_2021_2025"

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 1: PDF Text Extraction")
    print("=" * 60)

    # Collect all PDF file paths across all year folders
    all_files = []
    for year in YEARS:
        year_dir = os.path.join(BASE_DIR, str(year))
        if not os.path.isdir(year_dir):
            print(f"  [WARN] Year folder missing: {year}")
            continue
        for fname in os.listdir(year_dir):
            if fname.upper().endswith(".PDF"):
                all_files.append({
                    "year": year,
                    "filename": fname,
                    "filepath": os.path.join(year_dir, fname)
                })

    print(f"\nTotal PDFs found: {len(all_files):,}")
    print(f"Saving output to: {OUTPUT_FILE}")
    print(f"Progress saves every {BATCH_SAVE} files\n")

    # Load existing progress if script was interrupted before
    records = []
    processed_files = set()
    if os.path.exists(OUTPUT_FILE):
        print("Found existing extracted.json — resuming from where we left off...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        processed_files = {r["filename"] for r in records}
        print(f"Already processed: {len(processed_files):,} files\n")

    # Process each PDF
    doc_id = len(records)
    failed = 0
    empty = 0

    for item in tqdm(all_files, desc="Extracting PDFs", unit="file"):
        # Skip already processed files (resume support)
        if item["filename"] in processed_files:
            continue

        raw_text = extract_text_from_pdf(item["filepath"])
        word_count = len(raw_text.split()) if raw_text else 0

        if not raw_text:
            failed += 1
            status = "failed"
        elif word_count < 20:
            empty += 1
            status = "too_short"
        else:
            status = "ok"

        record = {
            "id": doc_id,
            "filename": item["filename"],
            "case_name": get_case_name(item["filename"]),
            "year": item["year"],
            "cohort": assign_cohort(item["year"]),
            "raw_text": raw_text,
            "word_count": word_count,
            "status": status
        }

        records.append(record)
        processed_files.add(item["filename"])
        doc_id += 1

        # Save progress every BATCH_SAVE files
        if len(records) % BATCH_SAVE == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            tqdm.write(f"  [Saved] Progress: {len(records):,} records")

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    # ─────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────
    ok_count = sum(1 for r in records if r["status"] == "ok")
    cohort_counts = {}
    for r in records:
        c = r["cohort"]
        cohort_counts[c] = cohort_counts.get(c, 0) + 1

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total processed : {len(records):,}")
    print(f"  Successfully extracted: {ok_count:,}")
    print(f"  Failed (corrupted)    : {failed:,}")
    print(f"  Too short (<20 words) : {empty:,}")
    print(f"\nCohort distribution:")
    for cohort, count in sorted(cohort_counts.items()):
        print(f"    {cohort}: {count:,} cases")
    print(f"\nOutput saved: {OUTPUT_FILE}")
    print("Next step: Run pipeline/02_clean.py")

if __name__ == "__main__":
    main()
