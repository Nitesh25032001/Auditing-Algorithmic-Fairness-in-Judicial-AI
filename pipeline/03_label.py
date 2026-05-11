"""
STEP 3: OUTCOME LABEL GENERATION
==================================
PURPOSE:
    This is one of the most important steps — extracting the TARGET VARIABLE
    (what we want the ML model to predict) from the judgment text.

WHAT IS THE LABEL?
    For each case, the Supreme Court gives one of these outcomes:
    - ALLOWED  (1): The appellant wins. The lower court decision is reversed.
    - DISMISSED (0): The appellant loses. The lower court decision is upheld.
    - PARTIAL  (2): Mixed outcome. Partly allowed, partly dismissed.

HOW WE EXTRACT IT:
    We search the LAST 2000 characters of each judgment (the conclusion/
    operative part) for specific legal keywords. The last section always
    contains the final order.

WHY LAST 2000 CHARS?
    Indian Supreme Court judgments end with phrases like:
    "The appeal is accordingly allowed."
    "The appeal is dismissed."
    These appear in the operative/conclusion paragraph at the end.

ADDITIONAL METADATA EXTRACTED:
    - case_type: Criminal / Civil / Constitutional / Other
    - bench_size: 1 judge, 2 judges, 3 judges (Constitution bench), 5+
    - word_count_quartile: Short / Medium / Long (controls for length bias)

OUTPUT:
    data_processed/labeled.csv
"""

import json
import re
import os
import csv
from tqdm import tqdm
from collections import Counter

BASE_DIR    = r"c:\Users\nites\OneDrive\Desktop\data"
INPUT_FILE  = os.path.join(BASE_DIR, "data_processed", "cleaned.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data_processed", "labeled.csv")

# ─────────────────────────────────────────────
# KEYWORD PATTERNS FOR OUTCOME EXTRACTION
# These are standard legal phrases used in Indian Supreme Court orders
# ─────────────────────────────────────────────

# ALLOWED keywords — the appeal was granted
ALLOWED_PATTERNS = [
    r'\bappeal[s]?\s+(?:is|are|hereby|stands?)\s+allowed\b',
    r'\bappeals?\s+allowed\b',
    r'\ballow(?:ed|s)?\s+(?:the\s+)?appeal[s]?\b',
    r'\bimpugned\s+(?:judgment|order|decision)\s+(?:is\s+)?(?:set\s+aside|quashed)\b',
    r'\bset\s+aside\s+(?:and|the)\s+(?:appeal|matter)\b',
    r'\border\s+(?:of\s+the\s+)?(?:high\s+court|lower\s+court|tribunal)\s+(?:is\s+)?set\s+aside\b',
    r'\baccordingly\s+allowed\b',
    r'\bhereby\s+allowed\b',
]

# DISMISSED keywords — the appeal was rejected
DISMISSED_PATTERNS = [
    r'\bappeal[s]?\s+(?:is|are|hereby|stands?)\s+dismissed\b',
    r'\bappeals?\s+dismissed\b',
    r'\bdismiss(?:ed|es|ing)?\s+(?:the\s+)?appeal[s]?\b',
    r'\baccordingly\s+dismissed\b',
    r'\bhereby\s+dismissed\b',
    r'\bno\s+merit\b.{0,50}\bdismiss\b',
]

# PARTIAL keywords — mixed outcome
PARTIAL_PATTERNS = [
    r'\bpartly\s+allowed\b',
    r'\bpartially\s+allowed\b',
    r'\ballowed\s+in\s+part\b',
    r'\bpartly\s+dismissed\b',
    r'\bpartially\s+dismissed\b',
    r'\bmodif(?:y|ied|ies)\b.{0,100}\bappeal\b',
    r'\bappeal[s]?\s+(?:is|are)\s+partly\b',
]

def extract_outcome(text):
    """
    Searches the last 2000 characters of the judgment for outcome keywords.
    Returns: 0 (Dismissed), 1 (Allowed), 2 (Partial), or -1 (Unknown)
    """
    # Focus on the operative part (conclusion) of the judgment
    conclusion = text[-2000:].lower() if len(text) > 2000 else text.lower()

    # Count matches for each category
    allowed_score  = sum(1 for p in ALLOWED_PATTERNS  if re.search(p, conclusion))
    dismissed_score= sum(1 for p in DISMISSED_PATTERNS if re.search(p, conclusion))
    partial_score  = sum(1 for p in PARTIAL_PATTERNS   if re.search(p, conclusion))

    # If partial signals found, that takes priority
    if partial_score > 0:
        return 2

    # If clear allowed or dismissed
    if allowed_score > dismissed_score:
        return 1
    elif dismissed_score > allowed_score:
        return 0
    elif allowed_score > 0 and allowed_score == dismissed_score:
        return 2  # Ambiguous = treat as partial

    return -1  # Unknown

def extract_case_type(filename, text):
    """
    Determines whether the case is Criminal, Civil, Constitutional, or Other.
    Based on filename keywords and petition type in text.
    """
    filename_lower = filename.lower()
    text_lower = text[:500].lower()

    if any(k in filename_lower for k in ['criminal', 'murder', 'crpc', 'ipc']):
        return "Criminal"
    if any(k in text_lower for k in ['criminal appeal', 'ipc', 'crpc', 'murder', 'rape', 'theft']):
        return "Criminal"
    if any(k in text_lower for k in ['writ petition', 'article 32', 'article 226', 'fundamental right']):
        return "Constitutional"
    if any(k in text_lower for k in ['civil appeal', 'property', 'contract', 'damages', 'rent']):
        return "Civil"
    return "Other"

def extract_bench_size(text):
    """
    Estimates the number of judges on the bench.
    Constitution benches (3, 5, 7 judges) are significant in Indian law.
    """
    # Look for judge signatures in text
    j_patterns = [
        r'\(([A-Z][A-Z\s\.]+J\.)\)',   # (JUSTICE NAME J.)
        r'\b([A-Z][A-Z\s\.]+),?\s*J\.\b',
    ]
    judges = set()
    for p in j_patterns:
        matches = re.findall(p, text[:3000])
        judges.update(matches)

    count = len(judges)
    if count >= 5:
        return 5
    elif count == 3:
        return 3
    elif count == 2:
        return 2
    else:
        return 1  # Default single bench

def main():
    print("=" * 60)
    print("JUDICIAL AI FAIRNESS PROJECT")
    print("Step 3: Outcome Label Generation")
    print("=" * 60)

    print(f"\nReading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Total records: {len(records):,}")

    labeled = []
    unknown_count = 0

    for record in tqdm(records, desc="Extracting labels", unit="doc"):
        text = record["clean_text"]

        outcome = extract_outcome(text)
        case_type = extract_case_type(record["filename"], text)
        bench_size = extract_bench_size(text)

        if outcome == -1:
            unknown_count += 1

        labeled.append({
            "id":           record["id"],
            "filename":     record["filename"],
            "case_name":    record.get("case_name", ""),
            "year":         record["year"],
            "cohort":       record["cohort"],
            "outcome":      outcome,          # 0=Dismissed, 1=Allowed, 2=Partial, -1=Unknown
            "case_type":    case_type,
            "bench_size":   bench_size,
            "word_count":   record["word_count"],
            "clean_text":   text,
        })

    # Save to CSV (text column saved separately)
    fieldnames = ["id","filename","case_name","year","cohort",
                  "outcome","case_type","bench_size","word_count","clean_text"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    # ─────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────
    labeled_only = [r for r in labeled if r["outcome"] != -1]
    outcome_counts = Counter(r["outcome"] for r in labeled)
    labels = {0: "Dismissed", 1: "Allowed", 2: "Partial", -1: "Unknown"}

    print("\n" + "=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)
    print(f"  Total labeled      : {len(labeled):,}")
    print(f"  Successfully labeled: {len(labeled_only):,}")
    print(f"  Unknown outcome    : {unknown_count:,} ({unknown_count/len(labeled)*100:.1f}%)")
    print(f"\nOutcome distribution:")
    for code, count in sorted(outcome_counts.items()):
        pct = count / len(labeled) * 100
        print(f"    {labels[code]:12}: {count:,}  ({pct:.1f}%)")

    print(f"\nCase type distribution:")
    ct_counts = Counter(r["case_type"] for r in labeled)
    for ct, count in ct_counts.most_common():
        print(f"    {ct:20}: {count:,}")

    print(f"\nCohort x Outcome (key for fairness analysis):")
    for cohort in sorted(set(r["cohort"] for r in labeled)):
        cohort_recs = [r for r in labeled if r["cohort"] == cohort and r["outcome"] != -1]
        if cohort_recs:
            allow_rate = sum(1 for r in cohort_recs if r["outcome"] == 1) / len(cohort_recs) * 100
            print(f"    {cohort}: {len(cohort_recs):,} cases | Allow rate: {allow_rate:.1f}%")

    print(f"\nOutput saved: {OUTPUT_FILE}")
    print("Next step: Run pipeline/04_featurize.py")

if __name__ == "__main__":
    main()
