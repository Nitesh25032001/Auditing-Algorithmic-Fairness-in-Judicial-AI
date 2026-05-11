import json, os
f = r'c:\Users\nites\OneDrive\Desktop\data\data_processed\extracted.json'
if not os.path.exists(f):
    print("extracted.json not yet created.")
else:
    size_mb = round(os.path.getsize(f) / 1024 / 1024, 1)
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    total = len(data)
    ok    = sum(1 for r in data if r.get('status') == 'ok')
    fail  = sum(1 for r in data if r.get('status') == 'failed')
    short = sum(1 for r in data if r.get('status') == 'too_short')
    pct   = round(total / 13958 * 100, 1)
    print(f"Records saved : {total:,} / 13,958  ({pct}% complete)")
    print(f"Successful    : {ok:,}")
    print(f"Failed        : {fail:,}")
    print(f"Too short     : {short:,}")
    print(f"File size     : {size_mb} MB")
    cohorts = {}
    for r in data:
        c = r.get('cohort', '?')
        cohorts[c] = cohorts.get(c, 0) + 1
    print("\nCohort breakdown:")
    for c, n in sorted(cohorts.items()):
        print(f"  {c}: {n:,}")
