import pdfplumber, os

BASE = r'c:\Users\nites\OneDrive\Desktop\data'
test_files = []
for year in [1990, 2000, 2010, 2020, 2024]:
    yr_dir = os.path.join(BASE, str(year))
    pdfs = [f for f in os.listdir(yr_dir) if f.upper().endswith('.PDF')][:2]
    for p in pdfs:
        test_files.append((year, p, os.path.join(yr_dir, p)))

print(f'Testing on {len(test_files)} sample PDFs...\n')
results = []
for year, fname, fpath in test_files:
    try:
        with pdfplumber.open(fpath) as pdf:
            text = ' '.join(p.extract_text() or '' for p in pdf.pages)
        words = len(text.split())
        status = 'OK' if words > 50 else 'TOO SHORT'
        print(f'  [{year}] {fname[:55]}')
        print(f'          Words: {words:,} | Status: {status}')
        results.append({'year': year, 'words': words, 'status': status})
    except Exception as e:
        print(f'  [{year}] FAILED: {e}')
        results.append({'year': year, 'words': 0, 'status': 'FAILED'})

ok = sum(1 for r in results if r['status'] == 'OK')
total_words = sum(r['words'] for r in results)
avg_words = int(total_words / len(results)) if results else 0
print(f'\nSample result: {ok}/{len(results)} PDFs extracted successfully')
print(f'Average word count per document: {avg_words:,} words')
print(f'Estimated total words across all 13,958 PDFs: ~{avg_words * 13958:,}')
