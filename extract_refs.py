import os
import pdfplumber

ref_dir = r"c:\Users\nites\OneDrive\Desktop\data\references\Ai - Courts"
output_file = r"c:\Users\nites\OneDrive\Desktop\data\ref_summary.txt"

with open(output_file, 'w', encoding='utf-8') as out:
    for filename in os.listdir(ref_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(ref_dir, filename)
            try:
                with pdfplumber.open(filepath) as pdf:
                    first_page = pdf.pages[0].extract_text()
                    if first_page:
                        # take the first 1000 characters to get the title and abstract
                        text = first_page[:1000].replace('\n', ' ')
                        out.write(f"FILE: {filename}\nTEXT: {text}\n{'-'*50}\n")
            except Exception as e:
                out.write(f"FILE: {filename}\nERROR: {e}\n{'-'*50}\n")
