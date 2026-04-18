"""Extract raw XML for key paragraph types and tables."""
import zipfile, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SRC = Path(r"c:\Users\Korisnik\Documents\GitHub\Analiza-i-Obrada\Izvestaj\Predviđanje_performansi_NFL_igrača_putem_mašinskog_učenja_Milan_Jovkić_Uroš_Petrašković.docx")
with zipfile.ZipFile(SRC) as z:
    doc = z.read('word/document.xml').decode('utf-8')

# Find a Heading1 paragraph XML (e.g., "ZAKLJUCAK")
print("=== Heading1 paragraph (ZAKLJUCAK) ===")
m = re.search(r'<w:p\s[^>]*>(?:(?!<w:p\s|<w:p>).)*?ZAKLJUCAK.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

print("=== BodyText paragraph (first after Heading1 of ZAKLJUCAK, short sample) ===")
# Find first Heading1 we skip, then BodyText
# Simpler: find 'Uvod' and find the paragraph
m = re.search(r'<w:p\s[^>]*>(?:(?!<w:p\s|<w:p>).)*?Uvod</w:t>.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

print("=== LITERATURA paragraph (Normal style heading) ===")
m = re.search(r'<w:p\s[^>]*>(?:(?!<w:p\s|<w:p>).)*?LITERATURA</w:t>.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

# Find first Caption (Tabela 1 ...)
print("=== Caption paragraph (Tabela 1) ===")
m = re.search(r'<w:p\s[^>]*>(?:(?!<w:p\s|<w:p>).)*?Tabela 1.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

# Find a short BodyText paragraph content
print("=== Body text paragraph (first paragraph of Uvod section) ===")
# "Predviđanje performansi individualnih igrača"
m = re.search(r'<w:p\s[^>]*>(?:(?!<w:p\s|<w:p>).)*?Predviđanje performansi individualnih.*?</w:p>', doc, re.S)
if m:
    sample = m.group(0)
    # show only first 1500 chars
    print(sample[:2000])
print()

# Find a table XML (first table)
print("=== First table XML (first 3000 chars) ===")
m = re.search(r'<w:tbl>.*?</w:tbl>', doc, re.S)
if m:
    print(m.group(0)[:3000])
print()

# Show sectPr (very end of body)
print("=== sectPr (end of body) ===")
m = re.search(r'<w:sectPr[^/>]*(?:/>|>.*?</w:sectPr>)', doc, re.S)
if m:
    print(m.group(0))
