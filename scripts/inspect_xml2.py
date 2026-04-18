"""More XML samples: Caption, Heading2, references."""
import zipfile, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SRC = Path(r"c:\Users\Korisnik\Documents\GitHub\Analiza-i-Obrada\Izvestaj\Predviđanje_performansi_NFL_igrača_putem_mašinskog_učenja_Milan_Jovkić_Uroš_Petrašković.docx")
with zipfile.ZipFile(SRC) as z:
    doc = z.read('word/document.xml').decode('utf-8')

# Find a Caption paragraph XML
print("=== Caption paragraph (Tabela 2 GirdSearch QB) ===")
m = re.search(r'<w:p\s[^>]*>[^<]*<w:pPr><w:pStyle w:val="Caption"/></w:pPr>.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

# Heading2 paragraph
print("=== Heading2 paragraph ===")
m = re.search(r'<w:p\s[^>]*>[^<]*<w:pPr><w:pStyle w:val="Heading2"/></w:pPr>.*?</w:p>', doc, re.S)
if m:
    print(m.group(0))
print()

# First references paragraph
print("=== references paragraph ===")
m = re.search(r'<w:p\s[^>]*>[^<]*<w:pPr><w:pStyle w:val="references"/></w:pPr>.*?</w:p>', doc, re.S)
if m:
    print(m.group(0)[:1500])
print()

# End of body
print("=== End of document (last 2000 chars) ===")
print(doc[-2000:])

# Find exact location of LITERATURA
print("\n=== LITERATURA paragraph ===")
idx = doc.find('LITERATURA')
print(f"Position: {idx}")
# Find the <w:p start before it
p_start = doc.rfind('<w:p ', 0, idx)
print("Paragraph XML:")
# find </w:p> after idx
p_end = doc.find('</w:p>', idx) + len('</w:p>')
print(doc[p_start:p_end])
