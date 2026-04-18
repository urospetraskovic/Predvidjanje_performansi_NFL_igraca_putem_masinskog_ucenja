"""Inspect the existing Izvestaj docx at XML level."""
import zipfile, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SRC = Path(r"c:\Users\Korisnik\Documents\GitHub\Analiza-i-Obrada\Izvestaj\Predviđanje_performansi_NFL_igrača_putem_mašinskog_učenja_Milan_Jovkić_Uroš_Petrašković.docx")

with zipfile.ZipFile(SRC) as z:
    doc = z.read('word/document.xml').decode('utf-8')
    styles = z.read('word/styles.xml').decode('utf-8')

print("=== STYLES listed ===")
# find all <w:style ... w:styleId="...">
for m in re.finditer(r'<w:style\s+[^>]*?w:styleId="([^"]+)"[^>]*?>', styles):
    sid = m.group(1)
    # find the <w:name in the next ~300 chars
    seg = styles[m.start():m.start()+400]
    name_m = re.search(r'<w:name\s+w:val="([^"]+)"', seg)
    ptype = re.search(r'w:type="([^"]+)"', m.group(0))
    print(f"  id={sid!r:30s} type={ptype.group(1) if ptype else '?':10s} name={name_m.group(1) if name_m else '?'}")

print("\n=== All pStyle vals used in document ===")
used = {}
for m in re.finditer(r'<w:pStyle\s+w:val="([^"]+)"', doc):
    used[m.group(1)] = used.get(m.group(1), 0) + 1
for k, v in sorted(used.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Extract text and style per paragraph
print("\n=== All paragraphs (index, style, text) ===")
paras = re.split(r'(?=<w:p[\s>])', doc)
# Strip the first chunk (doesn't start with <w:p)
idx = 0
for para in paras:
    if not para.startswith('<w:p'):
        continue
    pstyle_m = re.search(r'<w:pStyle\s+w:val="([^"]+)"', para)
    style = pstyle_m.group(1) if pstyle_m else '(Normal)'
    text_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para)
    text = ''.join(text_parts).strip()
    if text or 'Heading' in style or 'Naslov' in style:
        print(f"  [{idx:4d}] {style:25s} | {text[:150]}")
    idx += 1
