"""Convert Strict OOXML (.docx) to Transitional OOXML so python-docx can read it.

The file Word generated uses ISO/IEC 29500 Strict namespaces (purl.oclc.org/ooxml/...).
python-docx only understands Transitional namespaces (schemas.openxmlformats.org/...).
This rewrites every XML part inside the .docx by substituting the namespace URLs.
"""
from __future__ import annotations
import io
import sys
import zipfile
from pathlib import Path

NS_MAP = {
    "http://purl.oclc.org/ooxml/officeDocument/relationships":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://purl.oclc.org/ooxml/officeDocument/math":
        "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "http://purl.oclc.org/ooxml/officeDocument/customProperties":
        "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
    "http://purl.oclc.org/ooxml/officeDocument/extendedProperties":
        "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "http://purl.oclc.org/ooxml/officeDocument/coreProperties":
        "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "http://purl.oclc.org/ooxml/officeDocument/customXml":
        "http://schemas.openxmlformats.org/officeDocument/2006/customXml",
    "http://purl.oclc.org/ooxml/officeDocument/bibliography":
        "http://schemas.openxmlformats.org/officeDocument/2006/bibliography",
    "http://purl.oclc.org/ooxml/wordprocessingml/main":
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/main":
        "http://schemas.openxmlformats.org/drawingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/picture":
        "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing":
        "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "http://purl.oclc.org/ooxml/drawingml/spreadsheetDrawing":
        "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "http://purl.oclc.org/ooxml/drawingml/diagram":
        "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "http://purl.oclc.org/ooxml/drawingml/chart":
        "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "http://purl.oclc.org/ooxml/drawingml/chartDrawing":
        "http://schemas.openxmlformats.org/drawingml/2006/chartDrawing",
    "http://purl.oclc.org/ooxml/drawingml/compatibility":
        "http://schemas.openxmlformats.org/drawingml/2006/compatibility",
    "http://purl.oclc.org/ooxml/drawingml/lockedCanvas":
        "http://schemas.openxmlformats.org/drawingml/2006/lockedCanvas",
    "http://purl.oclc.org/ooxml/schemaLibrary":
        "http://schemas.openxmlformats.org/schemaLibrary/2006/main",
}

XML_LIKE_EXT = (".xml", ".rels")


def convert(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin:
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.lower().endswith(XML_LIKE_EXT):
                    text = data.decode("utf-8")
                    for old, new in NS_MAP.items():
                        text = text.replace(old, new)
                    data = text.encode("utf-8")
                zout.writestr(info, data)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    convert(src, dst)
    print(f"Wrote {dst}")
