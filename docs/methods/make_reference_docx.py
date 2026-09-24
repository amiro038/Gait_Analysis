# -*- coding: utf-8 -*-
"""Build reference.docx: the styles pandoc uses for Gait_Analysis_Methods.docx
(fonts, headings, captions, tables, the three coloured boxes, page numbers).
Run by build_methods_doc.py; needs python-docx."""

import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
NAVY, BLUE, GREY = RGBColor(0x1F, 0x38, 0x64), RGBColor(0x2E, 0x55, 0x97), RGBColor(0x59, 0x59, 0x59)
BODY_FONT, CODE_FONT = "Calibri", "Consolas"


def shade(pPr_or_tcPr, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr_or_tcPr.append(shd)


def left_border(pPr, colour, size=18):
    bdr = OxmlElement("w:pBdr")
    for side in ("left",):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), str(size))
        b.set(qn("w:space"), "8")
        b.set(qn("w:color"), colour)
        bdr.append(b)
    pPr.append(bdr)


def font(style, name=BODY_FONT, size=None, bold=None, italic=None, colour=None):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(k), name)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(k)) is not None:
            del fonts.attrib[qn(k)]
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if colour is not None:
        f.color.rgb = colour


def para_style(doc, name, base="Normal"):
    try:
        return doc.styles[name]
    except KeyError:
        s = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = doc.styles[base]
        return s


def main(out):
    pandoc = __import__("pypandoc").get_pandoc_path()
    default = HERE / "_ref_default.docx"
    subprocess.run([pandoc, "-o", str(default), "--print-default-data-file",
                    "reference.docx"], check=True)
    doc = Document(str(default))
    st = doc.styles

    # page: US Letter, 1 inch margins -> 6.5 in of text
    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = Inches(1)
        s.top_margin = s.bottom_margin = Inches(1)
        # page number, centred in the footer
        p = s.footer.paragraphs[0] if s.footer.paragraphs else s.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                r._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = text
                r._r.append(it)
        r.font.size = Pt(9)
        r.font.color.rgb = GREY

    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        try:
            font(st[name], size=11)
        except KeyError:
            pass
    for name in ("Body Text", "First Paragraph"):
        pf = st[name].paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.12
    st["Compact"].paragraph_format.space_after = Pt(2)

    heads = {"Title": (24, NAVY, True), "Subtitle": (14, BLUE, False),
             "Heading 1": (17, NAVY, True), "Heading 2": (13.5, BLUE, True),
             "Heading 3": (11.5, BLUE, True), "Heading 4": (11, GREY, True)}
    for name, (size, colour, bold) in heads.items():
        font(st[name], size=size, bold=bold, colour=colour, italic=False)
        pf = st[name].paragraph_format
        pf.keep_with_next = True
        if name == "Heading 1":
            pf.page_break_before = True
            pf.space_before, pf.space_after = Pt(0), Pt(10)
        elif name == "Heading 2":
            pf.space_before, pf.space_after = Pt(14), Pt(4)
        elif name == "Heading 3":
            pf.space_before, pf.space_after = Pt(10), Pt(3)
    font(st["Date"], size=11, colour=GREY)
    font(st["Author"], size=11, colour=GREY)

    for name in ("Image Caption", "Table Caption", "Caption"):
        try:
            s = next(x for x in st if x.name == name)
        except StopIteration:
            continue
        font(s, size=9.5, italic=False, colour=GREY)
        s.paragraph_format.space_before, s.paragraph_format.space_after = Pt(2), Pt(12)
    st["Table Caption"].paragraph_format.keep_with_next = True
    st["Table Caption"].paragraph_format.space_before = Pt(10)
    st["Table Caption"].paragraph_format.space_after = Pt(4)
    try:
        st["Captioned Figure"].paragraph_format.keep_with_next = True
    except KeyError:
        pass
    font(st["Verbatim Char"], name=CODE_FONT, size=9.5, colour=RGBColor(0x1F, 0x38, 0x64))
    try:
        font(st["Source Code"], name=CODE_FONT, size=9)
    except KeyError:
        pass

    # three boxes: in the code / why / healthy range
    for name, fill, edge in (("Code Box", "F2F4F7", "2E5597"),
                             ("Why Box", "EEF4FB", "7FA7D9"),
                             ("Range Box", "EAF4E4", "5E9E4B"),
                             ("Note Box", "FFF6E0", "E0A400")):
        s = para_style(doc, name, "Body Text")
        font(s, size=10)
        pf = s.paragraph_format
        pf.space_before, pf.space_after = Pt(3), Pt(3)
        pf.left_indent, pf.right_indent = Inches(0.12), Inches(0.05)
        pPr = s.element.get_or_add_pPr()
        shade(pPr, fill)
        left_border(pPr, edge)

    # tables: thin grey rules, shaded header row, small text
    tbl = st["Table"]
    tblPr = tbl.element.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.element.append(tblPr)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "bottom", "insideH"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "A6A6A6")
        borders.append(b)
    tblPr.append(borders)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("left", 80), ("right", 80), ("top", 20), ("bottom", 20)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(w))
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    tblPr.append(mar)
    rpr = tbl.element.get_or_add_rPr() if hasattr(tbl.element, "get_or_add_rPr") else None
    if rpr is not None:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "18")
        rpr.append(sz)
    first = OxmlElement("w:tblStylePr")
    first.set(qn("w:type"), "firstRow")
    fr = OxmlElement("w:rPr")
    fr.append(OxmlElement("w:b"))
    first.append(fr)
    tcPr = OxmlElement("w:tcPr")
    shade(tcPr, "DDE6F0")
    first.append(tcPr)
    tbl.element.append(first)

    # contents entries: "toc 1" / "toc 2", page number right-aligned after dots
    for level, (indent, bold, before) in {1: (0.0, True, 6), 2: (0.25, False, 0)}.items():
        s = doc.styles.add_style(f"toc {level}", WD_STYLE_TYPE.PARAGRAPH)
        s.element.set(qn("w:styleId"), f"TOC{level}")
        s.base_style = st["Normal"]
        font(s, size=10.5 if level == 1 else 10, bold=bold)
        pf = s.paragraph_format
        pf.left_indent = Inches(indent)
        pf.space_before, pf.space_after = Pt(before), Pt(1)
        pf.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    font(st["TOC Heading"], size=17, bold=True, colour=NAVY)

    doc.save(str(out))
    default.unlink()
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "reference.docx"))
