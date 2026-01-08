# Better structured PDF using ReportLab Platypus (recommended)
from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    ListFlowable,
    ListItem,
    KeepTogether,
)
from reportlab.lib import colors


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_story_metrics(payload):
    slides = payload.get("slides", []) if isinstance(payload, dict) else []
    slide_ids, wpms, jaccards, fillers = [], [], [], []
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        sid = slide.get("slide_id")
        speech = slide.get("speech") or {}
        align = slide.get("content_alignment") or {}
        sim = (align.get("similarity") or {}).get("jaccard")
        slide_ids.append(str(sid))
        wpms.append(_safe_float(speech.get("wpm"), 0.0) or 0.0)
        jaccards.append(_safe_float(sim, 0.0) or 0.0)
        filler_count = (speech.get("filler_word_count") or 0) + (speech.get("filler_phrase_count") or 0)
        fillers.append(int(filler_count))
    return slide_ids, wpms, jaccards, fillers


def _extract_visual_metrics(payload):
    overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
    gaze = (overall.get("gaze") or {}).get("overall_summary") or {}
    gaze_dist = gaze.get("focus_dist") or {}
    gesture = (overall.get("gesture") or {}).get("overall") or {}
    joint_stats = (gesture.get("joint_statistics") or {})
    open_palms = _safe_float(joint_stats.get("open_palms_pct"), 0.0) or 0.0
    emotion = (overall.get("emotion") or {}).get("overall_stats") or {}
    emotion_dist = emotion.get("average_scores") or {}
    return gaze_dist, open_palms, emotion_dist


def _make_summary_chart(content_payload, visual_payload):
    slide_ids, wpms, jaccards, fillers = _extract_story_metrics(content_payload)
    gaze_dist, open_palms, emotion_dist = _extract_visual_metrics(visual_payload)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    ax = axes[0, 0]
    if slide_ids:
        ax.bar(slide_ids, wpms)
        ax.set_title("WPM by slide")
        ax.set_xlabel("Slide")
        ax.set_ylabel("WPM")
    else:
        ax.text(0.5, 0.5, "No WPM data", ha="center", va="center")
        ax.axis("off")

    ax = axes[0, 1]
    if slide_ids:
        ax.bar(slide_ids, jaccards)
        ax.set_title("Content alignment (Jaccard)")
        ax.set_xlabel("Slide")
        ax.set_ylabel("Jaccard")
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No alignment data", ha="center", va="center")
        ax.axis("off")

    ax = axes[1, 0]
    if gaze_dist:
        labels = list(gaze_dist.keys())
        values = [gaze_dist[k] for k in labels]
        ax.pie(values, labels=labels, autopct="%1.0f%%")
        ax.set_title("Gaze distribution")
    else:
        ax.text(0.5, 0.5, "No gaze data", ha="center", va="center")
        ax.axis("off")

    ax = axes[1, 1]
    if emotion_dist:
        labels = list(emotion_dist.keys())
        values = [emotion_dist[k] for k in labels]
        ax.bar(labels, values)
        ax.set_title("Average emotion scores")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, f"Open palms: {open_palms:.1f}%", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    return fig


def _build_styles():
    base = getSampleStyleSheet()

    # Tweak defaults a bit for readability
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 10
    base["Normal"].leading = 13

    h1 = ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4)

    body = ParagraphStyle("Body", parent=base["Normal"], spaceAfter=6)
    # Monospace block for excerpts
    mono = ParagraphStyle(
        "Mono",
        parent=base["Normal"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        backColor=colors.whitesmoke,
        borderPadding=6,
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=8,
    )

    return {"H1": h1, "H2": h2, "H3": h3, "Body": body, "Mono": mono}


def _escape_paragraph(text: str) -> str:
    # ReportLab Paragraph uses a mini-HTML; escape the critical characters.
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _looks_like_excerpt_line(line: str) -> bool:
    # Heuristic: your report uses "- **Content excerpt**:" etc.
    return ("excerpt" in line.lower()) or ("alignment check" in line.lower())


def _markdown_to_flowables(md_text: str, styles):
    """
    Minimal markdown-ish renderer for your specific report format:
    - # / ## / ### headings
    - bullets "- "
    - numbered lists "1. "
    - bold markers **...** (Paragraph supports <b>...</b>)
    - excerpt-ish lines rendered as monospace blocks for readability
    """
    story = []

    bullet_buf = []
    number_buf = []
    para_buf = []

    def flush_paragraph():
        nonlocal para_buf
        if not para_buf:
            return
        text = " ".join(s.strip() for s in para_buf).strip()
        if text:
            story.append(Paragraph(_md_inline_to_html(_escape_paragraph(text)), styles["Body"]))
        para_buf = []

    def flush_bullets():
        nonlocal bullet_buf
        if not bullet_buf:
            return
        items = [ListItem(Paragraph(_md_inline_to_html(_escape_paragraph(t)), styles["Body"])) for t in bullet_buf]
        story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletIndent=6))
        story.append(Spacer(1, 6))
        bullet_buf = []

    def flush_numbers():
        nonlocal number_buf
        if not number_buf:
            return
        items = [ListItem(Paragraph(_md_inline_to_html(_escape_paragraph(t)), styles["Body"])) for t in number_buf]
        story.append(ListFlowable(items, bulletType="1", leftIndent=22, bulletIndent=6))
        story.append(Spacer(1, 6))
        number_buf = []

    def flush_all():
        flush_paragraph()
        flush_bullets()
        flush_numbers()

    lines = md_text.splitlines()
    for raw in lines:
        line = raw.rstrip()

        if not line.strip():
            flush_all()
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            flush_all()
            story.append(Paragraph(_escape_paragraph(line[2:].strip()), styles["H1"]))
            continue

        if line.startswith("## "):
            flush_all()
            story.append(Paragraph(_escape_paragraph(line[3:].strip()), styles["H2"]))
            continue

        if line.startswith("### "):
            flush_all()
            story.append(Paragraph(_escape_paragraph(line[4:].strip()), styles["H3"]))
            continue

        # Bullet
        if line.lstrip().startswith("- "):
            flush_paragraph()
            flush_numbers()
            bullet_buf.append(line.lstrip()[2:].strip())
            continue

        # Numbered list: "1. " "2. " ...
        stripped = line.lstrip()
        if len(stripped) >= 4 and stripped[0].isdigit() and stripped[1].isdigit() is False and stripped[1:3] == ". ":
            flush_paragraph()
            flush_bullets()
            number_buf.append(stripped[3:].strip())
            continue

        # Excerpt-ish lines: put into monospace blocks to avoid ugly wrapping/indent issues
        if _looks_like_excerpt_line(line):
            flush_all()
            story.append(Paragraph(_escape_paragraph(line.strip()), styles["Mono"]))
            continue

        # Default: accumulate as paragraph text
        flush_bullets()
        flush_numbers()
        para_buf.append(line.strip())

    flush_all()
    return story


def _md_inline_to_html(text: str) -> str:
    # Convert **bold** to <b>bold</b> for Paragraph.
    # Keep it simple and safe for your report format.
    out = []
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                out.append("<b>")
                out.append(text[i + 2:j])
                out.append("</b>")
                i = j + 2
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def build_pdf_report(story_text, visual_text, content_payload, visual_payload, output_path):
    output_path = Path(output_path)
    styles = _build_styles()

    # --- create chart image ---
    fig = _make_summary_chart(content_payload, visual_payload)
    tmp_dir = Path(tempfile.mkdtemp())
    chart_path = tmp_dir / "summary_charts.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Presentation Feedback Report",
    )

    story = []

    # Cover / Title
    story.append(Paragraph("Presentation Feedback Report", styles["H1"]))
    story.append(Spacer(1, 8))

    # Chart section
    if chart_path.exists():
        # Fit chart to width
        img = Image(str(chart_path))
        img.drawWidth = doc.width
        img.drawHeight = img.drawHeight * (doc.width / img.drawWidth)
        img._restrictSize(doc.width, 3.2 * inch)
        story.append(img)
        story.append(Spacer(1, 12))

    # Slide + Speech section (keep heading with first content)
    slide_section = []
    slide_section.append(Paragraph("Slide + Speech Feedback Report", styles["H2"]))
    slide_section.extend(_markdown_to_flowables(story_text, styles))
    story.append(KeepTogether(slide_section))

    story.append(PageBreak())

    # Visual Coaching section
    visual_section = []
    visual_section.append(Paragraph("Visual Coaching Report", styles["H2"]))
    visual_section.extend(_markdown_to_flowables(visual_text, styles))
    story.append(KeepTogether(visual_section))

    doc.build(story)
    return output_path
