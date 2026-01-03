from __future__ import annotations

import re
from typing import Dict, List, Optional


_BULLET_RE = re.compile(r"^(\s*[-*\\u2022]\s+|\s*\d+[.)]\s+)")
_FIGURE_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d*\s*[:\-\.]?\s*(.*)$", re.IGNORECASE)
_TABLE_RE = re.compile(r"^\s*(table|tbl\.?)\s*\d*\s*[:\-\.]?\s*(.*)$", re.IGNORECASE)


def _normalize_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln]


def _split_columns(line: str) -> Optional[List[str]]:
    if "|" in line:
        parts = [p.strip() for p in line.strip("|").split("|") if p.strip()]
        return parts if len(parts) >= 2 else None

    parts = [p.strip() for p in re.split(r"\s{2,}|\t", line) if p.strip()]
    return parts if len(parts) >= 2 else None


def _table_blocks(lines: List[str]) -> List[List[List[str]]]:
    blocks: List[List[List[str]]] = []
    current: List[List[str]] = []

    def flush():
        nonlocal current
        if len(current) >= 2:
            blocks.append(current)
        current = []

    for ln in lines:
        parts = _split_columns(ln)
        if parts:
            current.append(parts)
        else:
            flush()
    flush()
    return blocks


def _build_markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    header = rows[0] + [""] * (col_count - len(rows[0]))
    body = [r + [""] * (col_count - len(r)) for r in rows[1:]]

    header_line = "| " + " | ".join(header) + " |"
    sep_line = "| " + " | ".join(["---"] * col_count) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in body] if body else []
    return "\n".join([header_line, sep_line] + body_lines)


def parse_slide_text(text: str) -> Dict[str, object]:
    """
    Extracts tables, figures, bullets, and clean text from OCR.
    Returns a dictionary suitable for report generation.
    """
    lines = _normalize_lines(text)
    bullets: List[str] = []
    figures: List[str] = []
    tables_captions: List[str] = []
    body_lines: List[str] = []

    for ln in lines:
        if _BULLET_RE.match(ln):
            bullets.append(_BULLET_RE.sub("- ", ln))
            continue

        fig = _FIGURE_RE.match(ln)
        if fig:
            caption = fig.group(2).strip() or ln
            figures.append(caption)
            continue

        tbl = _TABLE_RE.match(ln)
        if tbl:
            caption = tbl.group(2).strip() or ln
            tables_captions.append(caption)
            continue

        body_lines.append(ln)

    blocks = _table_blocks(body_lines)
    tables = [{"markdown": _build_markdown_table(rows), "rows": rows} for rows in blocks]

    # Remove table lines from clean text
    table_lines = set()
    for rows in blocks:
        for row in rows:
            table_lines.add(" ".join(row).strip())

    clean_lines = [ln for ln in body_lines if " ".join(_split_columns(ln) or [ln]).strip() not in table_lines]
    clean_text = "\n".join(clean_lines).strip()

    return {
        "clean_text": clean_text,
        "bullets": bullets,
        "figures": figures,
        "table_captions": tables_captions,
        "tables": tables,
    }
