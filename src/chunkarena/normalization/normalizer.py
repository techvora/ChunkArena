"""Stage 2 — Normalization (bronze extraction JSON → silver atomic units).

Converts the raw, source-shaped JSON produced by Stage 1 into a flat list
of atomic units — headings, paragraphs, images, tables and formulas —
each with a stable type tag, a sequential id and cleaned text. The silver
output is the single input every chunking strategy in Stage 3 reads, so
any cleanup that should affect every chunker (whitespace collapsing, PDF
escape unwrap, URL de-spacing, markdown heading detection) lives here
and nowhere else.

Exposes small, pure helpers (clean_text, clean_url, is_heading,
extract_images_and_remainder) that are covered by tests/unit/, plus a
top-level normalize_raw_json(input, output) driver invoked by
scripts/run_normalization.py.
"""

import json
import re
from typing import List, Dict, Any, Tuple

def clean_text(text: str) -> str:
    """Remove excessive whitespace and fix common escape sequences."""
    text = text.replace("\\_", "_")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_url(url: str) -> str:
    """Remove spaces from inside a URL (PDF extraction artifact)."""
    return re.sub(r'\s+', '', url)

def extract_text(raw_data: Any) -> str:
    """Extract the main content from the raw JSON structure."""
    if isinstance(raw_data, dict):
        return raw_data.get("content", "")
    elif isinstance(raw_data, list):
        return "\n".join(
            item.get("content") or item.get("text", "")
            for item in raw_data if isinstance(item, dict)
        )
    return ""

def is_heading(line: str) -> Tuple[bool, int]:
    """Detect markdown heading. Returns (is_heading, level)."""
    stripped = line.lstrip()
    if stripped.startswith('#'):
        level = 0
        for ch in stripped:
            if ch == '#':
                level += 1
            else:
                break
        if level > 0 and len(stripped) > level and stripped[level] == ' ':
            content = stripped[level:].lstrip()
            if content.lower() == "image:":
                return False, 0
            return True, min(level, 6)
    return False, 0

def extract_images_and_remainder(line: str) -> Tuple[List[str], str]:
    """Extract image URLs from a line and return (urls, remaining_text)."""
    urls = []
    remainder = line

    if line.lower().startswith("image:"):
        after_prefix = line[6:].lstrip()
        pattern = r'https?://\S+'
        found_urls = re.findall(pattern, after_prefix)
        if found_urls:
            urls.extend(found_urls)
            for url in found_urls:
                after_prefix = after_prefix.replace(url, '', 1)
            remainder = after_prefix.strip()
        else:
            remainder = after_prefix
        return urls, remainder

    md_pattern = r'!\[.*?\]\((https?://[^\s)]+)\)'
    md_urls = re.findall(md_pattern, line)
    if md_urls:
        urls.extend(md_urls)
        temp = line
        for url in md_urls:
            temp = re.sub(r'!\[.*?\]\(' + re.escape(url) + r'\)', '', temp, count=1)
        remainder = temp.strip()
        return urls, remainder

    stripped_line = line.strip()
    if stripped_line.startswith('http'):
        urls.append(stripped_line)
        remainder = ""
        return urls, remainder

    return [], line

def is_table_row(line: str) -> bool:
    """Check if line is a markdown table row."""
    stripped = line.strip()
    if stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2:
        return True
    return False

def is_table_separator(line: str) -> bool:
    """Check if line is a markdown table separator."""
    stripped = line.strip()
    if stripped.startswith('|') and stripped.endswith('|'):
        if re.match(r'^[\|\s\-:]+$', stripped):
            return True
    return False

def has_formula(line: str) -> bool:
    """
    Detect LaTeX formulas while ignoring currency amounts (e.g., $100, US$96.4).
    Returns True only for actual math expressions.
    """
    # Display math: $$ ... $$
    if re.search(r'\$\$[^\$]+\$\$', line):
        return True

    # Inline math: $ ... $ (but not $$)
    matches = re.findall(r'(?<!\$)\$(?!\$)([^\$]+?)(?<!\$)\$(?!\$)', line)
    for match in matches:
        # Remove common currency prefixes (US$, CA$, etc.)
        content = re.sub(r'^[A-Z]{2,}\s*', '', match.strip())
        # If content is purely numeric (digits, commas, dots, spaces, %), it's currency
        if re.match(r'^[\d\.,\s%]+$', content):
            continue
        # If content contains LaTeX commands or math symbols, it's a formula
        if re.search(r'\\[a-zA-Z]+|[\^_{}]', content):
            return True
    return False

def normalize_raw_json(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_text = extract_text(raw_data)
    lines = raw_text.splitlines()

    normalized: List[Dict[str, Any]] = []
    position = 0
    sec_counter = 1
    para_counter = 1
    img_counter = 1
    table_counter = 1
    formula_counter = 1

    current_paragraph_lines: List[str] = []
    current_table_rows: List[str] = []
    in_table = False

    def flush_paragraph():
        nonlocal position, para_counter, current_paragraph_lines
        if current_paragraph_lines:
            para_text = " ".join(current_paragraph_lines).strip()
            if para_text:
                normalized.append({
                    "id": f"p_{para_counter}",
                    "type": "paragraph",
                    "content": clean_text(para_text),
                    "position": position
                })
                position += 1
                para_counter += 1
                current_paragraph_lines = []

    def flush_table():
        nonlocal position, table_counter, current_table_rows, in_table
        if current_table_rows:
            table_content = "\n".join(current_table_rows)
            normalized.append({
                "id": f"tbl_{table_counter}",
                "type": "table",
                "content": clean_text(table_content),
                "raw_rows": current_table_rows,
                "position": position
            })
            position += 1
            table_counter += 1
            current_table_rows = []
            in_table = False

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if raw_line.strip() == "":
            flush_paragraph()
            flush_table()
            i += 1
            continue

        line = raw_line.rstrip()

        # Heading
        is_head, level = is_heading(line)
        if is_head:
            flush_paragraph()
            flush_table()
            heading_content = line.lstrip('#').lstrip()
            normalized.append({
                "id": f"sec_{sec_counter}",
                "type": "heading",
                "level": level,
                "content": clean_text(heading_content),
                "position": position
            })
            position += 1
            sec_counter += 1
            i += 1
            continue

        # Table
        if is_table_row(line) or (in_table and is_table_separator(line)):
            flush_paragraph()
            if not in_table:
                in_table = True
            current_table_rows.append(line)
            i += 1
            if i < len(lines):
                next_line = lines[i].strip()
                if next_line == "" or (not is_table_row(next_line) and not is_table_separator(next_line)):
                    flush_table()
            else:
                flush_table()
            continue

        # Formula (improved)
        if has_formula(line):
            flush_paragraph()
            flush_table()
            normalized.append({
                "id": f"form_{formula_counter}",
                "type": "formula",
                "content": clean_text(line),
                "position": position
            })
            position += 1
            formula_counter += 1
            i += 1
            continue

        # Image
        urls, remainder = extract_images_and_remainder(line)
        if urls:
            flush_paragraph()
            flush_table()
            for url in urls:
                cleaned_url = clean_url(url)
                is_chart = any(keyword in cleaned_url.lower() for keyword in ['chart', 'graph', 'plot', 'figure'])
                normalized.append({
                    "id": f"img_{img_counter}",
                    "type": "image",
                    "subtype": "chart" if is_chart else "general",
                    "content": cleaned_url,
                    "position": position
                })
                position += 1
                img_counter += 1
            if remainder.strip():
                current_paragraph_lines.append(remainder)
            i += 1
            continue

        # Normal text
        if line.lower() == "image:":
            i += 1
            continue

        current_paragraph_lines.append(line)
        i += 1

    flush_paragraph()
    flush_table()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"✅ Normalized JSON saved to: {output_path}")
    print(f"📊 Total atomic units: {len(normalized)}")
    print(f"   - Headings: {sec_counter - 1}")
    print(f"   - Paragraphs: {para_counter - 1}")
    print(f"   - Images: {img_counter - 1}")
    print(f"   - Tables: {table_counter - 1}")
    print(f"   - Formulas: {formula_counter - 1}")

if __name__ == "__main__":
    import os
    from chunkarena.config import EXTRACTED_DATA_OUTPUT_DIR, NORMALIZED_FILE
    normalize_raw_json(
        os.path.join(EXTRACTED_DATA_OUTPUT_DIR, "Banking_system_extraction.json"),
        NORMALIZED_FILE,
    )