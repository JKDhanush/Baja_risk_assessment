import os
import re
from fpdf import FPDF


# ---------------- CLEAN & NORMALIZE TEXT ---------------- #
def clean(text):
    replacements = {
        "**": "",
        "---": "",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "•": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Fix broken decimals like "59. 7" -> "59.7"
    text = re.sub(r"(\d+)\.\s+(\d+)", r"\1.\2", text)

    cleaned_lines = []
    for line in text.split("\n"):
        # Remove standalone numeric-only lines (e.g., "59.7")
        if re.fullmatch(r"\s*\d+(\.\d+)?\s*", line):
            continue
        # Remove duplicate quantitative headers if LLM outputs them
        if line.strip().lower() == "quantitative risk indicators":
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Fix inline numbered lists: "1. xxx 2. yyy"
    text = re.sub(r"(\d+\.)\s*", r"\n\1 ", text)

    return text.encode("latin-1", "ignore").decode("latin-1")


# ---------------- TABLE HELPERS ---------------- #
def is_table_separator(line):
    return all(c in "-| " for c in line.strip())


def is_real_table(lines, index):
    """
    A real table must:
    - Have at least 2 pipe characters
    - Be followed by another pipe-row or a separator
    """
    if lines[index].count("|") < 2:
        return False

    if index + 1 >= len(lines):
        return False

    next_line = lines[index + 1].strip()
    return "|" in next_line or is_table_separator(next_line)


def parse_table(lines, start):
    rows = []
    i = start

    while i < len(lines):
        line = lines[i].strip()
        if "|" not in line:
            break

        if is_table_separator(line):
            i += 1
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]

        # Merge wrapped description lines
        while i + 1 < len(lines) and "|" not in lines[i + 1]:
            cells[-1] += " " + lines[i + 1].strip()
            i += 1

        rows.append(cells)
        i += 1

    return rows, i


def draw_table(pdf, rows):
    page_width = pdf.w - 2 * pdf.l_margin
    col_count = len(rows[0])
    col_width = page_width / col_count

    # Header
    pdf.set_font("Arial", "B", 11)
    for h in rows[0]:
        pdf.cell(col_width, 8, h, border=1, align="C")
    pdf.ln()

    # Body
    pdf.set_font("Arial", size=11)
    for row in rows[1:]:
        max_lines = 1
        wrapped_cells = []

        for cell in row:
            wrapped = pdf.multi_cell(col_width, 7, cell, split_only=True)
            wrapped_cells.append(wrapped)
            max_lines = max(max_lines, len(wrapped))

        for i in range(max_lines):
            for col in range(col_count):
                txt = wrapped_cells[col][i] if i < len(wrapped_cells[col]) else ""
                pdf.cell(col_width, 7, txt, border=1)
            pdf.ln()


# ---------------- PDF GENERATION ---------------- #
def generate_pdf(report_text, ml_results):
    os.makedirs("reports", exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.cell(
        0, 12,
        "AI-Assisted Risk Assessment & Mitigation Report",
        ln=True, align="C"
    )
    pdf.ln(8)

    pdf.set_font("Arial", size=11)

    lines = clean(report_text).split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            pdf.ln(4)
            i += 1
            continue

        # Section headings (1. Executive Summary etc.)
        if line[0].isdigit() and "." in line[:3]:
            pdf.ln(6)
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 9, line)
            pdf.ln(2)
            pdf.set_font("Arial", size=11)
            i += 1
            continue

        # STRICT table detection
        if is_real_table(lines, i):
            table, new_i = parse_table(lines, i)
            col_count = len(table[0])

            # Validate consistent columns
            if all(len(row) == col_count for row in table):
                pdf.ln(4)
                draw_table(pdf, table)
                pdf.ln(6)
                i = new_i
                continue

        # Bullets / numbered lists
        if line.startswith("-") or re.match(r"\d+\.", line):
            pdf.multi_cell(0, 7, f"  {line}")
            i += 1
            continue

        # Normal paragraph
        pdf.multi_cell(0, 7, line)
        i += 1

    # ---------------- FINAL QUANTITATIVE SECTION ---------------- #
    pdf.ln(8)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 9, "Quantitative Risk Indicators", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(
        0, 7,
        f"Overall Risk Classification : {ml_results.get('risk_classifier', 'N/A')}"
    )
    pdf.multi_cell(
        0, 7,
        f"Estimated Schedule Delay     : {ml_results.get('delay_predictor', 'N/A')} days"
    )
    pdf.multi_cell(
        0, 7,
        f"Estimated Cost Overrun       : {ml_results.get('cost_overrun_predictor', 'N/A')} %"
    )

    # Footer
    pdf.ln(12)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 6,
        "Prepared for strategic decision-making using publicly available information."
    )

    pdf.output("reports/AI_Risk_Assessment_Report.pdf")
