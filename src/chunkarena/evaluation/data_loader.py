"""Golden dataset loader.

Reads the golden dataset file (CSV or Excel) at GOLDEN_DATASET_PATH and
returns a list of question dictionaries with a stable integer id, the
question text, the gold spans split on semicolons, and a gold answer
text used by the RAG-quality proxies.
"""

import pandas as pd
from pathlib import Path

from chunkarena.config import GOLDEN_DATASET_PATH


def _read_golden_file(path: str) -> pd.DataFrame:
    """Read a golden dataset file, auto-detecting CSV or Excel format.

    Args:
        path: File path ending in .csv, .xlsx, or .xls.

    Returns:
        A pandas DataFrame with the golden dataset rows.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(
            f"Unsupported golden dataset format '{ext}'. Use .csv or .xlsx"
        )


def load_questions():
    """Load the golden dataset into the runner's in-memory format.

    Reads :data:`config.GOLDEN_DATASET_PATH` (CSV or Excel), splits each
    row's ``Facts`` / ``Gold_spans`` field on semicolons to recover the
    list of gold relevance strings, and takes ``Golden Answer`` /
    ``Answer`` / ``Gold_text`` as the gold answer when present; otherwise
    joins the gold spans to synthesise one.

    Returns:
        List of dicts with keys ``id`` (int), ``question`` (str),
        ``gold_spans`` (list[str]) and ``gold_answer`` (str).
    """
    print(f"Loading gold dataset from {GOLDEN_DATASET_PATH}...")
    gold_df = _read_golden_file(GOLDEN_DATASET_PATH)

    # Support multiple answer column names
    answer_col = None
    for col in ("Golden Answer", "Answer", "Gold_text"):
        if col in gold_df.columns:
            answer_col = col
            break

    # Support multiple spans column names
    spans_col = None
    for col in ("Facts", "Gold_spans"):
        if col in gold_df.columns:
            spans_col = col
            break
    if spans_col is None:
        raise KeyError(
            f"Golden dataset must have a 'Facts' or 'Gold_spans' column. "
            f"Found columns: {list(gold_df.columns)}"
        )

    questions_data = []
    for _, row in gold_df.iterrows():
        gold_spans = [s.strip() for s in str(row[spans_col]).split(";") if s.strip()]
        if answer_col and pd.notna(row[answer_col]):
            gold_answer = str(row[answer_col]).strip()
        else:
            gold_answer = " ".join(gold_spans)
        questions_data.append({
            "id"         : len(questions_data),
            "question"   : row["Question"],
            "gold_spans" : gold_spans,
            "gold_answer": gold_answer,
        })
    print(f"  Loaded {len(questions_data)} questions")
    return questions_data
