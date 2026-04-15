"""Golden dataset loader.

Reads the CSV at CSV_PATH and returns a list of question dictionaries
with a stable integer id, the question text, the gold spans split on
semicolons, and a gold answer text used by the RAG-quality proxies.
If the CSV has an Answer column it is used as-is; otherwise the joined
gold spans serve as the answer.
"""

import pandas as pd

from config import CSV_PATH


def load_questions():
    print("Loading gold dataset...")
    gold_df = pd.read_csv(CSV_PATH)
    has_answer = "Answer" in gold_df.columns
    questions_data = []
    for _, row in gold_df.iterrows():
        gold_spans = [s.strip() for s in str(row["Gold_spans"]).split(";") if s.strip()]
        if has_answer and pd.notna(row["Answer"]):
            gold_answer = str(row["Answer"]).strip()
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
