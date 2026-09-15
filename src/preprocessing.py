import pandas as pd
import re

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text



def build_model_text(title, text):
    title = normalize_text(title)
    text = normalize_text(text)

    if not text:
        raise ValueError("Article body is required.")

    if title:
        return f"{title}. {text}"

    return text


