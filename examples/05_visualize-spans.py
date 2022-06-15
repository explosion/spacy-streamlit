"""
Example of using `visualize_spans` with a non-default spans_key
"""
import spacy_streamlit
import streamlit as st

import spacy
from spacy_streamlit import visualize_spans

text = "Welcome to the Bank of China."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

doc.spans["sc"] = [
    spacy.tokens.Span(doc, 3, 6, "ORG"),
    spacy.tokens.Span(doc, 5, 6, "GPE"),
]
visualize_spans(
    doc,
)
from pathlib import Path

html = spacy.displacy.render(doc, style="span", page=True)
output_path = Path("sentence.html")
output_path.open("w", encoding="utf-8").write(html)
