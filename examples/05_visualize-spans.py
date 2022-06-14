"""
Example of using `visualize_spans` with a non-default spans_key
"""
import spacy_streamlit
import streamlit as st

import spacy
from spacy_streamlit import visualize_spans

nlp = spacy.load("en_core_web_sm")
doc = nlp("Sundar Pichai is the CEO of Google.")
span = doc[4:7]  # CEO of Google
span.label_ = "CEO"
doc.spans["job_role"] = [span]
visualize_spans(
    doc, spans_key="job_role", displacy_options={"colors": {"CEO": "#09a3d5"}}
)
