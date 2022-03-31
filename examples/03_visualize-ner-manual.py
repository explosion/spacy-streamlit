"""
Example of using manual=True for visualize_ner.
"""
import spacy_streamlit
import streamlit as st

st.title("My cool app")

doc = [{
    "text": "But Google is starting from behind.",
    "ents": [{"start": 4, "end": 10, "label": "ORG"}],
    "title": None
}]

spacy_streamlit.visualize_ner(
    doc,
    labels=["ORG"],
    show_table=False,
    title="Manual visualization of organisations",
    manual=True
)
