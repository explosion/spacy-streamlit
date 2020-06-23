"""
Example using the components provided by spacy-streamlit in an existing app.

Prerequisites:
python -m spacy download en_core_web_sm
"""
import spacy_streamlit
import streamlit as st

DEFAULT_TEXT = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."""

spacy_model = "en_core_web_sm"

st.title("My cool app")
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)
doc = spacy_streamlit.process_text(spacy_model, text)

spacy_streamlit.visualize_ner(
    doc,
    labels=["PERSON", "DATE", "GPE"],
    show_table=False,
    title="Persons, dates and locations",
    sidebar_title=None,
)
st.text(f"Analyzed using spaCy model {spacy_model}")
