"""
Example of using extra_options for visualize_ner.
"""
import spacy
import streamlit as st

import spacy_streamlit

st.title("My cool app")

nlp = spacy.blank("en")
text = "But Google is starting from behind."
doc = nlp.make_doc(text)
ent = doc.char_span(4, 10, label="ORG", kb_id="Q95")
doc.ents = [ent]

spacy_streamlit.visualize_ner(
    doc,
    labels=["ORG"],
    show_table=False,
    title="Persons, dates and locations",
    extra_options={"kb_url_template": "https://www.wikidata.org/wiki/{}"}
)
