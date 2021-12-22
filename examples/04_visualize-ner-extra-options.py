"""
Example of using extra_options for visualize_ner.
"""
import spacy

import spacy_streamlit

nlp = spacy.blank("en")
text = "But Google is starting from behind."
doc = nlp.make_doc(text)
ent = doc.char_span(4, 10, label="ORG", kb_id="Q95")
doc.ents = [ent]

spacy_streamlit.visualize_ner(
    doc,
    labels=["ORG"],
    show_table=False,
    title="Custom Colors NER Visualization",
    colors={"ORG": "#EEE"},
    options={
        "kb_url_template": "https://www.wikidata.org/wiki/{}"
    },
    key="Custom Colors"
)

spacy_streamlit.visualize_ner(
    doc,
    labels=["ORG"],
    show_table=False,
    title="Default Colors NER Visualization",
    options={
        "kb_url_template": "https://www.wikidata.org/wiki/{}"
    },
    key="Default Colors"
)
