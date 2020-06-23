from typing import List, Sequence, Tuple, Optional
import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

from .util import load_model, process_text, get_svg, get_html, get_color_styles, LOGO


# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num"]
# fmt: on

DESCRIPTION = """
Process text with [spaCy](https://spacy.io) models and visualize the output.
Uses spaCy's built-in [displaCy](http://spacy.io/usage/visualizers) visualizer
under the hood.
"""


def visualizer(
    models: List[str],
    default_text: str = "",
    visualizers: List[str] = ["parser", "ner", "textcat", "similarity", "tokens"],
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    similarity_texts: Tuple[str, str] = ("apple", "orange"),
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_model_meta: bool = True,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = DESCRIPTION,
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
) -> None:
    """Embed the full visualizer with selected components."""
    if color:
        st.write(get_color_styles(color), unsafe_allow_html=True)
    if show_logo:
        st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        st.sidebar.title(sidebar_title)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

    spacy_model = st.sidebar.selectbox("Model name", models)
    model_load_state = st.info(f"Loading model '{spacy_model}'...")
    nlp = load_model(spacy_model)
    model_load_state.empty()

    text = st.text_area("Text to analyze", default_text)
    doc = process_text(spacy_model, text)

    if "parser" in visualizers:
        visualize_parser(doc)
    if "ner" in visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, attrs=ner_attrs)
    if "textcat" in visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers:
        visualize_similarity(nlp)
    if "tokens" in visualizers:
        visualize_tokens(doc, attrs=token_attrs)

    if show_json_doc:
        st.header("JSON Doc")
        if st.button("Show JSON Doc"):
            st.json(doc.to_json())

    if show_model_meta:
        st.header("JSON model meta")
        if st.button("Show JSON model meta"):
            st.json(nlp.meta)


def visualize_parser(
    doc: spacy.tokens.Doc,
    *,
    title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    sidebar_title: Optional[str] = "Dependency Parse",
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    split_sents = st.sidebar.checkbox("Split sentences", value=True)
    options = {
        "collapse_punct": st.sidebar.checkbox("Collapse punctuation", value=True),
        "collapse_phrases": st.sidebar.checkbox("Collapse phrases"),
        "compact": st.sidebar.checkbox("Compact mode"),
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options, style="dep")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)


def visualize_ner(
    doc: spacy.tokens.Doc,
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
    sidebar_title: Optional[str] = "Named Entities",
) -> None:
    """Visualizer for named entities."""
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    label_select = st.sidebar.multiselect(
        "Entity labels", options=labels, default=list(labels)
    )
    html = displacy.render(doc, style="ent", options={"ents": label_select})
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
        ]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)


def visualize_textcat(
    doc: spacy.tokens.Doc, *, title: Optional[str] = "Text Classification"
) -> None:
    """Visualizer for text categories."""
    if title:
        st.header(title)
    st.markdown(f"> {doc.text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


def visualize_similarity(
    nlp: spacy.language.Language,
    default_texts: Tuple[str, str] = ("apple", "orange"),
    *,
    threshold: float = 0.5,
    title: Optional[str] = "Vectors & Similarity",
) -> None:
    """Visualizer for semantic similarity using word vectors."""
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if meta.get("width", 0):
        if not meta.get("width", 0):
            st.warning("No vectors available in the model.")
        st.code(meta)
    text1 = st.text_input("Text or word 1", default_texts[0])
    text2 = st.text_input("Text or word 2", default_texts[1])
    doc1 = nlp.make_doc(text1)
    doc2 = nlp.make_doc(text2)
    similarity = doc1.similarity(doc2)
    if similarity > threshold:
        st.success(similarity)
    else:
        st.error(similarity)


def visualize_tokens(
    doc: spacy.tokens.Doc,
    *,
    attrs: List[str] = TOKEN_ATTRS,
    title: Optional[str] = "Token attributes",
) -> None:
    """Visualizer for token attributes."""
    if title:
        st.header(title)
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)
