from typing import List, Sequence, Tuple, Optional, Dict
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
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""


def visualize(
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
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
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

    spacy_model = st.sidebar.selectbox("Model name", models, key=f"{key}_visualize_models")
    model_load_state = st.info(f"Loading model '{spacy_model}'...")
    nlp = load_model(spacy_model)
    model_load_state.empty()

    text = st.text_area("Text to analyze", default_text, key=f"{key}_visualize_text")
    doc = process_text(spacy_model, text)

    if "parser" in visualizers:
        visualize_parser(doc, key=key)
    if "ner" in visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(doc, labels=ner_labels, attrs=ner_attrs, key=key)
    if "textcat" in visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers:
        visualize_similarity(nlp, key=key)
    if "tokens" in visualizers:
        visualize_tokens(doc, attrs=token_attrs)

    if show_json_doc:
        st.header("JSON Doc")
        if st.button("Show JSON Doc", key=f"{key}_visualize_show_json_doc"):
            st.json(doc.to_json())

    if show_model_meta:
        st.header("JSON model meta")
        if st.button("Show JSON model meta", key=f"{key}_visualize_show_model_meta"):
            st.json(nlp.meta)

    st.sidebar.markdown(
        FOOTER, unsafe_allow_html=True,
    )


def visualize_parser(
    doc: spacy.tokens.Doc,
    *,
    title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    sidebar_title: Optional[str] = "Dependency Parse",
    key: Optional[str] = None,
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    split_sents = st.sidebar.checkbox("Split sentences", value=True, key=f"{key}_parser_split_sents")
    options = {
        "collapse_punct": st.sidebar.checkbox("Collapse punctuation", value=True, key=f"{key}_parser_collapse_punct"),
        "collapse_phrases": st.sidebar.checkbox("Collapse phrases", key=f"{key}_parser_collapse_phrases"),
        "compact": st.sidebar.checkbox("Compact mode", key=f"{key}_parser_compact"),
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
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
) -> None:
    """Visualizer for named entities."""
    if title:
        st.header(title)
    if sidebar_title:
        st.sidebar.header(sidebar_title)
    label_select = st.sidebar.multiselect(
        "Entity labels", options=labels, default=list(labels), key=f"{key}_ner_label_select"
    )
    html = displacy.render(
        doc, style="ent", options={"ents": label_select, "colors": colors}
    )
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
    key: Optional[str] = None,
) -> None:
    """Visualizer for semantic similarity using word vectors."""
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if not meta.get("width", 0):
        st.warning("No vectors available in the model.")
    st.code(meta)
    text1 = st.text_input("Text or word 1", default_texts[0], key=f"{key}_similarity_text1")
    text2 = st.text_input("Text or word 2", default_texts[1], key=f"{key}_similarity_text2")
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
