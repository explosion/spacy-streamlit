"""
Very basic out-of-the-box example using the full visualizer. This file can be
run using the "streamlit run" command.

Prerequisites:
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
"""
import spacy_streamlit

models = ["en_core_web_sm", "en_core_web_md"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualizer(models, default_text)
