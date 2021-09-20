import spacy_streamlit

models = ["en_core_web_md","en_core_web_sm"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualize(models, default_text, show_visualizer_select=True)