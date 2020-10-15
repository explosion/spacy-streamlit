<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-streamlit: spaCy building blocks for Streamlit apps

This package contains utilities for visualizing [spaCy](https://spacy.io) models
and building interactive spaCy-powered apps with
[Streamlit](https://streamlit.io). It includes various building blocks you can
use in your own Streamlit app, like visualizers for **syntactic dependencies**,
**named entities**, **text classification**, **semantic similarity** via word
vectors, token attributes, and more.

[![Current Release Version](https://img.shields.io/github/release/explosion/spacy-streamlit.svg?style=flat-square&logo=github&include_prereleases)](https://github.com/explosion/spacy-streamlit/releases)
[![pypi Version](https://img.shields.io/pypi/v/spacy-streamlit.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spacy-streamlit/)

<img width="50%" align="right" src="https://user-images.githubusercontent.com/13643239/85388081-f2da8700-b545-11ea-9bd4-e303d3c5763c.png">

## 🚀 Quickstart

You can install `spacy-streamlit` from pip:

```bash
pip install spacy-streamlit --pre
```

The package includes **building blocks** that call into Streamlit and set up all
the required elements for you. You can either use the individual components
directly and combine them with other elements in your app, or call the
`visualize` function to embed the whole visualizer.

Download the English model from spaCy to get started.

```bash
python -m spacy download en_core_web_sm
```

Then put the following example code in a file.

```python
# streamlit_app.py
import spacy_streamlit

models = ["en_core_web_sm", "en_core_web_md"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualize(models, default_text)
```

You can then run your app with `streamlit run streamlit_app.py`. The app should
pop up in your web browser. 😀

#### 📦 Example: [`01_out-of-the-box.py`](examples/01_out-of-the-box.py)

Use the embedded visualizer with custom settings out-of-the-box.

```bash
streamlit run https://raw.githubusercontent.com/explosion/spacy-streamlit/master/examples/01_out-of-the-box.py
```

#### 👑 Example: [`02_custom.py`](examples/02_custom.py)

Use individual components in your existing app.

```bash
streamlit run https://raw.githubusercontent.com/explosion/spacy-streamlit/master/examples/02_custom.py
```

## 🎛 API

### Visualizer components

These functions can be used in your Streamlit app. They call into `streamlit`
under the hood and set up the required elements.

#### <kbd>function</kbd> `visualize`

Embed the full visualizer with selected components.

```python
import spacy_streamlit

models = ["en_core_web_sm", "/path/to/model"]
default_text = "Sundar Pichai is the CEO of Google."
visualizers = ["ner", "textcat"]
spacy_streamlit.visualize(models, default_text, visualizers)
```

| Argument                 | Type                       | Description                                                                                                                                                                                         |
| ------------------------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models`                 | List[str] / Dict[str, str] | Names of loadable spaCy models (paths or package names). The models become selectable via a dropdown. Can either be a list of names or the names mapped to descriptions to display in the dropdown. |
| `default_text`           | str                        | Default text to analyze on load. Defaults to `""`.                                                                                                                                                  |
| `default_model`          | Optional[str]              | Optional name of default model. If not set, the first model in the list of `models` is used.                                                                                                        |
| `visualizers`            | List[str]                  | Names of visualizers to show. Defaults to `["parser", "ner", "textcat", "similarity", "tokens"]`.                                                                                                   |
| `ner_labels`             | Optional[List[str]]        | NER labels to include. If not set, all labels present in the `"ner"` pipeline component will be used.                                                                                               |
| `ner_attrs`              | List[str]                  | Span attributes shown in table of named entities. See [`visualizer.py`](spacy_streamlit/visualizer.py) for defaults.                                                                                |
| `token_attrs`            | List[str]                  | Token attributes to show in token visualizer. See [`visualizer.py`](spacy_streamlit/visualizer.py) for defaults.                                                                                    |
| `similarity_texts`       | Tuple[str, str]            | The default texts to compare in the similarity visualizer. Defaults to `("apple", "orange")`.                                                                                                       |
| `show_json_doc`          | bool                       | Show button to toggle JSON representation of the `Doc`. Defaults to `True`.                                                                                                                         |
| `show_meta`              | bool                       | Show button to toggle `meta.json` of the current pipeline. Defaults to `True`.                                                                                                                      |
| `show_config`            | bool                       | Show button to toggle `config.cfg` of the current pipeline. Defaults to `True`.                                                                                                                     |
| `show_visualizer_select` | bool                       | Show sidebar dropdown to select visualizers to display (based on enabled visualizers). Defaults to `False`.                                                                                         |
| `sidebar_title`          | Optional[str]              | Title shown in the sidebar. Defaults to `None`.                                                                                                                                                     |
| `sidebar_description`    | Optional[str]              | Description shown in the sidebar. Accepts Markdown-formatted text.                                                                                                                                  |
| `show_logo`              | bool                       | Show the spaCy logo in the sidebar. Defaults to `True`.                                                                                                                                             |
| `color`                  | Optional[str]              | Experimental: Primary color to use for some of the main UI elements (`None` to disable hack). Defaults to `"#09A3D5"`.                                                                              |

#### <kbd>function</kbd> `visualize_parser`

Visualize the dependency parse and part-of-speech tags using spaCy's
[`displacy` visualizer](https://spacy.io/usage/visualizers).

```python
import spacy
from spacy_streamlit import visualize_parser

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a text")
visualize_parser(doc)
```

| Argument        | Type          | Description                                  |
| --------------- | ------------- | -------------------------------------------- |
| `doc`           | `Doc`         | The spaCy `Doc` object to visualize.         |
| _keyword-only_  |               |                                              |
| `title`         | Optional[str] | Title of the visualizer block.               |
| `sidebar_title` | Optional[str] | Title of the config settings in the sidebar. |

#### <kbd>function</kbd> `visualize_ner`

Visualize the named entities in a `Doc` using spaCy's
[`displacy` visualizer](https://spacy.io/usage/visualizers).

```python
import spacy
from spacy_streamlit import visualize_ner

nlp = spacy.load("en_core_web_sm")
doc = nlp("Sundar Pichai is the CEO of Google.")
visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
```

| Argument        | Type          | Description                                                                   |
| --------------- | ------------- | ----------------------------------------------------------------------------- |
| `doc`           | `Doc`         | The spaCy `Doc` object to visualize.                                          |
| _keyword-only_  |               |                                                                               |
| `labels`        | Sequence[str] | The labels to show in the labels dropdown.                                    |
| `attrs`         | List[str]     | The span attributes to show in entity table.                                  |
| `show_table`    | bool          | Whether to show a table of entities and their attributes. Defaults to `True`. |
| `title`         | Optional[str] | Title of the visualizer block.                                                |
| `sidebar_title` | Optional[str] | Title of the config settings in the sidebar.                                  |
| `colors`        | Dict[str,str] | A dictionary mapping labels to display colors ({"LABEL": "COLOR"})            |

#### <kbd>function</kbd> `visualize_textcat`

Visualize text categories predicted by a trained text classifier.

```python
import spacy
from spacy_streamlit import visualize_textcat

nlp = spacy.load("./my_textcat_model")
doc = nlp("This is a text about a topic")
visualize_textcat(doc)
```

| Argument       | Type          | Description                          |
| -------------- | ------------- | ------------------------------------ |
| `doc`          | `Doc`         | The spaCy `Doc` object to visualize. |
| _keyword-only_ |               |                                      |
| `title`        | Optional[str] | Title of the visualizer block.       |

#### `visualize_similarity`

Visualize semantic similarity using the model's word vectors. Will show a
warning if no vectors are present in the model.

```python
import spacy
from spacy_streamlit import visualize_similarity

nlp = spacy.load("en_core_web_lg")
visualize_similarity(nlp, ("pizza", "fries"))
```

| Argument        | Type            | Description                                                                                                                                          |
| --------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nlp`           | `Language`      | The loaded `nlp` object with vectors.                                                                                                                |
| `default_texts` | Tuple[str, str] | The default texts to compare on load. Defaults to `("apple", "orange")`.                                                                             |
| _keyword-only_  |                 |                                                                                                                                                      |
| `threshold`     | float           | Threshold for what's considered "similar". If the similarity score is greater than the threshold, the result is shown as similar. Defaults to `0.5`. |
| `title`         | Optional[str]   | Title of the visualizer block.                                                                                                                       |

#### <kbd>function</kbd> `visualize_tokens`

Visualize the tokens in a `Doc` and their attributes.

```python
import spacy
from spacy_streamlit import visualize_tokens

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a text")
visualize_tokens(doc, attrs=["text", "pos_", "dep_", "ent_type_"])
```

| Argument       | Type          | Description                                                                                              |
| -------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| `doc`          | `Doc`         | The spaCy `Doc` object to visualize.                                                                     |
| _keyword-only_ |               |                                                                                                          |
| `attrs`        | List[str]     | The names of token attributes to use. See [`visualizer.py`](spacy_streamlit/visualizer.py) for defaults. |
| `title`        | Optional[str] | Title of the visualizer block.                                                                           |

### Cached helpers

These helpers attempt to cache loaded models and created `Doc` objects.

#### <kbd>function</kbd> `process_text`

Process a text with a model of a given name and create a `Doc` object. Calls
into the `load_model` helper to load the model.

```python
import streamlit as st
from spacy_streamlit import process_text

spacy_model = st.sidebar.selectbox("Model name", ["en_core_web_sm", "en_core_web_md"])
text = st.text_area("Text to analyze", "This is a text")
doc = process_text(spacy_model, text)
```

| Argument     | Type  | Description                                             |
| ------------ | ----- | ------------------------------------------------------- |
| `model_name` | str   | Loadable spaCy model name. Can be path or package name. |
| `text`       | str   | The text to process.                                    |
| **RETURNS**  | `Doc` | The processed document.                                 |

#### <kbd>function</kbd> `load_model`

Load a spaCy model from a path or installed package and return a loaded `nlp`
object.

```python
import streamlit as st
from spacy_streamlit import load_model

spacy_model = st.sidebar.selectbox("Model name", ["en_core_web_sm", "en_core_web_md"])
nlp = load_model(spacy_model)
```

| Argument    | Type       | Description                                             |
| ----------- | ---------- | ------------------------------------------------------- |
| `name`      | str        | Loadable spaCy model name. Can be path or package name. |
| **RETURNS** | `Language` | The loaded `nlp` object.                                |
