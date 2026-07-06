# The Desk — AG News Classifier (Streamlit Deployment)

A small newsroom-styled web app that deploys the Word2Vec + Optuna-tuned
neural network trained in `AG_News_Word2Vec_Classification.ipynb`. Paste
in any news text and it files it under World, Sports, Business, or Sci/Tech.

## 1. Folder setup

Put these five files in the **same folder**:

```
app.py
requirements.txt
word2vec_agnews.model
scaler.pkl
label_encoder.pkl
agnews_classification_model.keras
```

The first two are in this download; the other four were produced by the
notebook (they're already sitting in your outputs from that step).

> Note: `word2vec_agnews.model` is a gensim save file — if a
> `word2vec_agnews.model.wv.vectors.npy` (or similar) sidecar file was
> generated alongside it, keep that in the same folder too. Gensim
> writes large models across multiple files.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run it

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (typically `http://localhost:8501`).

## What it does

1. Cleans and tokenizes your input the same way the training notebook did.
2. Averages the word vectors from the self-trained Word2Vec model into a
   single 200-dim sentence vector.
3. Scales that vector with the saved `StandardScaler`.
4. Runs it through the trained Keras classifier to get a probability for
   each of the 4 categories.
5. Shows the top call, a confidence breakdown for all four categories, and
   how many of your words were actually recognized by the model's vocabulary.

## Notes

- If you paste text with mostly unrecognized words (e.g. non-English text,
  or very unusual slang), the app will tell you rather than guessing on a
  near-empty vector.
- This is a demo/deployment wrapper for a coursework model — it's only as
  good as the 120K-article AG News training set it learned from, so don't
  expect it to handle every real-world headline perfectly.
