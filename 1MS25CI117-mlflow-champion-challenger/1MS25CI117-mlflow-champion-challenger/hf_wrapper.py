"""Sentiment helper + custom MLflow pyfunc wrapper around a Hugging Face text-classification model (Lab 9 style).
Both models output NEGATIVE / POSITIVE, which are mapped to the SST-2 ids 0 / 1 here.
Student: Shashikala | USN: 1MS25CI117"""
import mlflow.pyfunc
import pandas as pd

LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1, "LABEL_0": 0, "LABEL_1": 1}


def to_label(raw):
    """Map whatever the model calls its classes to 'negative' / 'positive'."""
    return "positive" if LABEL2ID[raw.upper()] == 1 else "negative"


def classify(tok, mdl, text, max_length=128):
    """Return (label, confidence) for one sentence."""
    import torch
    enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        probs = mdl(**enc).logits.softmax(-1)[0]
    i = int(probs.argmax())
    return to_label(mdl.config.id2label[i]), round(float(probs[i]), 4)


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a `text` column. Output: DataFrame with text, predicted label and confidence."""

    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        cfg = context.model_config or {}
        self.max_length = int(cfg.get("max_length", 128))
        path = context.artifacts["hf_model_dir"]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        res = [classify(self.tok, self.mdl, t, self.max_length) for t in texts]
        return pd.DataFrame({"text": texts, "prediction": [l for l, _ in res], "confidence": [c for _, c in res]})
