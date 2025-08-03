import json

import joblib, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from datasets import load_from_disk, load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_LIST = json.load(open("data/labels.json"))  # ['intent_a', 'intent_b', ...]
DATA_PATH = "data/train.jsonl"   # each line: {"text": ..., "labels": ["intent_a", ...]}

# 1. Dataset ------------------------------------------------------------------
mlb = MultiLabelBinarizer(classes=LABEL_LIST)

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model = AutoModelForSequenceClassification.from_pretrained("artifacts/deberta_lora")
LABEL_LIST = json.load(open("data/labels.json"))

raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
# raw_ds = raw_ds.train_test_split(test_size=0.1, seed=42)

logits, y_true = [], []
for ex in raw_ds:
    out = model(**tok(ex["text"], return_tensors="pt", truncation=True))
    logits.append(out.logits.detach().cpu().numpy()[0])
    y_true.append(mlb.transform([ex["labels"]])[0])
logits, y_true = np.array(logits), np.array(y_true)

thresh = {}
for i, lbl in enumerate(LABEL_LIST):
    iso = IsotonicRegression(out_of_bounds="clip").fit(logits[:, i], y_true[:, i])
    prob = iso.predict([0])[0]
    thresh[lbl] = prob
joblib.dump(thresh, "artifacts/thresh.pkl")