import json

import joblib, numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABEL_LIST = json.load(open("data/labels.json"))
THRESH = joblib.load("artifacts/thresh.pkl")

sess = ort.InferenceSession("artifacts/deberta_intent.onnx", providers=["CPUExecutionProvider"])
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

def classify(prompt: str, k=5):
    inputs = tok(prompt, return_tensors="np", truncation=True)
    ort_inputs = {k: v for k, v in inputs.items() if k in sess.get_inputs()[0].name}
    logits = sess.run(None, ort_inputs)[0][0]
    probs = 1 / (1 + np.exp(-logits))
    return [lbl for lbl, p in zip(LABEL_LIST, probs) if p >= THRESH[lbl]]

if __name__ == "__main__":
    print(classify("Deploy an NFT marketplace contract"))