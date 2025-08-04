import json

import joblib, numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_LIST = json.load(open("data/labels.json"))
# id2label = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


# sess = ort.InferenceSession("artifacts/deberta_intent.onnx", providers=["CPUExecutionProvider"])
# tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# If you merged LoRA → use the merged dir; else load your fine-tuned dir
MERGED_DIR = "artifacts/deberta_lora"  # the folder you saved after merge

tok = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MERGED_DIR).eval()

# (optional) GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = [model.config.id2label[i] for i in range(model.config.num_labels)]
print("model id2label:", id2label)
print("inference LABEL_LIST:", LABEL_LIST)
# assert id2label == LABEL_LIST, "Mismatch: reorder LABEL_LIST to match model.config.id2label"
# load temperature (default to 1.0 if not present)
try:
    T = float(np.load("artifacts/temperature.npy")[0])
except Exception:
    T = 1.0

def classify(text: str):
    with torch.inference_mode():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        logits = model(**enc).logits[0].cpu().numpy()
    # print("logits:", logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    # print("probs:", probs, "argmax:", probs.argmax(), "label:", id2label[probs.argmax()])
    logits = logits / T
    # softmax
    pred_id = int(probs.argmax())
    return {
        "text": text,
        "label": id2label[pred_id],
        "prob": float(probs[pred_id]),
        "uncertainty": 1.0 - float(probs[pred_id])   # or entropy(probs)
    }

def check():
    LABEL_LIST = ['awk', 'find', 'others', 'sort', 'xargs']
    id2label = {i: l for i, l in enumerate(LABEL_LIST)}
    label2id = {l: i for i, l in enumerate(LABEL_LIST)}

    SRC = "artifacts/deberta_lora"  # your trained folder (or merged LoRA)
    DST = "artifacts/deberta_ft_named"  # new folder with names

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large" )

    model = PeftModel.from_pretrained(model, MERGED_DIR)
    print("num_labels:", model.config.num_labels)  # likely 2
    print(model.classifier)  # shows out_features=2
    print(model.state_dict()["classifier.weight"].shape)  # (2, hidden_size)
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.save_pretrained(DST)

    tok = AutoTokenizer.from_pretrained(SRC)
    tok.save_pretrained(DST)

    model = AutoModelForSequenceClassification.from_pretrained(DST).eval()
    print("-----------")
    print([model.config.id2label[i] for i in range(model.config.num_labels)])

# check()
print(classify("Calculate the md5 sum of the list of files in the current directory"))
print(classify("(GNU specific) Display cumulative CPU usage over 5 seconds."))
print(classify("Display cumulative CPU usage over 5 seconds."))
print(classify("Show aggregate CPU utilization over a five-second interval."))
print(classify("Report cumulative CPU usage for the next 5 seconds."))
print(classify("Measure total CPU usage across a 5-second window."))
print(classify("Display overall CPU utilization over a 5-second period."))
print(classify("Summarize CPU usage collected over five seconds"))
print(classify("test"))
print(classify("instantiate contract"))
print(classify("hallo kak dela bye"))
print(classify("test"))
