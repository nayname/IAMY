# calibrate_single.py
import json, numpy as np, torch
from datasets import load_from_disk, load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F

# ----- 1) Labels & mapping
LABEL_LIST = json.load(open("data/labels.json"))      # ["deploy_contract", ...]
label2id = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
id2label = {i: lbl for i, lbl in enumerate(LABEL_LIST)}

# ----- 2) Validation data (RAW, with "text" and a single "label" string)
# If you previously saved: val_ds = load_from_disk("data/val")
# Or re-slice from the same file used in training:
val_ds = load_dataset("json", data_files="data/train.jsonl", split="train[:10%]")

# Ensure each row has exactly ONE label string field named "label".
# If your file still has ["labels"] with a single item, convert once:
def ensure_single_label(ex):
    if "label" in ex:
        return ex
    # if it's ["labels"]: take the first
    ex["label"] = ex["labels"][0]
    return ex
val_ds = val_ds.map(ensure_single_label, load_from_cache_file=False)

# ----- 3) Load model (merged or full fine-tuned checkpoint)
MODEL_DIR = "artifacts/deberta_lora"  # or your fine-tuned directory
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=len(LABEL_LIST),
    problem_type="single_label_classification",
    # torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    # id2label=id2label,
    # label2id=label2id,
)

model = PeftModel.from_pretrained(model, "artifacts/deberta_lora")

# ----- 4) Collect logits and integer targets
logits_list, y_list = [], []
with torch.no_grad():
    for ex in val_ds:
        enc = tok(ex["text"], return_tensors="pt", truncation=True)
        out = model(**enc)
        logits_list.append(out.logits[0].cpu().numpy())      # shape: [K]
        y_list.append(label2id[ex["label"]])

logits = np.asarray(logits_list)   # [N, K]
y_true = np.asarray(y_list)        # [N]

# ----- 5) (Optional) Temperature scaling
# Learn scalar T > 0 to minimize NLL on the val set.
T = torch.nn.Parameter(torch.ones(()))   # start at 1.0
opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

logits_t = torch.from_numpy(logits)      # [N, K]
targets  = torch.from_numpy(y_true).long()

def nll_with_T():
    opt.zero_grad()
    # clamp T to be positive: use softplus
    temp = F.softplus(T) + 1e-6
    loss = F.cross_entropy(logits_t / temp, targets)
    loss.backward()
    return loss

opt.step(nll_with_T)
T_star = float(F.softplus(T).item())

print(f"Learned temperature T = {T_star:.3f}")

# ----- 6) Save calibration (single scalar)
np.save("artifacts/temperature.npy", np.array([T_star], dtype=np.float32))
