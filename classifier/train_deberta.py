"""QLoRA fine‑tune DeBERTa‑v3‑large for multi‑label classification."""
import os, json

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_NAME = "microsoft/deberta-v3-large"
DATA_PATH = "data/train.jsonl"   # each line: {"text": ..., "labels": ["intent_a", ...]}

ADAPTER_DIR = "artifacts/deberta_lora"          # folder that has adapter_config.json
MERGED_DIR  = "artifacts/deberta_merged_full"   # <- full model will be saved here

LABEL_LIST = json.load(open("data/labels.json"))  # ['intent_a', 'intent_b', ...]
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# 1. Dataset ------------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
raw_ds = raw_ds.train_test_split(test_size=0.1, seed=42)
id2label = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
label2id = {lbl: i for i, lbl in enumerate(LABEL_LIST)}

def preprocess(ex):
    ex["text_raw"] = ex["text"]
    enc = tokenizer(ex["text"], truncation=True)
    # if you used "label": ex["label"]; if ["labels"]: ex["labels"][0]
    enc["labels"] = np.int64(label2id[ex["labels"][0]])  # a single int, not a vector
    return enc

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_ds = raw_ds.map(preprocess,
                remove_columns=["text", "labels"],
                load_from_cache_file=False)

# 2. QLoRA setup --------------------------------------------------------------
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"],
                     lora_dropout=0.05, bias="none")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # T4 supports FP16, NOT BF16
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    problem_type="single_label_classification",
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    id2label=id2label,
    label2id=label2id,
    # quantization_config=bnb_cfg,
    device_map="auto",
)
print("num_labels model:", model.config.num_labels)
print("id2label:", model.config.id2label)
print(raw_ds.column_names)   # should include: 'input_ids', 'attention_mask', 'labels'
print(raw_ds["train"][0]["labels"], type(raw_ds["train"][0]["labels"]))  # int for single-label; fixed-size vector for multi-label
print(len(raw_ds["train"][0]["input_ids"]), len(raw_ds["train"][0]["attention_mask"]))  # > 0 and align
print(raw_ds["train"][0]["text_raw"])
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable:,} / total {total:,}")
model = get_peft_model(model, peft_cfg)

# for p in model.parameters():
#     p.requires_grad = False
#
# # Unfreeze LoRA adapters and classifier/pooler
# for name, p in model.named_parameters():
#     if ("lora_" in name) or name.startswith("classifier.") or name.startswith("pooler.dense."):
#         p.requires_grad = True

# 3. Training -----------------------------------------------------------------
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=20,
    # fp16=False, bf16=False,
    eval_strategy ="epoch",
    save_total_limit=2,

    gradient_accumulation_steps=2,  # raise if you OOM
    fp16=False,    # ✅
    bf16=False,   # ❌ (T4 has no BF16)
    optim="paged_adamw_8bit",  # works well with bitsandbytes

    logging_strategy="steps",
    logging_steps=10,          # Log every 10 steps instead of 50
    logging_first_step=True,   # Log the first step to see initial loss
)
#
collator = DataCollatorWithPadding(tokenizer=tokenizer)  # no padding="..." here
#
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=raw_ds["train"],
    eval_dataset=raw_ds["test"],
    data_collator=collator,      # <-- make sure this is passed
)


# show_trainable(model)   # should list lora_* and head params

trainer.train()
trainer.save_model("artifacts/deberta_lora")
tokenizer.save_pretrained("artifacts/deberta_lora")

merged = trainer.model.merge_and_unload()
merged.save_pretrained("artifacts/deberta_merged_full")
tokenizer.save_pretrained("artifacts/deberta_merged_full")

print("Saved merged full model to:", MERGED_DIR)