"""QLoRA fine‑tune DeBERTa‑v3‑large for multi‑label classification."""
import os, json

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_NAME = "microsoft/deberta-v3-large"
DATA_PATH = "data/train.jsonl"   # each line: {"text": ..., "labels": ["intent_a", ...]}
LABEL_LIST = json.load(open("data/labels.json"))  # ['intent_a', 'intent_b', ...]

# 1. Dataset ------------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
raw_ds = raw_ds.train_test_split(test_size=0.1, seed=42)
mlb = MultiLabelBinarizer(classes=LABEL_LIST)

def preprocess(ex):
    enc = tokenizer(ex["text"], truncation=True)
    y = mlb.fit_transform([ex["labels"]])[0].astype(np.float32)  # shape: (len(LABEL_LIST),)
    enc["labels"] = y
    return enc

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_ds = raw_ds.map(preprocess,
                remove_columns=["text", "labels"],
                load_from_cache_file=False)

# 2. QLoRA setup --------------------------------------------------------------
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"],
                     lora_dropout=0.05, bias="none")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    problem_type="multi_label_classification",
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
)
model = get_peft_model(model, peft_cfg)

# 3. Training -----------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=4,
    fp16=False, bf16=False,
    eval_strategy ="epoch",
    # logging_steps=50,
    save_total_limit=2,

    logging_strategy="steps",
    logging_steps=10,          # Log every 10 steps instead of 50
    logging_first_step=True,   # Log the first step to see initial loss
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)  # no padding="..." here

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=raw_ds["train"],
    eval_dataset=raw_ds["test"],
    data_collator=collator,      # <-- make sure this is passed
)

trainer.train()
trainer.save_model("artifacts/deberta_lora")