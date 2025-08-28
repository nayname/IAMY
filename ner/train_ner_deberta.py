"""Fine-tune RoBERTa for Named Entity Recognition using a wide range of CLI examples."""
import os
import json
import re
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model


FILES_DIR = "data/rephrases"
# --- 1. Configuration ---
MODEL_NAME = "microsoft/deberta-v3-large"

ADAPTER_DIR = "artifacts/deberta_lora"          # folder that has adapter_config.json
MERGED_DIR  = "artifacts/deberta_merged_full"   # <- full model will be saved here

# Expanded list of entities for a wider range of commands
LABEL_LIST = [
    "O",
    "B-WALLET", "I-WALLET",
    "B-ADDRESS", "I-ADDRESS",
    "B-AMOUNT", "I-AMOUNT",
    "B-MEMO", "I-MEMO",
    "B-NODE", "I-NODE",
    "B-CHAIN_ID", "I-CHAIN_ID",
    "B-DENOM", "I-DENOM",
    "B-CONTRACT", "I-CONTRACT",
    "B-GAS", "I-GAS",
    "B-FEES", "I-FEES",
    "B-WASM_MSG", "I-WASM_MSG",
    "B-FILE_PATH", "I-FILE_PATH",
    "B-CODE_ID", "I-CODE_ID",
]
id2label = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
label2id = {lbl: i for i, lbl in enumerate(LABEL_LIST)}


# --- 2. Advanced Data Preparation ---

def extract_entities_from_command(command_str):
    """Uses a series of regex patterns to extract entities from various command types."""
    entities = {}

    # General flags
    if node_match := re.search(r"--node\s+([\w\:\/\-\.\$]+)", command_str):
        entities['node'] = node_match.group(1)
    if chain_id_match := re.search(r"--chain-id\s+([\w\-]+)", command_str):
        entities['chain_id'] = chain_id_match.group(1)
    if fees_match := re.search(r"--fees\s+([\w]+)", command_str):
        entities['fees'] = fees_match.group(1)
    if gas_match := re.search(r"--gas\s+([\w\.]+)", command_str):
        entities['gas'] = gas_match.group(1)

    # Command-specific patterns
    if "tx bank send" in command_str:
        match = re.search(r"tx bank send ([\w\$\{\}\(\)]+)\s+([\w\d]+)\s+([\w\/\d]+)", command_str)
        if match:
            entities['wallet'] = match.group(1)
            entities['address'] = match.group(2)
            entities['amount'] = match.group(3)
    elif "query bank balances" in command_str:
        match = re.search(r"query bank balances ([\w\$\(\)\<\>]+)", command_str)
        if match:
            entities['address'] = match.group(1)
    elif "tx wasm execute" in command_str:
        match = re.search(r"tx wasm execute ([\w\$\d]+)\s+'(.*?)'", command_str)
        if match:
            entities['contract'] = match.group(1)
            entities['wasm_msg'] = match.group(2)
            if "migrate" in entities['wasm_msg']:
                code_id_match = re.search(r"new_code_id\":(\d+)", entities['wasm_msg'])
                if code_id_match:
                    entities['code_id'] = code_id_match.group(1)
    elif "tx wasm store" in command_str:
        match = re.search(r"tx wasm store ([\.\/\w\-\$]+)", command_str)
        if match:
            entities['file_path'] = match.group(1)

    return entities


def create_ner_dataset(filepaths):
    """Loads data from all provided CLI files and creates a unified NER dataset."""
    processed_data = []

    with os.scandir(filepaths) as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file (not a directory)
                if 'result' not in entry.name:
                    print(f"File: {entry.name}")
                # print(f"Full path: {entry.path}")
                    with open(entry.path, 'r') as f:
                        content = json.load(f)

                        for item in content:
                            for intent in item['generated_examples']:
                                text = intent['rephrase']
                                valid_ner_tags = intent['bio_tags']

                                # Simple tokenization by space for initial tagging
                                tokens = text.split()
                                if len(tokens) == len(valid_ner_tags):
                                    processed_data.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in valid_ner_tags]})
                                else:
                                    print(f"WARNING: Mismatched tokenization and NER tags for {text}: {valid_ner_tags}")

    with open("processed_data.json", 'w') as f:
        f.write(json.dumps(processed_data))
    return Dataset.from_list(processed_data)


# Create the dataset from all files and split it
raw_ds = create_ner_dataset(FILES_DIR)
raw_ds = raw_ds.train_test_split(test_size=0.2, seed=42)
print(f"Unified NER Dataset created with {len(raw_ds['train'])} training samples.")

# --- 3. Tokenization and Label Alignment ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = raw_ds.map(tokenize_and_align_labels, batched=True)

# --- 4. Model & LoRA Setup ---
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query_proj", "value_proj"],
                     lora_dropout=0.05, bias="none")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
    device_map="auto",
)

model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# --- 5. Training ---
training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=20,  # Adjusted epochs for the larger dataset
    # fp16=True,
    # logging_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,

    gradient_accumulation_steps=2,  # raise if you OOM
    fp16=False,    # ✅
    bf16=False,   # ❌ (T4 has no BF16)
    optim="paged_adamw_8bit",  # works well with bitsandbytes

    logging_strategy="steps",
    logging_steps=10,          # Log every 10 steps instead of 50
    logging_first_step=True,   # Log the first step to see initial loss
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# --- 6. Save Model ---
trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# Merge the LoRA adapter with the base model and save it
merged = trainer.model.merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("\n--- Training Complete ---")
print("Saved LoRA adapter to:", ADAPTER_DIR)
print("Saved merged full model to:", MERGED_DIR)