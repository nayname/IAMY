import json

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

# Use the directory where your merged RoBERTa NER model was saved
MERGED_DIR = "artifacts/artifacts_ner/deberta_merged_full"


class NERExtractor:
    """
    A class to load a trained NER model and extract entities from text.
    """

    def __init__(self):
        print(f"Loading model from: {MERGED_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
        self.model = AutoModelForTokenClassification.from_pretrained(MERGED_DIR)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device.upper()}.")

        # Load label mapping from the model's configuration
        self.id2label = self.model.config.id2label

    def extract_entities(self, text: str):
        """
        Takes a natural language string and returns a dictionary of extracted entities.
        """
        with torch.no_grad():
            # 1. Tokenization
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 2. Model Inference
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)

            # 3. Post-processing to group entities
            entities = []
            current_entity = None

            # Get tokens and predicted label IDs
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_label_ids = predictions[0].tolist()

            for token, label_id in zip(tokens, predicted_label_ids):
                label_name = self.id2label[label_id]

                # Ignore special tokens and padding
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue

                if label_name.startswith("B-"):
                    # If we are in the middle of another entity, save it first
                    if current_entity:
                        entities.append(current_entity)
                    # Start a new entity
                    current_entity = {"type": label_name[2:], "tokens": [token]}
                elif label_name.startswith("I-") and current_entity:
                    # Continue the current entity only if the type matches
                    if label_name[2:] == current_entity["type"]:
                        current_entity["tokens"].append(token)
                    else:  # If type doesn't match, save the old one and start a new one
                        entities.append(current_entity)
                        current_entity = {"type": label_name[2:], "tokens": [token]}

                else:  # O-tag or misprediction
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = None

            # Add the last entity if it exists
            if current_entity:
                entities.append(current_entity)

            # 4. Clean up and group tokens
            grouped_entities = {}
            for entity in entities:
                # Use the tokenizer's decoder to correctly join word pieces
                full_entity_text = self.tokenizer.convert_tokens_to_string(entity["tokens"])
                entity_type = entity["type"]

                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []

                grouped_entities[entity_type].append(full_entity_text.strip())

        return grouped_entities


# --- Example Usage ---
if __name__ == "__main__":
    ner_extractor = NERExtractor()

    # Example sentences from your training data
    test_sentences = [
        "Send 10 NTRN from my default wallet to Bob's address ntrn1bobaddressxx",
        "Upload my compiled smart contract to Neutron mainnet from mykey",
        "Check my CW20 token balance for contract <cw20_contract_address>",
        "Upgrade the Main DAO core contract to the latest code ID 325",
        "Show my wallet balance on Neutron mainnet using node https://grpc-kaiyo-1.neutron.org:443"
    ]

    for sentence in test_sentences:
        result = ner_extractor.extract_entities(sentence)
        print(json.dumps(result, indent=2))
        print("-" * 30)
#
# # check()
# print(classify("Calculate the md5 sum of the list of files in the current directory"))
# print(classify("(GNU specific) Display cumulative CPU usage over 5 seconds."))
# print(classify("Display cumulative CPU usage over 5 seconds."))
# print(classify("Show aggregate CPU utilization over a five-second interval."))
# print(classify("Report cumulative CPU usage for the next 5 seconds."))
# print(classify("Measure total CPU usage across a 5-second window."))
# print(classify("Display overall CPU utilization over a 5-second period."))
# print(classify("Summarize CPU usage collected over five seconds"))
# print(classify("test"))
# print(classify("instantiate contract"))
# print(classify("hallo kak dela bye"))
# print(classify("test"))
