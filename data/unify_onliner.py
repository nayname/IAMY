import json

CLI_DATA_PATHS = {
    "Query All User Balances": "../data/intents_oneliner/Query All User Balances",
    "Execute Contract": "../data/intents_oneliner/Execute Contract",
    "Upload Contract": "../data/intents_oneliner/Upload Contract",
    "Migrate Contract": "../data/intents_oneliner/Migrate Contract",
    "Query Specific Balance": "../data/intents_oneliner/Query Specific Balance",
    "Query Contract State": "../data/intents_oneliner/Query Contract State",
    "Send Tokens": "../data/intents_oneliner/Send Tokens",
}

def unify(filepaths):
    """Loads data from all provided CLI files and creates a unified NER dataset."""
    processed_data = []
    for key, filepath in filepaths.items():
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
                cli_data = content[key]
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not read or parse {filepath}. Skipping. Error: {e}")
            continue

        for intent, details in cli_data.items():
            text = intent
            command = details.get("command", "")
            entities = extract_entities_from_command(command)

            # Simple tokenization by space for initial tagging
            tokens = text.split()
            ner_tags = ['O'] * len(tokens)

            # A more robust tagging logic would involve character-level indexing
            for i, token in enumerate(tokens):
                clean_token = token.strip('.,!;"\'')
                for entity_type, entity_value in entities.items():
                    if clean_token in entity_value:
                        # Basic B-tagging; a more complex system would handle I-tags for multi-word entities
                        ner_tags[i] = f'B-{entity_type.upper()}'

            # Ensure all generated tags are valid
            valid_ner_tags = [tag if tag in label2id else 'O' for tag in ner_tags]
            processed_data.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in valid_ner_tags]})
            print(processed_data)

    return Dataset.from_list(processed_data)

unify(CLI_DATA_PATHS)