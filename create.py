import json
import random
import sys

from classifier.inference import Classifier
from lib.pp import Pipeline
from ner.inference import NERExtractor

#unfine-tuned prompt intended for script generation
generate_flex = """
    You are an expert-level command-line assistant specializing in the `neutrond` CLI for the Neutron blockchain. Your sole purpose is to take structured input and construct a single, valid, executable `neutrond` command.
    
    **TASK:**
    Given the user's classified intent (Label), the entities extracted from their request (NER), and a base command template, construct the final, executable CLI command.
    
    **CONTEXT:**
    
      * **NER Output:** A JSON object containing the entities extracted from the user's natural language request.
      * **Label:** The specific intent that was classified (e.g., "Send Tokens", "Execute Contract").
      * **Base Command:** A template representing the structure of the command for the given label. It contains placeholders like `<placeholder>`.
    
    **INSTRUCTIONS:**
    
    1.  **Fill the Template:** Replace the placeholders in the `Base Command` with the corresponding values from the `NER Output`. Your output command should sctrictly follow the base command.
    2.  **Use Sensible Defaults:** If a common, non-critical parameter (like `--node`, `--chain-id`, or `--gas`) is missing from the NER output, use a standard default value (e.g., `https://rpc.neutron.org:443` for the node, `neutron-1` for the chain ID, `auto` for gas). Always include the `-y` flag for transaction commands unless otherwise specified.
    3.  **Handle Missing Critical Information:** If a critical parameter (like `<wallet>`, `<address>`, or `<amount>`) is missing from the NER output, use a clear placeholder in the final command (e.g., `<YOUR_WALLET_NAME>`, `<RECIPIENT_ADDRESS>`).
    4.  **Output ONLY the Command:** Do not add any explanations, comments, or conversational text. Your entire output must be a single, executable command string.
    
    -----
    
    ### EXAMPLES
      * **NER Output:**
        {{
          "WALLET": "mywallet",
          "AMOUNT": "10000000untrn",
          "ADDR": "ntrn1bobaddressxxxxxxxxxx"
        }}

      * **Label:** `Send Tokens`
      * **Base Command:** `neutrond tx bank send <from> <to> <amount> --node <grpc_node> <signing info> <gas>`
      * **Output:**
        neutrond tx bank send mywallet ntrn1bobaddressxxxxxxxxxx 10000000untrn --node https://rpc.neutron.org:443 --chain-id neutron-1 --gas auto -y
    
    
    ### YOUR TASK
    
      * **NER Output:**
        {{*#*NER_OUTPUT*#*}}
      * **Label:** `{{*#*LABEL*#*}}`
      * **Base Command:** `{{*#*BASE_COMMAND*#*}}`
      * **Output:**
"""


def get_command(label):
    commands = {"query_balance": "neutrond query bank balances/balances <address> --node <grpc_node>",
	"query_contract":"neutrond query wasm contract-state smart <contract-address> <query-msg> --node <grpc_node>",
	"send":"neutrond tx bank send <from> <to> <amount> --node <grpc_node> <signing info> <gas>",
	"upload":"neutrond tx wasm store <wasm-file> --node <grpc_node> <signing info> <gas>",
	"instantiate":"neutrond tx wasm instantiate <code_id> <init-msg> --node <grpc_node> <signing info> <gas>",
	"execute":"neutrond tx wasm execute <contract-address> <exec-msg> --node <grpc_node> <signing info> <gas>"}

    return commands[label]


def get_context(query, label, ner_, base):
    """
    strings to context for prompt
    :param query:
    :param context:
    :return:
    """
    messages = [
        {
            "role": "system",
            "content": generate_flex.replace("*#*NER_OUTPUT*#*", json.dumps(ner_)).replace("*#*LABEL*#*", label)
                .replace("*#*BASE_COMMAND*#*", base)
        },
        {
            "role": "user",
            "content": query
        }
    ]

    print(messages)
    return messages


def validate(class_, answer):
    """
    validate output, call to Validator class
    :return:
    """

    hash = random.getrandbits(128)
    f = open("generated/" + str(hash) + "_" + class_, "a")
    f.write(answer.choices[0].message.content)
    f.close()


def glue(ner_):
    res = {}
    for k, v in ner_.items():
        res[k] = ""
        for i in v:
            res[k] += i + " "
    return res


def generate_code(query):
    class_ = Classifier().classify(query)
    ner_ = NERExtractor().extract_entities(query)
    base_command = get_command(class_['label'])
    #APP config

    if class_['label'] != "others":
        p = Pipeline()
        messages = get_context(query, class_['label'], ner_, base_command)
        request, answer = p.ask_gpt(messages, "gpt-4o")

        return {
            "label": class_['label'].title().replace("_", " "),
            "params": glue(ner_),
            "base_command": base_command.replace('"', ''),
            "command": answer.choices[0].message.content.replace("```", "")
        }
    else:
        return {"label": "Others", "params": "UNDEF", "command": "NO COMMAND"}

