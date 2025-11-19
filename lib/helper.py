import json
import os

"""
I will provide a list of functions. Please process them as follows:

 1.  **Literal Porting:** Identify any non-Python code and port it to Python. This must be a **direct translation** of the syntax only; do not 'improve' or 'pythonize' the logic.
 2.  **Deduplication:** Compare all functions to find exact duplicates. A duplicate is defined as having **identical inputs and identical logic**.
 3.  **Reporting:** List clearly which functions are being removed as duplicates.
 4.  **Assembly:** Generate the final Python file with all imports moved to the top.

 **STRICT CONSTRAINTS:**
 * **NO NEW FUNCTIONS:** Do not invent, add, or hallucinate any helper functions, wrappers, or classes that are not explicitly present in the input text.
 * **NO REFACTORING:** Do not optimize, clean up, or modernize the code inside the functions. Preserve the original implementation style and variable names exactly as they are.
 * **VERBATIM PRESERVATION:** If a function is already in Python, do not change a single character of its body (other than indentation adjustments if necessary).
 * **Output:** Provide only the final valid Python code.
"""

"""
Role: You are an expert Python Backend Developer specializing in blockchain interactions and automation scripts.

Context: I have a Python function named tamples_recipes that handles various "intents" (user actions) such as compiling contracts, sending tokens, or querying balances. Currently, the code expects parameters to be passed in a req dictionary, but it lacks a robust way to handle cases where the user wants to run a "demo" version without providing any inputs.

The Challenge:
I need to support two execution modes for a Chat UI:

Default / One-Click Mode: The user clicks a button (e.g., "Check Balance") without providing details. The script must detect this and use pre-configured "Safe Defaults" (e.g., a demo wallet address, a specific testnet contract) to execute a successful demonstration immediately.

Custom / Full Mode: The user provides specific details (e.g., a specific wallet address). The script must use these details, overriding any defaults.

Your Task:
Refactor the provided code to implement this "Dual Mode" logic.

Specific Requirements:

Configuration Map: Create a dictionary named DEFAULT_SCENARIOS at the top of the file. Keys should be the exact input_text strings from the case statements. Values should be a dictionary of valid default parameters (e.g., {'address': 'juno1...', 'chain_id': 'uni-6'}) necessary to run that specific intent successfully.

Merge Logic: At the start of the function, implement logic to merge the DEFAULT_SCENARIOS for the given intent with the user's req data. User data must always take precedence over defaults.

Mocking Dependencies (CRITICAL): The original code calls many external functions (e.g., verify_docker_installed, construct_msg_send, broadcast_tx). You must generate mock/stub implementations for ALL of these missing functions. The mocks should return realistic dummy data (like {"txhash": "A1B2..."}) so the code is fully runnable and testable immediately without a blockchain connection.

Error Handling: Ensure the code doesn't crash if req is None.

Input Code:
[Paste the content of function.txt here]
"""

"""
"Given the user-provided function library `Backend.py` and the intent descriptions in `intents.txt`, write me a single Python function `async def intents(input_text, req):` that implements *all* the intents using a `match input_text:` statement. The function body for each `case` should execute the sequence of functions specified in the JSON, assigning intermediate results to unique variables (e.g., `output1`, `output2`), and appending descriptive strings to a list named `res` after each function call.

The structure for each case *must* follow this exact pattern:

```python
case <intent_string>:
    res = []
    <output1> = <lib func1>(<args>)
    res.append(<text describing what was done mentioning received output1>)
    <output2> = <lib func2(output1)>(<other args>)
    res.append(<text describing what was done mentioning received output2>)
    # ... continue for all functions in the intent ...
    return {"status": "success", "message": "...", "data": ...}
```

The function should handle all 17 intents defined in `intents.txt`. Since many functions require parameters like `address`, `project_root`, or `txhash`, assume these are available within the input `req: Dict[str, Any]` and extract them at the start of the `intents` function.

Also, compile me a new, minimal library file named `Backend.py` that contains *only* the Python code for the functions actually called within the `intents(input_text, req)` implementation."
"""


"""
Given the lift of functions - remove duplicates (the ones that duplicate the functionality with the same input arguments). All functions should be in JavaScript (port from othe lanhgs if needed). Do not consolidate any functions, just remove duplicates. Provide the final file. 
"""

"""
Two files backend.py and backend_btc.py. Do not consolidate any functions, just remove duplicates. Provide the final file, which contains functions from backend_btc.py that have no duplicates in backend.py
"""

"""
Given the list of functions:
1) remove duplicates - find the ones that duplicate the functionality with the same input arguments and leave only one of them, removing redundancy
2) find cases where external libs were used (functions will not work in mintlify)
2.1) in fucntions where those libs are unnecessary - refactor functions to implement the same functionality without external libs
2.2) in cases wree it's not possible, try to propose a version of the same functionality on on a backend
2.3) if this also impossible, write a functionn with a body "alert('Function is not implemented')"
Do not consolidate any functions, just remove duplicates.
Provide an answer in a file
"""

pages = {
    # "NeutronTemplate":"data/pages/NeutronTemplate",
    # "Cron": "data/pages/Cron",
    #***
    # COSMOS
    #***
    "UserGuides": "data/pages/UserGuides",
    "ToolingResources": "data/pages/ToolingResources",
    "EthereumJSON_RPC": "data/pages/EthereumJSON_RPC",
    "Learn": "data/pages/Learn",
}

def escape(param):
    return param.lower().replace(" ", "_").replace("'", "_").replace('"', '_').replace("\n", "_")\
            .replace("\t", "_").replace("\r", "_").replace("/", "_").replace("\\", "_")

def count_actions():
    counter = {"actions":{}, "tools":{}}
    by_pages = {}
    with os.scandir("../data/pre_recipes") as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                if not entry.path.endswith("result.txt"):
                    for title in pages.keys():
                        if title in entry.path:
                            with open(entry.path, 'r') as f:
                                items = json.load(f)

                            for key in items:
                                if os.path.exists('../recipes/actions/' + escape(key['intent'])):
                                    if title not in counter["actions"]:
                                        counter["actions"][title] = 0
                                    counter["actions"][title] += 1
                                    
                                    if title not in by_pages:
                                        by_pages[title] = []
                                    by_pages[title].append(escape(key['intent']))

                                if os.path.exists('../recipes/tools/' + escape(key['intent'])):
                                    if title not in counter["tools"]:
                                        counter["tools"][title] = 0
                                    counter["tools"][title] += 1
                                    
    with open("../data/tools_by_pages", "w") as f:
        json.dump(by_pages, f, indent=4)

    print(counter)

def create_actions():
    with os.scandir("../recipes/actions") as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                # if not entry.path.endswith("result.txt"):
                    with open(entry.path, 'r') as f:
                        print(entry.path)
                        items = json.load(f)

                    for key in items['workflow']:
                        if key['label'] == 'backend':
                            with open("../recipes/backend.py", "a") as f:
                                f.write("# step:"+str(key['step'])+" file: "+entry.name+"\n")
                                f.write(key['code'])
                                f.write("\n\n\n")
                        else:
                            with open("../recipes/frontend.jsx", "a") as f:
                                f.write("// step:"+str(key['step'])+" file: "+entry.name+"\n")
                                f.write(key['code'])
                                f.write("\n\n\n")

def create_functions_json():
    # with open("../recipes/functions.json", 'r') as f:
    #     functions = json.load(f)
    functions = {'backend':{}, 'frontend':{},}
        
    with os.scandir("../recipes/actions") as entries:
        for entry in entries:
            if entry.is_file():  # Check if it's a file
                # if not entry.path.endswith("result.txt"):
                    with open(entry.path, 'r') as f:
                        print(entry.path)
                        items = json.load(f)

                    for key in items['workflow']:
                        if key['label'] == 'backend':
                            function = key['code'][key['code'].find("def "):]
                            name = function[4:function.find("(")].strip()

                            if name not in functions['backend'].keys():
                                functions['backend'][name] = function
                        else:
                            function = key['code'][key['code'].find("export const "):]
                            name = function[13:function.find("=")].strip()

                            if name not in functions['frontend'].keys():
                                functions['frontend'][name] = function

    with open("../recipes/functions.json", 'w') as f:
        json.dump(functions, f, indent=4)



if __name__ == "__main__":
    create_functions_json()