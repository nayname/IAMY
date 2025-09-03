import json
import os

"""
Given the lift of functions - remove duplicates (the ones that duplicate the functionality with the same input arguments). Do not consolidate any functions, just remove duplicates. Provide the final file.
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
                            with open("../recipes/backend_cron.py", "a") as f:
                                f.write("# step:"+str(key['step'])+" file: "+entry.name+"\n")
                                f.write(key['code'])
                                f.write("\n\n\n")
                        else:
                            with open("../recipes/frontend_cron.jsx", "a") as f:
                                f.write("// step:"+str(key['step'])+" file: "+entry.name+"\n")
                                f.write(key['code'])
                                f.write("\n\n\n")



if __name__ == "__main__":
    create_actions()