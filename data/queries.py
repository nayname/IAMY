import json
import psycopg2

# You will be given a list of MCP tools. These tools intended to work with Neutron. Based on descriptions, coinsolidate this this tools in a bigger tool. Each of these tools:
# a) should contain detailed description of the Consolidated tool
# b) labels of corresponding from the original list tools
# c) propose a scaffold of each consolidated module. While consolidateing, consider possibility if implementing the consolidated tools, try not to inflate the functionality by joining to much different tools into one
# d) embed the tools from the original list that duplicates functionality of that consolidated tool
# Consolidated tools should include all  tools from the original list, you can include each tool from the original list only in one Consolidated tool. Provide your answer in strict JSON format {"label_of_group1":{"description1","scaffold1","tools_included":[tools icluded]}, "label_of_group2":{"description2","scaffold2","tools_included":[tools icluded]},}. List of tools follows:

# lets suppose we have an application (MCP server) that executes different user intents using MCP tools, each of them implements functions based on Neutron docs. I will give you several intents, response me with JSON in format [
# {"intent1_label": {"intent": "original text of intent","tools": [{"name_of_tool": "description of tool","name_of_tool": "description of tool"}]}},
# {"intent2_label": {"intent": "original text of intent","tools": [{"name_of_tool": "description of tool","name_of_tool": "description of tool"}]}}]. Intents:

# Based on description of the app (https://hackmd.io/@augustas/ByMnh1jSeg) synthesize me a 10 plausible user queries for an intent "Query Contract State Intent: Query smart contract state. Action: Run neutrond query wasm contract-state smart <contract-address> <query-msg> --node <grpc_node>. Outcome: Returns query result from the smart contract state."

# lets suppose we have an application (MCP server) that executes different user intents using MCP tools, each of them implements functions based on Neutron docs.
# You will be given a list of objects.
# For each object:
# take 'intent'
# take the list 'stages' that describes MCP tools
#  add variable 'workflow' to describe how this intent will be proceeded with that MCP tools, which functionality based on the 'stages'. Output modified input JSON by adding 'workflow' to each original object. List of objects:

conn = psycopg2.connect(database="backup",
                        user='nayname', password='thDKkLifDWsXbmtLGhagzaz7H',
                        host='88.198.17.207', port='5432'
                        )

conn.autocommit = True
cur = conn.cursor()

cur.execute("select * from stage.queries")
print(cur.query, cur.rowcount)

save = []
types = []

for i in cur.fetchall():
    print(i)


f = open('/root/andromeda/IAMY/data/tools.json')
tools = json.load(f)

f = open('/root/andromeda/IAMY/data/intents.json')
data = json.load(f)

for d in data:
    tls = []
    for key_d in d['tools']:
        tls.append({'name':key_d, 'scaffold':tools[key_d]['scaffold']})
        sql3 = "INSERT INTO stage.tools (name, scaffold) VALUES (%s, %s) on conflict do nothing;"
        cur.execute(sql3, (key_d, tools[key_d]['scaffold']))

    sql3 = "INSERT INTO stage.queries (type, intent, workflow, tools) VALUES (%s, %s, %s, %s) on conflict do nothing;"
    cur.execute(sql3, (d['type'], d['intent'], d['workflow'], json.dumps(tls)))

die
f = open('/root/andromeda/IAMY/data/tools.json')
tools = json.load(f)

f = open('/root/andromeda/IAMY/data/queries_json_one_list.json')
data = json.load(f)
for d in data:
    new_obj = {}
    for key_d, value_d in d.items():
        new_obj['intent'] = d[key_d]['intent']
        tools_list = []
        scaffold_list = []

        for k in d[key_d]['tools']:
            for key_k, value_k in k.items():
                found = None

                for tool_k, tool_v in tools.items():
                    for t_k in tools[tool_k]['tools_included']:
                        if key_k == t_k:
                            found = tool_k
                            scaffold = tools[tool_k]['scaffold']

                if not found:
                    print(key_k)
                    die

                if found not in tools_list:
                    tools_list.append(found)
                    scaffold_list.append(scaffold)

        tools_list.sort()
        if not tools_list in types:
            types.append(tools_list)

        new_obj['type'] = types.index(tools_list)
        new_obj['tools'] = tools_list
        new_obj['scaffolds'] = scaffold_list

    save.append(new_obj)

with open('/root/andromeda/IAMY/data/intents.json', "w") as f:  # opening a file handler to create new file
    json.dump(save, f)  # writing content to file

with open('/root/andromeda/IAMY/data/types.json', "w") as f:  # opening a file handler to create new file
    json.dump(types, f)  # writing content to file

die
f = open('/root/andromeda/IAMY/data/queries_json_one_list.json')
data = json.load(f)
for d in data:
    for key_d, value_d in d.items():
        for k in d[key_d]['tools']:
            for key_k, value_k in k.items():
                if key_k in save:
                    save[key_k]['count'] = save[key_k]['count']+1
                    save[key_k]['descriptions'].append(value_k)
                else:
                    save[key_k] = {}
                    save[key_k]['count'] = 1
                    save[key_k]['descriptions'] = [value_k]

with open('/root/andromeda/IAMY/data/saved.json', "w") as f:  # opening a file handler to create new file
    json.dump(save, f)  # writing content to file