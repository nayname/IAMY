import json
import random

from lib.pp import Pipeline

#ados types
operations = ["nft_marketplace", "crowdfund", "cw20_exchange", "auction_using_cw20_tokens", "extended_marketplace",
              "commission_based_sales", "vesting_and_staking"]
#unfine-tuned prompt intended for query classification
classify_query = "Lets pretend that we have an LLM app that generates Andromeda Protocol app contracts" \
                 " using user promtps in natural language. You will be given a user's promt. Based on the context, classify the query to one of the following classes. " \
                 "Classes: ***OPERATIONS***. User's query: ***QUERY***"
#unfine-tuned prompt intended for script generation
generate_flex = "You will be given a description of the modules and the schema of the modules. Based on this context and the" \
                " user's query, generate the schema that fulfills the users intent. User's query: ***QUERY***"

def get_context(query, context):
    """
    strings to context for prompt
    :param query:
    :param context:
    :return:
    """
    messages = []
    messages.append(
            {
                "role": "user",
                "content": f"Context: {context}\n\nUser question: {query}"
            })

    return messages


def parse_query(query):
    """
    query to params object
    :param query:
    :return:
    """
    pass


def context_from_params(params):
    """
    fill context wit query params
    :param params:
    :return:
    """
    pass

#
def generate(query, map):
    """
    test function to generate script from query
    :param query:
    :param map:
    :return:
    """

    f = open("context.json", "r")
    context = f.read()

    p = Pipeline()
    messages = get_context(
        classify_query.replace("***OPERATIONS***", json.dumps(operations)).replace("***QUERY***", query), context)
    request, answer = p.ask_gpt(messages)

    query_params = parse_query(query)

    class_ = answer.choices[0].message.content.replace('"', '')
    print("Class: ", class_)

    f = open('config/config_all.json')
    classes_config = json.load(f)
    prog_context = {"ados_components":[]}

    for c in classes_config[class_]["classes"]:
        f = open("config/objects/" + c + ".json", "r")
        context = f.read()
        prog_context["ados_components"].append(context)
    prog_context["application_description"] = classes_config[class_]["descr"]

    messages = get_context(generate_flex.replace("***QUERY***", query), json.dumps(prog_context))
    request, answer = p.ask_gpt(messages)

    hash = random.getrandbits(128)
    f = open("generated/" + str(hash) + "_" + class_, "a")
    f.write(answer.choices[0].message.content)
    f.close()

    map.append({"name":str(hash) + "_" + class_, "query":query, "label":class_})
    f = open('generated_map.json', "w")
    json.dump(map, f)
    f.close()


#synthesized queries
f = open('queries.json')
queries = json.load(f)

#list of all generated scripts (for the frontend)
f = open('generated_map.json')
map = json.load(f)

#examples
generate(queries['nft_marketplace'][0], map)
generate(queries['crowdfund'][0], map)
generate(queries['cw20_exchange'][0], map)
generate(queries['auction_using_cw20_tokens'][0], map)
generate(queries['extended_marketplace'][0], map)
generate(queries['commission_based_sales'][0], map)
generate(queries['vesting_and_staking'][0], map)
generate(queries['extended_marketplace'][1], map)
generate(queries['cw20_exchange'][1], map)
generate(queries['vesting_and_staking'][1], map)

