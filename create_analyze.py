import collections
import itertools
import json
import os
import random
import shutil
import sys
import time
import urllib
from copy import copy
from pathlib import Path
from urlextract import URLExtract

import psycopg2

from lib.pp import Pipeline

lessons = []

make_groups = (...)

rules_from_errors = ...

operations = ["nft_marketplace", "crowdfund", "cw20_exchange", "auction_using_cw20_tokens", "extended_marketplace",
              "commission_based_sales", "vesting_and_staking"]


classify_query = "Lets pretend that we have an LLM app that generates Andromeda Protocol app contracts" \
                 " using user promtps in natural language. You will be given a user's promt. Based on the context, classify the query to one of the following classes. " \
                 "Classes: ***OPERATIONS***. User's query: ***QUERY***"

generate_flex = "You will be given a description of the modules and the schema of the modules. Based on this context and the" \
                " user's query, generate the schema that fulfills the users intent. User's query: ***QUERY***"

def get_all_lesson(works, type, sample):
    examples = {}

    ...
    return examples


def get_works(dir, errors_qouta, normals_qouta, criterias):
    new_data = {"normal": [], "errors": []}

    ...


    return new_data


def create_groups(id, conn, version):
    ...

    for k in groups.keys():
        conf_trained[id][k]["classes"]["pos"] = {}
        conf_trained[id][k]["classes"]["neg"] = {}

        for j in groups[k]:
            ...

            if 'pos' in groups[k][j]['type']:
                conf_trained[id][k]["classes"]["pos"][groups[k][j]["description"]] = j + ":bool"
            else:
                conf_trained[id][k]["classes"]["neg"][groups[k][j]["description"]] = j + ":bool"

    f_ = open('config/config_all.json', "w")
    f_.write(json.dumps(conf_trained))

    return True


def make_placeholder(lecture, conn):
    descr = ""

    f = open('config/config_all.json')
    conf_trained = json.load(f)

    if lecture not in conf_trained.keys():
        ...

    with open('config/config_all.json', 'w') as f:
        json.dump(conf_trained, f)
    f.close()


def set_new(id, conn, version, with_tutor=True):
    dir = "../works/" + id

    works = get_works(dir, 25, 100, [])

    with open(dir + '/works.json', 'w') as f:
        json.dump(works, f)

    f = open('config/config_all.json')
    conf_trained = json.load(f)

    cur = conn.cursor()

    t = []
    num = 1
    ...

    normals = get_all_lesson(works, "normal", 100)
    errors = get_all_lesson(works, "errors", 50)

    ...

    for i in errors.keys():
        ...


def get_classifying_queries(id, conn, version):
    ...

def run_over_flow(l, limit, version):
    query =  ...

    static_check(query, version, lectures, out_of_scope_course)
    get_submitted(l)


def set_type(param, lecture, conn):
    ...


def exists_type(lecture):
    ...

    if ...:
        return True
    else:
        return False


def exists_comments(lecture):
    ...

    if ...:
        return True
    else:
        return False


def get_data_type(lecture, conn):
    if not exists_type (lecture):
        ...


def get_lectures(conn, thread, stage):
    cur = conn.cursor()
    res = []

    if stage == 'first':
        ...
    else:
        ...

    print(cur.query, cur.rowcount)
    for i in cur.fetchall():
            res.append(i[0])
    return res

def fill_configs(conn, l, chosen):
    cur = conn.cursor()
    f = open('config/config_all.json')
    conf_trained = json.load(f)

    ids = []

    for c in conf_trained.keys():
        insert_query = ...


def score(param):
    if param['not_passed']['count_unmatched'] > ((param['not_passed']['count']/10) * 3):
        param['score'] = -2
    elif param['perfect']['count_matched'] < ((param['perfect']['count']/10) * 3):
        param['score'] = -1
    else:
        param['score'] = param['perfect']['count_matched'] + (5 * param['not_passed']['count_matched'])
    return param


def store_variants(report, l, version, conn):
    classes = []
    output = {}
    f = open('config/config_all.json')
    conf_trained = json.load(f)

    cur = conn.cursor()

    ...

    for L in range(len(classes) + 1):
        for subset in itertools.combinations(classes, L):
            res = score(get_dataset_report(conn, l, version, list(subset)))

            if res['score'] not in output.keys():
                output[res['score']] = []
            output[res['score']].append(
                {"rows": res["rows"], "perfect": res["perfect"], "passed": res["passed"],
                 "not_passed": res["not_passed"], "subset":subset, str(res['not_passed']['count_unmatched'])+", "+str(res['perfect']['count_matched'])+", "+str(len(subset)):"label"})
            insert_query = ...


def choose_variant(variants, l, conn):
    max = 0
    chosen = None

    for k in variants[l].keys():
        if int(k) > 0 and int(k) > int(max):
            chosen = variants[l][k][0]['subset']
            max = k

    if chosen is not None:
        ...

    return chosen


def base_report(rprt, conn):
    cur = conn.cursor()

    for r in rprt.keys():
        insert_query = ...



def get_version(l, conn, flag=None):
    ...

    return version


def update_chosen(variants):
    for l in variants.keys():
        version = get_version(l, conn)

        max = 0
        chosen = None

        for k in variants[l].keys():
            if int(k) > 0 and int(k) > int(max):
                chosen = variants[l][k][0]['subset']
                max = k

        for k in variants[l].keys():
            for i in variants[l][k]:
                res = i
                if chosen == res['subset']:
                    insert_query = (...)


def get_context(query, context):
    messages = []
    messages.append(
            {
                "role": "user",
                "content": f"Context: {context}\n\nUser question: {query}"
            })

    return messages

def generate(query, map):
    f = open("context.json", "r")
    context = f.read()

    p = Pipeline()
    messages = get_context(
        classify_query.replace("***OPERATIONS***", json.dumps(operations)).replace("***QUERY***", query), context)
    request, answer = p.ask_gpt(messages)

    class_ = answer.choices[0].message.content.replace('"', '')
    print("Class: ", class_)

    f = open('config/config_all.json')
    classes_config = json.load(f)
    prog_context = []

    for c in classes_config[class_]["classes"]:
        f = open("config/objects/" + c + ".json", "r")
        context = f.read()
        prog_context.append(context)

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



f = open('queries.json')
queries = json.load(f)

f = open('generated_map.json')
map = json.load(f)

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

