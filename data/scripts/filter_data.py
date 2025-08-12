"""
Filter data pairs and retain only grammatical and frequently used commands.

Usage: python3 filter_data.py data_dir
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os, sys
import random

from bashlint import bash, data_tools

data_splits = ['train', 'dev', 'test']

NUM_UTILITIES = 100

def compute_top_utilities(path, k):
    print('computing top most frequent utilities...') 
    utilities = collections.defaultdict(int)
    with open(path, encoding='utf-8') as f:
        while (True):
            command = f.readline().strip()
            if not command:
                break
            ast = data_tools.bash_parser(command, verbose=False)
            for u in data_tools.get_utilities(ast):
                utilities[u] += 1
    top_utilities = []

    freq_threshold = -1   
    for u, freq in sorted(utilities.items(), key=lambda x:x[1], reverse=True):
        if freq_threshold > 0 and freq < freq_threshold:
            break 
        if u in bash.BLACK_LIST or u in bash.GREY_LIST:
            continue
        top_utilities.append(u)
        print('{}: {} ({})'.format(len(top_utilities), u, freq))
        if len(top_utilities) == k:
            freq_threshold = freq
    top_utilities = set(top_utilities)
    return top_utilities


def filter_by_most_frequent_utilities(data_dir, num_utilities):
    def select(ast, utility_set):
        for ut in data_tools.get_utilities(ast):
            if not ut in utility_set:
                print('Utility currently not handled: {} - {}'.format(
                    ut, data_tools.ast2command(ast, loose_constraints=True).encode('utf-8')))
                return False
        return True

    cm_path = os.path.join(data_dir, 'all.cm')
    top_utilities = compute_top_utilities(cm_path, num_utilities)
    for split in ['all']:
        nl_file_path = os.path.join(data_dir, split + '.nl')
        cm_file_path = os.path.join(data_dir, split + '.cm')
        with open(nl_file_path, encoding='utf-8') as f:
            nls = [nl.strip() for nl in f.readlines()]
        with open(cm_file_path, encoding='utf-8') as f:
            cms = [cm.strip() for cm in f.readlines()]
        nl_outfile_path = os.path.join(data_dir, split + '.nl.filtered')
        cm_outfile_path = os.path.join(data_dir, split + '.cm.filtered')
        with open(nl_outfile_path, 'w', encoding='utf-8') as nl_outfile:
            with open(cm_outfile_path, 'w', encoding='utf-8') as cm_outfile:
                for nl, cm in zip(nls, cms):
                    if len(nl.split()) > 50:
                        print('lenthy description skipped: {}'.format(nl))
                        continue
                    ast = data_tools.bash_parser(cm)
                    for ut in data_tools.get_utilities(ast):
                        print("-", ut)
                    if ast and select(ast, top_utilities):
                        nl_outfile.write('{}\n'.format(nl))
                        cm_outfile.write('{}\n'.format(cm))


def spil_files(limit, chosen):
    all = json.load(open("all.json"))
    limits = {}
    output = []
    keys = set()

    for k, v in all['ast'].items():
        print(limits)
        if not v['too_many_labels'] and v['top_label']:
            for ls in v['labels']:
                for l in ls['labels']:
                    if l in chosen:
                        set_b = set(ls['labels'])

                        intersection_set = chosen & set_b

                        if len(intersection_set) < 2:
                            if not l in limits:
                                limits[l] = 0
                            if limits[l] <= limit:
                                all['ast'][k]['dataset'] = True

                                if not k in keys:
                                    output.append({"text": k, "labels": [l]})
                                    keys.add(k)

                                limits[l] += 1
    others = []
    while len(others) < (limit*2):
        random_key = random.choice( list(all['others'].keys()) )
        if not random_key in others:
            others.append(random_key)
            all['others'][random_key]['dataset'] = True

            if not random_key in keys:
                output.append({"text": "test", "labels": ["others"]})
                keys.add(random_key)


    with open("chosen.json", 'w') as f:
        f.write(json.dumps(all))

    with open("train.jsonl", 'w') as f:
        f.write("")
    for o in output:
        with open("train.jsonl", 'a') as f:
            f.write(json.dumps(o) + "\n")
#     else:
#     with open("train_out.jsonl", 'a') as f:
#         f.write(json.dumps({"text": k, "labels": t['labels'], 'cmd': t['cmd']}) + "\n")
#         # else:
#         #     with open("train_out.jsonl", 'a') as f:
#         #         f.write(json.dumps({"text": nl.replace('"', '""'), "labels": [cm.split()[0]]}) + "\n")


def gen_non_specific_description_check_csv(data_dir):
    with open(os.path.join(data_dir, 'all.nl')) as f:
        nl_list = [nl.strip() for nl in f.readlines()]
    with open(os.path.join(data_dir, 'all.cm')) as f:
        cm_list = [cm.strip() for cm in f.readlines()]
    assert(len(nl_list) == len(cm_list))
    labels = json.load(open("labels.json"))

    sntc = {'ast':{}, 'others':{}}

    with open('annotation_check_sheet.non.specific.csv', 'w') as o_f:
        o_f.write('Utility,Command,Description\n')
        for nl, cm in zip(nl_list, cm_list):
            # o_f.write('"{}","{}","{}"\n'.format(cm.split()[0],cm.replace('"', '""'),
            #                                 nl.replace('"', '""')))

            # if "=" not in cm.split()[0]:
            #     if cm.split()[0] not in labels:
            #         labels[cm.split()[0]] = {}
            #         labels[cm.split()[0]]['cnt'] = 0
            #     else:
            #         labels[cm.split()[0]]['cnt'] += 1

            st_ = []

            ast = data_tools.bash_parser(cm)
            if ast:
                for ut in data_tools.get_utilities(ast):
                    st_.append(ut)
            if st_:
                if nl.replace('"', '""') not in sntc['ast']:
                    sntc['ast'][nl.replace('"', '""')] = {"labels": [{"text": nl.replace('"', '""'), "labels": st_, 'cmd': cm}]}
                else:
                    sntc['ast'][nl.replace('"', '""')]["labels"].append({"text": nl.replace('"', '""'), "labels": st_, 'cmd': cm})
            else:
                if nl.replace('"', '""') not in sntc['others']:
                    sntc['others'][nl.replace('"', '""')] = {"labels":[{"text": nl.replace('"', '""'), "labels": st_, 'cmd': cm}]}
                else:
                    sntc['others'][nl.replace('"', '""')]["labels"].append({"text": nl.replace('"', '""'), "labels": st_, 'cmd': cm})

    for k, v in sntc['ast'].items():
        if len(v) > 1:
            sntc['ast'][k]['too_many_labels'] = True
        else:
            sntc['ast'][k]['too_many_labels'] = False
            sntc['ast'][k]['top_label'] = False
            for t in v['labels']:
                for l in t['labels']:
                    if l in labels.keys():
                        sntc['ast'][k]['top_label'] = True

                    # if l not in labels:
                    #     labels[l] = 0
                    # else:
                    #     labels[l] += 1

    with open("all.json", 'w') as f:
        f.write(json.dumps(sntc))
    spil_files()


    # sorted_items = sorted(labels.items(), key=lambda item: item[1])
    # sorted_dict = dict(sorted_items)
    # with open("labels.json", 'w') as f:
    #     f.write(json.dumps(sorted_dict))

if __name__ == '__main__':
    dataset = sys.argv[1]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))), dataset)
    # gen_non_specific_description_check_csv(data_dir)
    spil_files(100, {"awk", "xargs", "find", "sort"})