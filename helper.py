"""
Helper for refining and optimizing prompts, visualizing data and statistics
"""

import difflib
import getpass
import inspect
import itertools
import json
import os
import shutil
import urllib
from pathlib import Path

import psycopg2
import nbformat
import requests
from bs4 import BeautifulSoup
from nbconvert.preprocessors import ExecutePreprocessor

from app.validator import Validator
from config.config import withErrors, Config, create_check_error_config
from lib.parse_code import load_files, parse_json
from lib.query import get_context_content, proceed_query, proceed_query_full
from lib.response import type_to_response, answers, log

import feedparser


query_get_label_script = ...
query_get_label_no_script = ...
query_rephrase = ...

def check_types():
    ...

    cats, errors = done_lectures(cur, lectures_da+lectures_front+lectures_it_start, lecture_front_names)
    cur.execute(sql3)
    overall_res = overall(cur, cur1, lectures_da + lectures_front + lectures_it_start)
    res = {'cats':cats, 'errors':errors, 'overall':overall_res[0],
           'not_published':overall_res[1], 'not_201':overall_res[2]}


def get_name(conn, lecture):
    ...


def get_flow_report(conn, lecture, version):
    ...
    for row in result:
        ...
    return report


def get_status(row_dict, not_valid_errors):
    if ...
        return row_dict['payload']['status']
    else:
        res = []
        for f in row_dict['errors_check']:
            err = f.replace(';False', '').replace(';True', '')

            if err not in not_valid_errors:
                res.append(err)

        if len(res) > 0:
            return 'redirect'
        else:
            return 'graded'


def get_dataset_report(conn, lecture, version, not_valid_errors):
    ...

    report = {"rows": {"all": cur.rowcount, "errors":0, "declined":0},"errors": {}}

    # transform result
    ...
    perfect = {'count': 0, 'count_matched': 0, 'count_unmatched': 0, 'errors':{}}
    passed = {'count': 0, 'count_matched': 0, 'count_unmatched': 0, 'count_slight': 0, 'errors':{}}
    not_passed = {'count': 0, 'count_matched': 0, 'count_unmatched': 0, 'errors':{}}

    for row in result:
        row_dict = {}
        for i, col in enumerate(columns):
            row_dict[col.name] = row[i]

        status = get_status(row_dict, not_valid_errors)

        if row_dict['tutor_score'] == 100:
            perfect['count'] += 1

            if status == 'graded':
                perfect['count_matched'] += 1
            else:
                perfect['count_unmatched'] += 1

            ...

        else:
            ...


        if 'group_label' in row_dict.keys() and row_dict['group_label']:
            ...

    for k in perfect["errors"].keys():
        perfect["errors"][k] = perfect["errors"][k] / (perfect['count'] / 100)

    report['perfect'] = {'count':perfect['count'], 'count_matched':perfect['count_matched'], 'count_unmatched':perfect['count_unmatched'], "errors":dict(sorted(perfect["errors"].items(), key=lambda item: item[1], reverse=True))}
    report['passed'] = passed
    report['not_passed'] = not_passed

    return report


def get_descriptors():
    ""


def get_overall(report, courses, conn):
    res = {}
    for c in courses:
        ...

        for c in reported.keys():
            ...

        graded = 0
        overall = 0
        for c in reported.keys():
            ...

            report[c]['overall'] = {'graded':graded, 'overall':overall, 'percentages':graded/(overall/100)}

    return res


def report(lecture, version, conn):
    name = get_name(conn, lecture)
    report = {}

    report[lecture] = {"name":name, "flow report":get_flow_report(conn, lecture, version),
           "dataset report":get_dataset_report(conn, lecture, version, []),
           "descriptors":get_descriptors(), "version": version}
    return report, name['course_id']


def overall_analysis_from_dataset(id, version):
    ...

    output = []
    normals = []
    ids = []
    count = 0

    for i in cur.fetchall():
        if i[3] not in ids and i[5]:
            ...

            to_send = check_one(val, version, normals, ...)

            ...
    return True, None
