import csv
import json
import sys

import psycopg2
import uvicorn
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from fastapi import FastAPI, Request, HTTPException
from lib.pp import Pipeline


csv.field_size_limit(sys.maxsize)

app = FastAPI(
    title="Query"
)

# Config of courses and lectures that we allow to process
with open('config/course_allowlist.json') as course_allowlist:
    # key - courseId
    # value - lectureId
    course_allowlist_config = json.load(course_allowlist)

records = []

app.mount("/static", StaticFiles(directory="static"), name="static")


def get_history(cur, user_query, uid, path, type):
    f = open(path, "r")
    context_string = (f.read())

    messages = []
    cur.execute(" SELECT role, message from report.history where uid='"+uid+"';")

    for i in cur.fetchall():
        messages.append({"role": i[0], "content": i[1]})
    # {“role”: “system”, “content”: system_instruction},
    # {“role”: “user”, “content”: “When does the sun come up in the summer?”},
    # {“role”: “assistant”, “content”: “Usually around 6am.”},
    # {“role”: “user”, “content”: “What about next season?”}

    if len(messages) < 1:
        messages.append({"role": "system",
         "content": "Ответьте на следующий вопрос, основываясь только на контексте. Отвечайте только из контекста. Если вы не знаете ответа, скажите 'Я не знаю'."
                    " Отвечайте максимально близко к контексту, слово в слово"})
        save_answer(cur, uid, 'system', "Ответьте на следующий вопрос, основываясь только на контексте. Отвечайте только из контекста. Если вы не знаете ответа, скажите 'Я не знаю'."
                    " Отвечайте максимально близко к контексту, слово в слово")

    if type != 'faq':
        messages[0] = {"role": "system",
         "content": "Вы можете отвечать используя также общие знания, не только контекст"}

    
    messages.append(
            {
                "role": "user",
                "content": f"Context: {context_string}\n\nUser question: {user_query}"
            })
    save_answer(cur, uid, 'user', user_query)
    return messages


def save_answer(cur, uid, role, content):
    insert_query = "INSERT INTO report.history (role, message, uid) VALUES (%s, %s, %s)"
    cur.execute(insert_query, (role, content, uid))


@app.get("/lavatop4.5/")
async def process_work(query: str, uid: str):
    conn = psycopg2.connect(database="backup",
                                 user='nayname', password='thDKkLifDWsXbmtLGhagzaz7H',
                                 host='88.198.17.207', port='5432'
                                 )

    conn.autocommit = True
    cur = conn.cursor()

    p = Pipeline()

    messages = get_history(cur, query, uid,"/root/data_parse_test/faq_lava.txt")
    request, answer = p.ask_gpt_support(messages,
                                        'gpt-4.5-preview')
    print("UID:",uid, "ANSWER:", answer.choices[0].message.content)
    save_answer(cur, uid, answer.choices[0].message.content)

    return answer.choices[0].message.content

@app.get("/easycode4.5/")
async def process_work(query: str):
    p = Pipeline()

    request, answer = p.ask_gpt_support(query, "/root/data_parse_test/faq_easycode.txt",
                                        'gpt-4.5-preview')
    print(answer.choices[0].message.content)

    f = open("/root/data_parse_test/ez_history", "a")
    f.write('\n'+query)
    f.close()

    return HTMLResponse (answer.choices[0].message.content)

@app.get("/lavatop4.0/")
async def process_work(query: str, uid: str):
    conn = psycopg2.connect(database="backup",
                                 user='nayname', password='thDKkLifDWsXbmtLGhagzaz7H',
                                 host='88.198.17.207', port='5432'
                                 )

    conn.autocommit = True
    cur = conn.cursor()

    p = Pipeline()

    type = 'faq'
    messages = get_history(cur, query, uid,"/root/data_parse_test/faq_lava.txt", type)
    request, answer = p.ask_gpt_support(messages,
                                        'gpt-4o')

    if 'я не знаю' in answer.choices[0].message.content.lower():
        type = 'all'
        messages = get_history(cur, query, uid, "/root/data_parse_test/faq_lava.txt", type)
        request, answer = p.ask_gpt_support(messages,
                                            'gpt-4o')

    if type == 'faq':
        response = "<b>ОТВЕТ ИЗ ФАК:</b> "+answer.choices[0].message.content
    else:
        response = "<b>ОБЩИЕ ЗНАНИЯ:</b> " + answer.choices[0].message.content
    # print("UID:",uid, "ANSWER:", answer.choices[0].message.content)
    request['uid'] = uid
    request['answer'] = response

    f = open("/root/faq/history", "w")
    f.write(json.dumps(request))
    f.close()

    save_answer(cur, uid, 'system', response)

    return response

@app.get("/urfu/")
async def process_work(id: str):
    # with open("/root/data_parse_test/57622256-d159-4a25-9d01-d439b536e7c2") as course_allowlist:
    #     current_work = json.load(course_allowlist)
    #
    # print(json.dumps(current_work))
    # val = Validator(json.dumps(current_work), 'config/config_all.json')
    #
    # json_result = val.get_response()
    # logger.info(json.dumps(json_result))
    print(id)
    return HTMLResponse (id)#json_result["response"]["check"]["response"]["summary"]["summary"])

@app.get("/lavatop_history/")
async def process_work():
    f = open("/root/data_parse_test/lavatop_history", "r", encoding="utf-8")

    return HTMLResponse (f.read())

if __name__ == "__main__":
    uvicorn.run(app, host='api.iawy.cc', port=80, reload=True)
