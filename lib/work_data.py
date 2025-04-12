"""
Stub for loading data from a DB
"""
import json

import psycopg2

class Work:
    conn = None
    cur = None

    def __init__(self) -> None:
        ...

    # is_allowed – flag that show which table data will be saved – general queue table or restricted one
    def save_work(self, client, req, valid, is_allowed):
        with open("../data/" + client + "/" + req['id'], "w") as f:  # opening a file handler to create new file
            json.dump(req, f)  # writing content to file
        print(str(valid).lower())

        ...

        sql_query = sql if is_allowed else sql_restricted

        self.cur.execute(sql_query, ("../data/" + req['id'], json.dumps(req), client, 'false', valid))

        # get the generated id back
        rows = self.cur.fetchone()
        if rows:
            print(rows[0])

    def get_work_by_id_and_clent(self, id, client_type):
        ...

        self.cur.execute(sql)
        query_result = self.cur.fetchone()

        if query_result and query_result[2] == client_type:
            return query_result[1]

        return None

    def close(self):
        self.conn.commit()
        self.conn.close()

    def is_valid(self, req):
        ...
