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

records = []

app.mount("/static", StaticFiles(directory="generated"), name="generated")

@app.get("/generate", response_class=HTMLResponse)
async def read_items():
    html_content = """
    <html>
        <head>
            <style>
                #resp-table {
                    width: 100%;
                    display: table;
                }
                #resp-table-body{
                    display: table-row-group;
                }
                .resp-table-row{
                    display: table-row;
                }
                .table-body-cell{
                    display: table-cell;
                    border: 1px solid #dddddd;
                    padding: 8px;
                    line-height: 1.42857143;
                    vertical-align: top;
                }
            </style>
        </head>
        <body>
            <div id="resp-table">
                <div id="resp-table-body">
                    <div class="resp-table-row"> 
                        <div class="table-body-cell">
                            App
                        </div>
                        <div class="table-body-cell">
                            User's query 
                        </div>
                        <div class="table-body-cell">
                            Url 
                        </div>
                    </div>
                        ***TABLE***
                </div>
            </div>
        </body>
    </html>
    """

    f = open('generated_map.json')
    map = json.load(f)

    rows = ""
    for m in map:
        rows += """<div class="resp-table-row"> 
                        <div class="table-body-cell">
                            """+m['label']+"""
                        </div>
                        <div class="table-body-cell">
                            """+m['query']+"""
                        </div>
                        <div class="table-body-cell">
                            <a href='http://88.198.17.207:1962/static/"""+m['name']+"""'>"""+m['name']+"""</a>
                        </div>
                    </div>"""

    return HTMLResponse(content=html_content.replace("***TABLE***", rows), status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host='api.iawy.cc', port=80, reload=True)
