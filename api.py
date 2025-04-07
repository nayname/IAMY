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

@app.get("/generate", response_class=HTMLResponse)
async def read_items():
    html_content = """
    <html>
        <head>
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
        </head>
        <body>
            <div id="resp-table">
                <div id="resp-table-body">
                    <div class="resp-table-row"> 
                        <div class="table-body-cell">
                            col 1 
                        </div>
                        <div class="table-body-cell">
                            col 2 
                        </div>
                        <div class="table-body-cell">
                            col 3 
                        </div>
                        <div class="table-body-cell">
                            col 4 
                        </div>
                    </div>
                    <div class="resp-table-row"> 
                        <div class="table-body-cell">
                            second row col 1 
                        </div>
                        <div class="table-body-cell">
                            second row col 2 
                        </div>
                        <div class="table-body-cell">
                            second row col 3 
                        </div>
                        <div class="table-body-cell">
                            second row col 4 
                        </div>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host='api.iawy.cc', port=80, reload=True)
