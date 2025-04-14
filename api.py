import csv
import json
import sys

import psycopg2
import uvicorn
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from fastapi import FastAPI, Request, HTTPException
from starlette.templating import Jinja2Templates

from lib.pp import Pipeline


csv.field_size_limit(sys.maxsize)

app = FastAPI(
    title="Query"
)

records = []

app.mount("/static", StaticFiles(directory="generated"), name="generated")
templates = Jinja2Templates(directory = "templates")

@app.get("/generate", response_class=HTMLResponse)
async def read_items(request : Request):
    """
    table with generated scripts
    :return:
    """
    f = open('generated_map.json')
    map = json.load(f)

    return templates.TemplateResponse("table.html", {"request": request, "table": map})


if __name__ == "__main__":
    uvicorn.run(app, host='api.iawy.cc', port=80, reload=True)
