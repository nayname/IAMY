import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from create import generate_code



# --- Placeholder for your generation logic ---
# This function simulates the work of your generate_code() and NER model.
# In your real application, you would replace this with your actual model inference.
def generate_mock_response(text: str):
    """
    Takes text and returns a mock dictionary with label, params, and command.
    """
    # Simple logic to choose a response based on keywords
    if "send" in text.lower() and "token" in text.lower():
        return {
            "label": "Send Tokens",
            "params": {"amount": "10 NTRN", "recipient": "ntrn1...", "wallet": "mywallet"},
            "command": "neutrond tx bank send mywallet ntrn1bobaddress... 100000000000000000000000000000000000000000000000000000000000000000000000000000untrn --gas auto"
        }
    elif "balance" in text.lower():
        return {
            "label": "Query Balance",
            "params": {"wallet": "$WALLET_ADDR"},
            "command": "neutrond query bank balances $WALLET_ADDR --node https://grpc-kaiyo-1.neutron.org:443"
        }
    else:
        return {
            "label": "Unknown Intent",
            "params": {},
            "command": "Could not generate a command for the given text."
        }


# --- FastAPI App Setup ---
app = FastAPI(
    title="Command Generator"
)

origins = [
   "https://thousandmonkeystypewriter.org", # The domain for Neutron docs
]

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"], # Allows all methods
   allow_headers=["*"], # Allows all headers
)

# Mount a directory for static files (like CSS, JS) if you have them
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Pydantic model for the request body to ensure type safety
class GenerateRequest(BaseModel):
    text: str


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def handle_generate(request_data: GenerateRequest):
    """
    Receives text from the frontend, generates a response,
    and returns it as JSON.
    """
    # Get the text from the request body
    input_text = request_data.text

    # Call your generation logic
    response_data = generate_code(input_text)
    print(response_data)

    return JSONResponse(content=response_data)
    # return templates.TemplateResponse("table.html", {"request": request, "table": map})

@app.get("/generate_")
async def handle_generate(intent: str):
    """
    Receives text from the frontend, generates a response,
    and returns it as JSON.
    """
    # Get the text from the request body
    # input_text = request_data.text

    # Call your generation logic
    response_data = generate_code(intent)
    return response_data

    # return JSONResponse(content=response_data)
    # return templates.TemplateResponse("table.html", {"request": request, "table": map})


if __name__ == "__main__":
    uvicorn.run(app, host='88.198.17.207', port=1958, reload=True)
