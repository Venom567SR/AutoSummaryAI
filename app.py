from fastapi import FastAPI, Request
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from AutoSummaryAI.pipeline.prediction import PredictionPipeline
from pydantic import BaseModel

# Create a model for the request body
class TextInput(BaseModel):
    text: str

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs-api")
async def docs_redirect():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(text_input: TextInput):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text_input.text)
        return {"summary": summary}
    except Exception as e:
        raise e

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)