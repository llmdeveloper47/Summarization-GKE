from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Body, Request
from pathlib import Path
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()

class TextIn(BaseModel):
    text: list[str]

class PredictionOut(BaseModel):
    summary: list[str]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)



@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
	
    predicted_summary = predict_pipeline(payload.text)
    return PredictionOut(summary = predicted_summary)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
