from typing import Dict
from fastapi import FastAPI, Depends
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()

class ModelRequest(BaseModel):
    text:str

class ModelResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

@app.post('/predict', response_model = ModelResponse)
def predict(request: ModelRequest, model: Model = Depends(get_model)): #dependence injection
    probabilities, sentiment, confidence = model.predict(request.text)
    return ModelResponse(
        probabilities = probabilities, 
        sentiment = sentiment, 
        confidence = confidence
    )