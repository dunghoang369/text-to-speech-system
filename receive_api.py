import torch
import uuid
import numpy as np
import json
from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI
from scipy.io.wavfile import write
from fastapi.encoders import jsonable_encoder

# Initialize instance
app = FastAPI()

class tts_output(BaseModel):
    result: list  # Speech features

@app.get("/")
def check_health():
    return {"status": "Oke"}

@app.post("/process")
def process(input_data: tts_output):
        input_data = jsonable_encoder(input_data)
        audio = input_data["result"]
        audio = np.array(audio)
        write(f"{uuid.uuid4()}.wav", 22050, audio)
        return {"response": 200}