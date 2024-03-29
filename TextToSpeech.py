import torch
import numpy as np
import json
from loguru import logger
from pydantic import BaseModel
import onnxruntime
from tts.common.text import cmudict
from tts.common.text.text_processing import get_text_processing
from fastapi import FastAPI
cmudict.initialize("tts/cmudict/cmudict-0.7b", "tts/cmudict/heteronyms")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fastpitch = onnxruntime.InferenceSession("onnx_models/fast_pitch.onnx")
hifigan = onnxruntime.InferenceSession("onnx_models/hifigan.onnx")
tp = get_text_processing("english_basic", ['english_cleaners_v2'], 1.0)

# Initialize instance
app = FastAPI()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
# Class to define the request body
class tts_input(BaseModel):
    Text: str = "Hello World"  # Number of times pregnant

@app.get("/")
def check_health():
    return {"status": "Oke"}

# Initialize cache
cache = {}

@app.post("/predict")
def predict(input_data: tts_input):
    if str(input_data) in cache:
        logger.info("Getting result from cache!")
        return cache[str(input_data)]
    else:
        logger.info("Making predictions...")
        logger.info(input_data)
        
        # Get text
        text = input_data.dict()["Text"] + ","

        # Process text
        inputs = torch.LongTensor(tp.encode_text(text + ","))
        inputs = torch.unsqueeze(inputs, 0)
        inputs_onnx = to_numpy(inputs)

        # Gen mel
        # results = fastpitch.run(None, {"inputs": inputs_onnx, "onnx::Cast_2": np.array((6)).astype(np.int64)})
        results = fastpitch.run(None, {"inputs": inputs_onnx})
        mel_onnx = results[0]
        mel_lens_onnx = torch.tensor(results[1])
        mel_lens_onnx = mel_lens_onnx.to(device)

        # Gen audio
        results = hifigan.run(None, {"inputs": mel_onnx})
        audios_onnx = results[0].squeeze(1) * 32768.0
        audios_onnx = torch.tensor(audios_onnx)
        audios_onnx = audios_onnx.to(device)
        for i, audio in enumerate(audios_onnx):
            audio = audio[:int(mel_lens_onnx[i].item()) * 256]
            fade_len = 10 * 256
            fade_w = torch.linspace(1.0, 0.0, fade_len)
            audio[-fade_len:] *= fade_w.to(audio.device)
            audio = audio / torch.max(torch.abs(audio))
        
        audio = list(to_numpy(audio))
        audio = [float(value) for value in audio]
        logger.info(audio)
        logger.info(type(audio))
        return {"result": audio}