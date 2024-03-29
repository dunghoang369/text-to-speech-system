import json
import sys
sys.path.append("/home/dunghoang300699/Downloads/mlops/module3/tts/tts")
sys.path.append("/home/dunghoang300699/Downloads/mlops/module3/tts")
import torch
import bentoml
import onnxruntime
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel
from tts.common.text import cmudict
from logs.logger import bentoml_logger
from tts.common.text.text_processing import get_text_processing
cmudict.initialize("cmudict/cmudict-0.7b", "cmudict/heteronyms")

class TCPConnection(BaseModel):
    text: str = "Hello world"

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class texttospeechrunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = "cpu" # Specify ("nvidia.com/gpu", "cpu") if GPU is supported
    SUPPORTS_CPU_MULTI_THREADING = True # Support multi-threading for this model

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fastpitch = onnxruntime.InferenceSession("/home/dunghoang300699/Downloads/mlops/module3/tts/onnx_models/fast_pitch.onnx")
        self.hifigan = onnxruntime.InferenceSession("/home/dunghoang300699/Downloads/mlops/module3/tts/onnx_models/hifigan.onnx")
        self.tp = get_text_processing("english_basic", ['english_cleaners_v2'], 1.0)

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_data):

        # Get text
        text = input_data.dict()["text"] + ","
        
        # Debug input data
        bentoml_logger.info("Input data:")
        bentoml_logger.info(text)
        
        # Process text
        inputs = torch.LongTensor(self.tp.encode_text(text + ","))
        inputs = torch.unsqueeze(inputs, 0)
        inputs_onnx = to_numpy(inputs)

        # Gen mel
        # results = self.fastpitch.run(None, {"inputs": inputs_onnx, "onnx::Cast_2": np.array((6)).astype(np.int64)})
        results = self.fastpitch.run(None, {"inputs": inputs_onnx})
        mel_onnx = results[0]
        mel_lens_onnx = torch.tensor(results[1])
        mel_lens_onnx = mel_lens_onnx.to(self.device)

        # Gen audio
        results = self.hifigan.run(None, {"inputs": mel_onnx})
        audios_onnx = results[0].squeeze(1) * 32768.0
        audios_onnx = torch.tensor(audios_onnx)
        audios_onnx = audios_onnx.to(self.device)
        for i, audio in enumerate(audios_onnx):
            audio = audio[:int(mel_lens_onnx[i].item()) * 256]
            fade_len = 10 * 256
            fade_w = torch.linspace(1.0, 0.0, fade_len)
            audio[-fade_len:] *= fade_w.to(audio.device)
            audio = audio / torch.max(torch.abs(audio))
            
        return audio
    
# Create a runner object
ad_runner = bentoml.Runner(texttospeechrunner)

# Create BentoML service
svc = bentoml.Service("text_to_speech_api", runners=[ad_runner])

# Create our API with route /detection
@svc.api(input=JSON(pydantic_model=TCPConnection), output=JSON())
async def detection(input_data):
    with bentoml.monitor("text_to_speed") as mon:
        mon.log(input_data.dict(), name="record", role="", data_type="")

    pred = await ad_runner.inference.async_run(input_data)
    pred = list(to_numpy(pred))
    return {"result": pred}



