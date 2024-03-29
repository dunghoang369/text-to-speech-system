import sys
sys.path.append("/home/dunghoang300699/Downloads/mlops/module3/tts/tts")
sys.path.append("/home/dunghoang300699/Downloads/mlops/module3/tts")
import torch
import numpy as np
from scipy.io.wavfile import write
import numpy as np
import tritonclient.http as httpclient  
from tts.common.text import cmudict
from tts.common.text.text_processing import get_text_processing
cmudict.initialize("DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/cmudict/cmudict-0.7b", "DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/cmudict/heteronyms")
ENDPOINT = "172.17.0.1:8006"
model1 = "fastpitch"
model2 = "hifigan"
fade_out = 10
hop_length = 256
sampling_rate = 22050
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    # Define text need to be converted to audio
    text = "love didn't meet her at her best, love meets het at her mess"

    # Define the client to interact with the endpoint
    client = httpclient.InferenceServerClient(url=ENDPOINT)

    # Load text processing 
    tp = get_text_processing("english_basic", ['english_cleaners_v2'], 1.0)

    # Processing text
    inputs = torch.LongTensor(tp.encode_text(text + ","))
    inputs = torch.unsqueeze(inputs, 0)
    inputs_onnx = to_numpy(inputs)

    # Define some arguments
    detection_input = httpclient.InferInput("inputs", inputs_onnx.shape, datatype="INT64")
    detection_input.set_data_from_numpy(inputs_onnx, binary_data=True)

    # Get the response
    detection_output = httpclient.InferRequestedOutput("3168", binary_data=True)
    seq_lens = httpclient.InferRequestedOutput("seq_lens", binary_data=True)
    detection = client.infer(
        model_name=model1, inputs=[detection_input], outputs=[detection_output, seq_lens], model_version="1"
    )
    result = detection.as_numpy("3168")
    mel_lens_onnx = detection.as_numpy("seq_lens")
    mel_lens_onnx = torch.tensor(mel_lens_onnx)
    mel_lens_onnx = mel_lens_onnx.to(device)
    print(f"Received result buffer of size {result.shape}")
    print(mel_lens_onnx)


    # Define some arguments
    detection_input = httpclient.InferInput("inputs", result.shape, datatype="FP32")
    detection_input.set_data_from_numpy(result, binary_data=True)
    
    # Get the response
    detection_output = httpclient.InferRequestedOutput("372", binary_data=True)
    detection = client.infer(
        model_name=model2, inputs=[detection_input], outputs=[detection_output], model_version="1"
    )
    results = detection.as_numpy("372")
    print(f"Received result buffer of size {results.shape}")

    audios_onnx = results.squeeze(1) * 32768.0
    audios_onnx = torch.tensor(audios_onnx)
    audios_onnx = audios_onnx.to(device)
    print(audios_onnx)
    for i, audio in enumerate(audios_onnx):
        audio = audio[:int(mel_lens_onnx[i].item()) * hop_length]
        if fade_out:
            fade_len = fade_out * hop_length
            fade_w = torch.linspace(1.0, 0.0, fade_len)
            audio[-fade_len:] *= fade_w.to(audio.device)
        audio = audio / torch.max(torch.abs(audio))
        write(f"test.wav", sampling_rate, audio.cpu().numpy())
    

if __name__ =="__main__":
    main()




