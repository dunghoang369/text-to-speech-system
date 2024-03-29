import torch
import numpy as np
import onnxruntime
from common.text import cmudict
from scipy.io.wavfile import write
from common.text.text_processing import get_text_processing

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize parameters
    text = "love didn't meet her at her best, love meets het at her mess"
    symbol_set = "english_basic"
    text_cleaners = ['english_cleaners_v2']
    p_arpabet = 1.0 
    cmudict.initialize("cmudict/cmudict-0.7b", "cmudict/heteronyms")
    fade_out = 10
    hop_length = 256
    sampling_rate = 22050
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)

    # Process text
    inputs = torch.LongTensor(tp.encode_text(text + ","))
    inputs = torch.unsqueeze(inputs, 0)
    inputs_onnx = to_numpy(inputs)

    # Gen mel
    onnx_session = onnxruntime.InferenceSession(
        "onnx_models/fast_pitch.onnx")
    results = onnx_session.run(None, {"inputs": inputs_onnx, "onnx::Cast_2": np.array((6)).astype(np.int64)})

    mel_onnx = results[0]
    mel_lens_onnx = torch.tensor(results[1])
    print("mel_lens_onnx: ", mel_lens_onnx)
    mel_lens_onnx = mel_lens_onnx.to(device)

    # Gen audio
    onnx_session = onnxruntime.InferenceSession("onnx_models/hifigan.onnx")
    results = onnx_session.run(None, {"inputs": mel_onnx})
    print(results[0].squeeze(1).shape)
    # audios_onnx = results[0].squeeze(1) * 32768.0
    # audios_onnx = torch.tensor(audios_onnx)
    # audios_onnx = audios_onnx.to(device)
    # for i, audio in enumerate(audios_onnx):
    #     audio = audio[:int(mel_lens_onnx[i].item()) * hop_length]
    #     if fade_out:
    #         fade_len = fade_out * hop_length
    #         fade_w = torch.linspace(1.0, 0.0, fade_len)
    #         audio[-fade_len:] *= fade_w.to(audio.device)
    #     audio = audio / torch.max(torch.abs(audio))
    #     print("-----------------------", audio.shape)
    #     write(f"test.wav", sampling_rate, audio.cpu().numpy())

