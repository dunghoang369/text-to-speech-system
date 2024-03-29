import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.functional import l1_loss
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

import models
from common import gpu_affinity
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import get_text_processing
from common.utils import l2_promote
from fastpitch.pitch_transform import pitch_transform_custom
from hifigan.data_function import MAX_WAV_VALUE, mel_spectrogram
from hifigan.models import Denoiser
import onnxruntime

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input text')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    return parser
    
def to_numpy(tensor):
    """
    Convert tensor to numpy array
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
def main():
    # Initialize needed parameters and parser
    fade_out = 10
    hop_length = 256
    p_arpabet = 1.0 
    sampling_rate = 22050
    symbol_set = "english_basic"
    text_cleaners = ['english_cleaners_v2']
    cmudict.initialize("cmudict/cmudict-0.7b", "cmudict/heteronyms")
    gen_kw = {'pace': 1.0,
            'speaker': 0,
            'pitch_tgt': None,
            'pitch_transform': None}
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models wanted to convert to onnx
    fastpitch, _, model_train_setup = models.load_and_setup_model(
            'FastPitch', parser, "models_storage/FastPitch_checkpoint_1000.pt", False, device,
            unk_args=[], forward_is_infer=True, jitable=False)
    
    vocoder, _, voc_train_setup = models.load_and_setup_model(
        'HiFi-GAN', parser, "models_storage/hifigan_gen_checkpoint_10000_ft.pt", False, device,
        unk_args=[], forward_is_infer=True, jitable=False)
    
    # Process input text
    tp = get_text_processing(symbol_set, text_cleaners, p_arpabet)
    inputs = torch.LongTensor(tp.encode_text(args.input + ","))
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.to(device)

    # Convert fastpitch to onnx
    torch.onnx.export(fastpitch,
                  inputs,
                  "onnx_models/fast_pitch.onnx",
                  input_names=["inputs"],
                  opset_version=14,
                  dynamic_axes={"inputs": [1]})
    
    with torch.no_grad():
        mel, mel_lens, *_ = fastpitch(inputs)

    # Convert hifigan to onnx
    torch.onnx.export(vocoder,
                    mel,
                    "onnx_models/hifigan.onnx",
                    input_names=["inputs"],
                    opset_version=14,
                    dynamic_axes={"inputs": [2]})


if __name__ == "__main__":
    main()