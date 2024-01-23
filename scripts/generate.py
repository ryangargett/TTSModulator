from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write 
from pedalboard import Pedalboard, Compressor, LowShelfFilter, NoiseGate
import numpy as np
import noisereduce as nr
from argparse import ArgumentParser

import utils.config as cfg

def process_audio(audio):
    
    noise_reduction = nr.reduce_noise(audio, SAMPLE_RATE, stationary=True, prop_decrease=0.75)
    
    board = Pedalboard([
        Compressor(threshold_db=-16, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250)
    ])
    
    processed_audio = board(noise_reduction, SAMPLE_RATE)
    return processed_audio         

def generate(prompt):
    audio = generate_audio(prompt, history_prompt=cfg.VOICE_PRESET)
    processed_audio = process_audio(audio)
    
    write("output.wav", SAMPLE_RATE, processed_audio)

if __name__ == "__main__":
    
    input_parser = ArgumentParser()
    input_parser.add_argument("--prompt", type=str, default="This is a test prompt, hello!", help="Prompt to generate audio from")
    args = input_parser.parse_args()
    
    generate(args.prompt)