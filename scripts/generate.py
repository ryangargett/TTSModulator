##################################################
#File: generate.py                               #
#Project: AdaptiveLLM                            #
#Created Date: Tue Jan 23 2024                   #
#Author: Ryan Gargett                            #
#-----                                           #
#Last Modified: Tue Jan 23 2024                  #
#Modified By: Ryan Gargett                       #
##################################################

from bark import SAMPLE_RATE, generate_audio
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform

from scipy.io.wavfile import write 
from pedalboard import Pedalboard, Compressor, LowShelfFilter, NoiseGate
import numpy as np
import noisereduce as nr
from argparse import ArgumentParser
import nltk

import utils.config as cfg

def _get_sentences(prompt):
    #nltk.download('punkt') # uncomment for first-time use
    
    sentences = nltk.sent_tokenize(prompt)
    return sentences

def collate_audio(sentences):
    
    padding = np.zeros(int(cfg.PADDING_LENGTH * SAMPLE_RATE))
    fragments = []
    
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(sentence,
                                                 history_prompt=cfg.VOICE_PRESET,
                                                 temp=cfg.SEMANTIC_TEMP,
                                                 min_eos_p=cfg.MIN_EOS_P)
        
        audio = semantic_to_waveform(semantic_tokens, history_prompt=cfg.VOICE_PRESET)
        fragments += [audio, padding.copy()]

    audio_collated = np.concatenate(fragments).astype(np.float32)
    return audio_collated


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
    
    sentences = _get_sentences(prompt)
    
    if len(sentences) > cfg.MIN_SENTENCES:
        audio = collate_audio(sentences)
    else:
        audio = generate_audio(prompt, history_prompt=cfg.VOICE_PRESET)
    
    processed_audio = process_audio(audio)
    
    write("output/generated.wav", SAMPLE_RATE, processed_audio)

if __name__ == "__main__":
    
    input_parser = ArgumentParser()
    input_parser.add_argument("--prompt", type=str, default="Hello! how are you?", help="Prompt to generate audio from")
    args = input_parser.parse_args()
    
    formatted_prompt = args.prompt.replace("\n", " ").strip()
    generate(formatted_prompt)