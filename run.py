import io
import os
import re
import sys
import torch
import whisper
import argparse

from tqdm import tqdm


def open_utt2value(file):
    s = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l_ = l.split()
            s[l_[0]] = l_[1]
    return s


def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_wav_scp_file_path",
                        default='data/trn/text_list',
                        type=str)

    parser.add_argument("--output_text_file_path",
                        default='CEFR_LABELS_PATH/trn_sst_scores.txt',
                        type=str)

    parser.add_argument("--level",
                        default='tiny',
                        type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # argparse
    args = argparse_function()

    assert args.level in ['tiny', 'base', 'small', 'medium']

    # variable
    input_wav_scp_file_path = args.input_wav_scp_file_path
    output_text_file_path = args.output_text_file_path

    wav_scp_dict = open_utt2value(input_wav_scp_file_path)
    
    save_file = open(output_text_file_path, 'w')
    
    # model
    model = whisper.load_model("{}.en".format(args.level))
    
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # retrieve sst score
    utt2text = {}
    for i, (utt_id, wav_abs_path) in enumerate(tqdm(wav_scp_dict.items())):

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(wav_abs_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # decode the audio
        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(model, mel, options)

        # print the recognized text
        utt2text[utt_id] = result.text
        save_file.write("{} {}\n".format(utt_id, result.text))

    # save
    save_file.close()