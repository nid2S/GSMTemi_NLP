import os
import librosa
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import audio_func
from hparams import hparams as hps
warnings.simplefilter(action='ignore', category=FutureWarning)


def text_to_sequence(text: str) -> List[int]:
    # eos(~)에 해당하는 "1"이 끝에 붙는다.
    pass

def _process_utterance(out_dir, wav_path, text):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = librosa.core.load(wav_path, sr=hps.sample_rate)[0]
    except FileNotFoundError:  # catch missing wav exception
        print(f'file {wav_path} present in csv metadata is not present in wav folder. skipping!')
        return None

    # rescale wav
    if hps.rescaling:  # hparams.rescale = True
        wav = wav / np.abs(wav).max() * hps.rescaling_max

    # M-AILABS extra silence specific
    if hps.trim_silence:  # hparams.trim_silence = True
        wav = librosa.effects.trim(wav, top_db=hps.trim_top_db, frame_length=hps.trim_fft_size, hop_length=hps.trim_hop_size)[0]

    # Mu-law quantize, default 값은 'raw'
    if hps.input_type == 'mulaw-quantize':
        # [0, quantize_channels)
        out = audio_func.mulaw_quantize(wav, hps.quantize_channels)
        # Trim silences
        start, end = audio_func.start_and_end_indices(out, hps.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]
        constant_values = audio_func.mulaw_quantize(0, hps.quantize_channels)
        out_dtype = np.int16
    elif hps.input_type == 'mulaw':
        # [-1, 1]
        mu = hps.quantize_channels
        out = np.sign(wav) * np.log1p(mu * np.abs(wav)) / np.log1p(mu)
        constant_values = np.sign(0.) * np.log1p(mu * np.abs(0.)) / np.log1p(mu)
        out_dtype = np.float32
    else:  # raw
        # [-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio_func.melspectrogram(wav).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hps.max_mel_frames and hps.clip_mels_length:  # hparams.max_mel_frames = 1000, hparams.clip_mels_length = True
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio_func.linearspectrogram(wav).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    # sanity check
    assert linear_frames == mel_frames

    if hps.use_lws:  # hparams.use_lws = False
        # Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hps.fft_size if hps.win_size is None else hps.win_size
        l, r = audio_func.pad_lr(wav, fft_size, audio_func.get_hop_size())

        # Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        pad = int(hps.fft_size // 2)

        # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, pad, mode='reflect')

    assert len(out) >= mel_frames * audio_func.get_hop_size()

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio_func.get_hop_size()]
    assert len(out) % audio_func.get_hop_size() == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    wav_id = os.path.splitext(os.path.basename(wav_path))[0]

    # Write the spectrograms to disk:
    audio_filename = '{}-audio.npy'.format(wav_id)
    mel_filename = '{}-mel.npy'.format(wav_id)
    linear_filename = '{}-linear.npy'.format(wav_id)
    npz_filename = '{}.npz'.format(wav_id)
    npz_flag = True
    if npz_flag:
        data = {
            'audio': out.astype(out_dtype),
            'mel': mel_spectrogram.T,
            'linear': linear_spectrogram.T,
            'time_steps': time_steps,
            'mel_frames': mel_frames,
            'text': text,
            'tokens': text_to_sequence(text),
            'loss_coeff': 1
        }

        np.savez(os.path.join(out_dir, npz_filename), **data, allow_pickle=False)
    else:
        np.save(os.path.join(out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
        np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
        np.save(os.path.join(out_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, npz_filename

def build_from_path(in_dir: str, out_dir: str):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of npz files
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm_fn: Optional, provides a nice progress bar
    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """
    executor = ProcessPoolExecutor(max_workers=hps.n_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir, 'transcript.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            path, _, script = line.split("|")
            if not os.path.exists(path) or path[-4:] != ".wav":
                continue
            futures.append(executor.submit(partial(_process_utterance, out_dir, path, script)))
            index += 1

    result = []
    for future in tqdm(futures, desc=f"building from {in_dir}"):
        if future.result() is not None:
            result.append(future.result())
    return result

def preprocess(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hps.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    preprocess(args.in_dir, args.out_dir)
