from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import audio_utils as U
from scipy.io import wavfile
import logging
import sys
import librosa
from glob import glob
from os.path import join
from functools import partial
from os.path import exists, basename, splitext


def build_from_path(srcs, out_dir, args, num_workers=1, tqdm=lambda x: x):
    """Extract Features from Audio

    Arguments:
        srcs {string} -- list of src names
        out_dir {string} -- directory to save features
    """
    
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(srcs, 'r') as f:
        src_files = f.readlines()
        for wav_path in src_files:
            wav_path = wav_path.rstrip()
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, wav_path, args)))
        index += 1
    return [future.result() for future in tqdm(futures)]

signed_int16_max = np.iinfo(np.int16).max

def _process_utterance(out_dir, wav_path, args):

    # Load the audio to a numpy array:
    sf, wav = wavfile.read(wav_path)

    if not sf == args.sampling_frequency:
        logging.error("sampling frequency is not matched.")
        sys.exit(1)

    # [-1, 1]
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / signed_int16_max

    # Trim begin/end silences
    wav = U.audio_trim(wav)

    # High Pass Filter
    if args.highpass_cutoff > 0.0:
        wav = U.low_cut_filter(wav, args.sampling_frequency, args.highpass_cutoff)

    if args.input_type == 'mulaw-quantize':
        # qunatize_channels = 256
        # [0, 255]
        constant_values = U.mulaw_quantize(0, args.quantize_channels - 1)
        out_dtype = np.int16
    elif args.input_type == 'mulaw':
        # [-1, 1]
        constant_values = U.mulaw(0.0, args.quantize_channels - 1)
        out_dtype = np.float32
    else:
        # [-1, 1]
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = U.logmelspectrogram(wav, sf, 
            n_fft=args.n_fft, hop_length = args.hop_length, n_mels= args.n_mels).astype(np.float32).T

    # Clip
    if np.abs(wav).max() > 1.0:
        print("""Warning: abs max value exceeds 1.0: {}""".format(np.abs(wav).max()))
        # ignore this sample
        return ("dummy", "dummy", -1)
    wav = np.clip(wav, -1.0, 1.0)

    # Set waveform output
    if args.input_type == 'mulaw-quantize':
        out = U.mulaw_quantize(wav, args.quantize_channels - 1)
    elif args.input_type == 'mulaw':
        out = U.mulaw(wav, args.quantize_channels - 1)
    else:
        out = wav

    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    N = mel_spectrogram.shape[0]
    len_mel = N * args.hop_length
    if((len_mel - len(out)) > 0) :
        out = np.pad(out, (0, len_mel - len(out)), mode="constant", constant_values=constant_values)

    assert len(out) >= len_mel

    # Write the spectrograms to disk:
    name = splitext(basename(wav_path))[0]
    audio_filename = '%s-wave.npy' % (name)
    audio_path = os.path.join(out_dir, audio_filename)
    mel_filename = '%s-feats.npy' % (name)
    mel_path = os.path.join(out_dir, mel_filename)

    np.save(audio_path,
            out.astype(out_dtype), allow_pickle=False)
    np.save(mel_path,
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_path, mel_path, N)
