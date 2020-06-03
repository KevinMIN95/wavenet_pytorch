# coding: utf-8
import argparse
import os
from os.path import join
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from feature_extract import build_from_path


def preprocess(data_dir, out_dir, out_files_dir, num_workers, args, data_type):
    metadata = build_from_path(data_dir, out_files_dir, args, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir, data_type)


def write_metadata(metadata, out_dir, data_type):
    metadata2 = []
    with open(os.path.join(out_dir, 'wav-pre.scp'), 'w', encoding='utf-8') as f:
        for m in metadata:
            if m[2] != -1:
                f.write('|'.join([str(x) for x in m]) + '\n')
                metadata2.append(m)
    
    print('Wrote %d utterances, (%s)' % (len(metadata2), data_type))
    print('Min frame length: %d' % min(m[2] for m in metadata2))
    print('Max frame length: %d' % max(m[2] for m in metadata2))


if __name__ == "__main__":
    num_workers = cpu_count()
    
    parser = argparse.ArgumentParser(
        description="preprocess data sets")

    parser.add_argument("--data_dir", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument("--data_type", default=None, 
        help="train or test")  
    parser.add_argument("--out_dir", default=None,
        help="directory to save list of filename of preprocessed files")
    parser.add_argument("--out_files_dir", default=None,
        help="directory to save preprocessed files")

    parser.add_argument("--num_workers", default=0, type=int,
        help="",)

    parser.add_argument("--input_type", default = "mulaw-quantize", type=str,
        help="raw / mulaw / mulaw-quantize")
    parser.add_argument("--quantize_channels", default = 256, type=int,
        help="number of channels of quantization of wavefile")

    parser.add_argument("--sampling_frequency", default=16000, type=int,
        help="sampling frequency")
    parser.add_argument("--highpass_cutoff", default = 70, type=int,
        help="highpass filter cutoff frequency (if 0, will not apply)")
    parser.add_argument("--n_fft", default = 2048, type=int,
        help="fft length")
    parser.add_argument("--hop_length", default = 512, type=int,
        help="hop length")
    parser.add_argument("--n_mels", default = 128, type=int,
        help="number of mel bands to generate")

    args = parser.parse_args()

    if args.num_workers > 0 :
        num_workers = args.num_workers

    preprocess(str(args.data_dir), str(args.out_dir), str(args.out_files_dir), 
        num_workers, args, str(args.data_type))