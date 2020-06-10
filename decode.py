#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import argparse
import os
import json

from distutils.util import strtobool

from wavenet import WaveNet
import torch
from batch_generator import decode_generator
from scipy.io import wavfile
from audio_utils import *

logging.basicConfig(format='%(message)s', level=logging.INFO)

def main():
    """RUN DECODING."""
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--featdir", required=True,
                        type=str, help="Directory of list of filenames of aux feat files")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="Model file")
    parser.add_argument("--config", required=True,
                        type=str, help="Directory of network config file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="Directory to save generated samples")
    parser.add_argument("--sr", default=16000,
                        type=int, help="Sampling rate")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Number of batch size in decoding")
    parser.add_argument("--mode", default="sampling",
                        type=str, help="Decoding mode(sampling or argmax)")
    parser.add_argument("--use_gpu", default=False,
                        type=strtobool, help="Whether to use gpu")
    # other setting
    parser.add_argument("--intervals", default=1000,
                        type=int, help="Log interval")

    args = parser.parse_args()

    # fix slow computation of dilated conv
    # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
    torch.backends.cudnn.benchmark = True

    # load config
    with open(os.path.join(args.config, 'config.json'), 'r') as j:
        config = json.load(j)

    # make out dir
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    model = WaveNet(
        n_quantize=config["n_quantize"], 
        n_aux=config["n_aux"], 
        n_resch=config["n_resch"], 
        n_skipch=config["n_skipch"],
        dilation_depth=config["dilation_depth"], 
        dilation_repeat=config["dilation_repeat"], 
        kernel_size=config["kernel_size"], 
        upsampling_factor=config["upsampling_factor"]
    )

    model.load_state_dict(torch.load(
        args.checkpoint,
        map_location=lambda storage,
        loc: storage)["model"])

    if args.use_gpu:
        model.cuda()
    
    model.eval()
    
    #define generator
    generator =  decode_generator(
        args.featdir,
        batch_size=args.batch_size,
        n_quantize=config["n_quantize"],
        upsampling_factor=config["upsampling_factor"],
        use_gpu=args.use_gpu)

    logging.info("**** DECODING FINISH ****")
    if args.batch_size==1:
        idx = 0
        for feat_name, (x, h, n_samples) in generator:
            idx += 1
            logging.info("decoding %s (length = %d)" % (feat_name, n_samples))
            samples = model.generate(x, h, n_samples, args.intervals, args.mode)
            wav = inv_mulaw_quantize(samples, config["n_quantize"])
            wavfile.write(args.outdir+"/"+feat_name+".wav", args.sr, wav)
            logging.info("wrote %s.wav in %s." % (feat_name, args.outdir))

        logging.info("%d wav files are decoded in %s.." % (idx, args.outdir))
        logging.info("**** DECODING FINISH ****")

    else:
        pass
        #TODO:// decode several batches simultaneously

if __name__ == "__main__":
    main()
