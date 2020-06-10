#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import time
import json

from dateutil.relativedelta import relativedelta
from distutils.util import strtobool

import numpy as np
import torch
from batch_generator import batch_generator
from torch import nn
from wavenet import WaveNet, initialize
from audio_utils import *
from tensorboard_writer import TensorboardWriter

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """SAVE CHECKPOINT.

    Args:
        checkpoint_dir (str): Directory to save checkpoint.
        model (torch.nn.Module): Pytorch model instance.
        optimizer (torch.optim.optimizer): Pytorch optimizer instance.
        iterations (int): Number of current iterations.

    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)

def _get_waveform(output, mode="sampling"):
    """Get Waveforom from Samples

    Arguments:
        output {Tensor} -- output tensor(BxTXC)
        mode {str} -- "sampling" or "argmax"
    """
    if mode=="sampling":
        posterior = torch.softmax(output, dim=2)
        dist = torch.distributions.Categorical(posterior)
        wave = dist.sample()
    elif mode=="argmax":
        wave = output.argmax(dim=2)

    return wave # B X T


def main():
    """RUN TRAINING."""
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--datadir", required=True,
                        type=str, help="directory or list of filename of input data")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")

    # network structure setting
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_aux", default=80,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--n_resch", default=512,
                        type=int, help="number of channels of residual output")
    parser.add_argument("--n_skipch", default=256,
                        type=int, help="number of channels of skip output")
    parser.add_argument("--dilation_depth", default=10,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_repeat", default=1,
                        type=int, help="number of repeating of dilation")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--upsampling_factor", default=512,
                        type=int, help="upsampling factor of aux features")
    parser.add_argument("--use_speaker_code", default=False,
                        type=strtobool, help="flag to use speaker code")

    # network training setting
    parser.add_argument("--use_gpu", default=False,
                        type=strtobool, help="using gpu")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_length", default=20000,
                        type=int, help="batch length (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="batch size (if use utterance batch, batch_size will be 1.")
    parser.add_argument("--iters", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--sr", default=16000, 
                        type=int, help="sampling rate")
    parser.add_argument("--checkpoint_interval", default=10000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--log_interval", default=100,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None, nargs="?",
                        type=str, help="model path to restart training")
    parser.add_argument("--mode", default="sampling",
                        type=str, help="decode mode(sampling or argmax)")
                        
    args = parser.parse_args()

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # make tensorboard writer
    tensorboard = TensorboardWriter(log_dir=os.path.join(args.expdir, 'tensorboard'))

    # # show arguments
    # for key, value in vars(args).items():
    #     logging.info("%s = %s" % (key, str(value)))

    # save args as config.json
    with open(os.path.join(args.expdir,'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
        f.close()

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # fix slow computation of dilated conv
    # https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
    torch.backends.cudnn.benchmark = True

    # define network
    
    model = WaveNet(
        n_quantize=args.n_quantize,
        n_aux=args.n_aux,
        n_resch=args.n_resch,
        n_skipch=args.n_skipch,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        kernel_size=args.kernel_size,
        upsampling_factor=args.upsampling_factor)
    logging.info(model)
    model.apply(initialize)
    model.train()

    if args.n_gpus > 1:
        device_ids = range(args.n_gpus)
        model = torch.nn.DataParallel(model, device_ids)
        model.receptive_field = model.module.receptive_field
        if args.n_gpus > args.batch_size:
            logging.warning("batch size is less than number of gpus.")

    # define optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define generator
    generator = batch_generator(
        args.datadir,
        receptive_field=model.receptive_field,
        batch_length=args.batch_length,
        batch_size=args.batch_size,
        shuffle=True,
        upsampling_factor=args.upsampling_factor,
        use_speaker_code=args.use_speaker_code,
        use_gpu=args.use_gpu
    )

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)

    # resume model and optimizer
    if args.resume is not None and len(args.resume) != 0:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        iterations = checkpoint["iterations"]
        if args.n_gpus > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("restored from %d-iter checkpoint." % iterations)
    else:
        iterations = 0

    # check gpu and then send to gpu
    if args.use_gpu:
        model.cuda()
        criterion.cuda()
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
    else:
        logging.warning("cpu is used for training. please check the setting.")

    # train
    loss = 0
    total = 0
    initial_time = time.time()
    logging.info('***** Training Begins at {} *****'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    logging.info('***** Total Interations = {} *****'.format(args.iters - iterations))
    for i in range(iterations, args.iters):
        start = time.time()
        (batch_x, batch_h), batch_t, _ = generator.next()
        batch_output = model(batch_x, batch_h)
        batch_loss = criterion(
            batch_output[:, model.receptive_field:].contiguous().view(-1, args.n_quantize),
            batch_t[:, model.receptive_field:].contiguous().view(-1))
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        total += time.time() - start
        logging.debug("batch loss = %.3f (%.3f sec / batch)" % (
            batch_loss.item(), time.time() - start))

        # report progress
        if (i + 1) % args.log_interval == 0:
            logging.info("(iter:%d) average loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.log_interval, total / args.log_interval))
            logging.info("estimated required time = "
                         "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                         .format(relativedelta(
                             seconds=int((args.iters - (i + 1)) * (total / args.log_interval)))))
            
            # write tensorboard
            tensorboard.write_loss(i+1, 'Loss', 'train', {'loss': loss / args.log_interval})
            train_output = batch_output[:,model.receptive_field:] # B x T x C
            train_label = batch_t[:,model.receptive_field:] # B X T 

            train_output = _get_waveform(train_output, mode=args.mode)
            for b in range(train_output.shape[0]):
                y_train = inv_mulaw_quantize(train_output[b])
                y_label = inv_mulaw_quantize(train_label[b])
                tensorboard.write_audio(i+1, f'iter{i+1}', f'train_{b}', y_train.cpu(), args.sr)
                tensorboard.write_audio(i+1, f'iter{i+1}', f'label_{b}', y_label.cpu(), args.sr)
            
            loss = 0
            total = 0

        # save intermidiate model
        if (i + 1) % args.checkpoint_interval == 0:
            if args.n_gpus > 1:
                save_checkpoint(args.expdir, model.module, optimizer, i + 1)
            else:
                save_checkpoint(args.expdir, model, optimizer, i + 1)

    # close tensorboard
    tensorboard.close()

    # save final model
    if args.n_gpus > 1:
        torch.save({"model": model.module.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    else:
        torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")
    logging.info("total training time ({0} iteration) = "
                         "{1.days:02}:{1.hours:02}:{1.minutes:02}:{1.seconds:02}"
                         .format((args.iters - iterations),relativedelta(
                             seconds=int(time.time() - initial_time))))
    
if __name__ == "__main__":
    main()