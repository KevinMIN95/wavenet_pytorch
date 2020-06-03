import argparse
import sys
import time
import numpy as np
import logging

from background_generator import background
import torch

def validate_length(x, y, upsampling_factor=None):
    """VALIDATE LENGTH.

    Args:
        x (ndarray): ndarray with x.shape[0] = len_x.
        y (ndarray): ndarray with y.shape[0] = len_y.
        upsampling_factor (int): Upsampling factor.

    Returns:
        ndarray: Length adjusted x with same length y.
        ndarray: Length adjusted y with same length x.

    """
    if upsampling_factor is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:y.shape[0] * upsampling_factor]
        if x.shape[0] < y.shape[0] * upsampling_factor:
            mod_y = y.shape[0] * upsampling_factor - x.shape[0]
            mod_y_frame = mod_y // upsampling_factor + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * upsampling_factor]
        assert len(x) == len(y) * upsampling_factor

    return x, y


@background(max_prefetch=16)
def batch_generator(data_dir, receptive_field,
                    batch_length=None,
                    batch_size=1,
                    shuffle=True,
                    upsampling_factor=512,
                    use_speaker_code=False):
    """GENERATE TRAINING BATCH.

    Args:
        data_dir: directory or list of filename of input data
        receptive_field (int): Size of receptive filed.
        batch_length (int): Batch length (if set None, utterance batch will be used.).
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        shuffle (bool): Whether to shuffle the file list.
        upsampling_factor (int): Upsampling factor.
        use_speaker_code (bool): Whether to use speaker code.

    Returns:
        generator: Generator instance.

    """
    with open(data_dir, 'r') as f:
        data_list = f.readlines()
        f.close()

    logging.info("Number of data : %d" %(len(data_list)))
    
    # shuffle list
    if shuffle:
        n_files = len(data_list)
        idx = np.random.permutation(n_files)
        wav_list = [data_list[i].strip().split('|')[0] for i in idx]
        feat_list = [data_list[i].strip().split('|')[1] for i in idx]

    # check batch_length
    if batch_length is not None:
        batch_mod = (receptive_field + batch_length) % upsampling_factor
        logging.warning("batch length is decreased due to upsampling (%d -> %d)" % (
            batch_length, batch_length - batch_mod))
        batch_length -= batch_mod

    # show warning
    if batch_length is None and batch_size > 1:
        batch_size = 1 
        logging.warning("in utterance batch mode, batchsize will be 1.")

    while True:
        batch_x, batch_h, batch_t = [], [], []
        # process over all of files
        for wavfile, featfile in zip(wav_list, feat_list):
            sys.stdout.flush()
            # load waveform and aux feature
            x = np.load(wavfile)
            h = np.load(featfile)

            # if use_speaker_code:
            #     sc = read_hdf5(featfile, "/speaker_code")
            #     sc = np.tile(sc, [h.shape[0], 1])
            #     h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            
            x, h = validate_length(x, h, upsampling_factor)

            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # ------------------------------------
            # use mini batch with upsampling layer
            # ------------------------------------
            if batch_length is not None:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(h_buffer) > (receptive_field + batch_length) // upsampling_factor:
                    # set batch size
                    h_bs = (receptive_field + batch_length) // upsampling_factor
                    x_bs = h_bs * upsampling_factor + 1

                    # get pieces
                    h_ = h_buffer[:h_bs]
                    x_ = x_buffer[:x_bs]

                    # convert to torch variable
                    x_ = torch.from_numpy(x_).long()
                    h_ = torch.from_numpy(h_).float()

                    # remove the last and first sample for training
                    batch_h += [h_.transpose(0, 1)]  # (D x T)
                    batch_x += [x_[:-1]]  # (T)
                    batch_t += [x_[1:]]  # (T)

                    # set shift size
                    h_ss = batch_length // upsampling_factor
                    x_ss = h_ss * upsampling_factor

                    # update buffer
                    h_buffer = h_buffer[h_ss:]
                    x_buffer = x_buffer[x_ss:]

                    # return mini batch
                    if len(batch_x) == batch_size:
                        batch_x = torch.stack(batch_x)
                        batch_h = torch.stack(batch_h)
                        batch_t = torch.stack(batch_t)

                        # TODO: gpu : send batches to cuda

                        yield (batch_x, batch_h), batch_t, wavfile

                        batch_x, batch_h, batch_t = [], [], []

            # -----------------------------------------
            # use utterance batch with upsampling layer
            # -----------------------------------------
            else:
                # remove last frame
                h = h[:-1]
                x = x[:-upsampling_factor + 1]

                # convert to torch variable
                x = torch.from_numpy(x).long()
                h = torch.from_numpy(h).float()

                # remove the last and first sample for training
                batch_h = h.transpose(0, 1).unsqueeze(0)  # (1 x D x T')
                batch_x = x[:-1].unsqueeze(0)  # (1 x T)
                batch_t = x[1:].unsqueeze(0)  # (1 x T)

                # TODO: gpu : send batches to cuda
                yield (batch_x, batch_h), batch_t, wavfile

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [data_list[i].strip().split('|')[0] for i in idx]
            feat_list = [data_list[i].strip().split('|')[1] for i in idx]



if __name__ == "__main__":
    data_dir = 'data/test/wav-pre.scp'

    generator = batch_generator(
        data_dir,
        receptive_field=2,
        batch_length=10000,
        batch_size=1,
        shuffle=True,
        upsampling_factor=512,
        use_speaker_code=False)

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)

    # print(generator.queue)
    
    s =set()
    for iter in range(3000):
        (batch_x, batch_h), batch_t, wavfile = generator.next()
        s.add(wavfile)
        print(iter)
    
    print(len(s))