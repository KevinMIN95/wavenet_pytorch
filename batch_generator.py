import argparse
import os
import sys
import time
import numpy as np
import logging


from background_generator import background
import torch
import torch.nn.functional as F

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
                    use_speaker_code=False,
                    use_gpu=False,
                    ):
    """GENERATE TRAINING BATCH.

    Args:
        data_dir: directory or list of filename of input data
        receptive_field (int): Size of receptive filed.
        batch_length (int): Batch length (if set None, utterance batch will be used.).
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        shuffle (bool): Whether to shuffle the file list.
        upsampling_factor (int): Upsampling factor.
        use_speaker_code (bool): Whether to use speaker code.
        use_gpu (bool): Whether to use gpu

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

                        if use_gpu:
                            batch_x = batch_x.cuda()
                            batch_h = batch_h.cuda()
                            batch_t = batch_t.cuda()

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

                if use_gpu:
                    batch_x = batch_x.cuda()
                    batch_h = batch_h.cuda()
                    batch_t = batch_t.cuda()

                yield (batch_x, batch_h), batch_t, wavfile

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [data_list[i].strip().split('|')[0] for i in idx]
            feat_list = [data_list[i].strip().split('|')[1] for i in idx]



@background(max_prefetch=16)
def decode_generator(data_dir,
                    batch_size=1,
                    n_quantize=256,
                    upsampling_factor=512,
                    use_speaker_code=False,
                    use_gpu=False,
                    ):
    """GENERATE TRAINING BATCH.

    Args:
        data_dir: Directory or list of filename of input data
        batch_size (int): Batch size (if batch_length = None, batch_size will be 1.).
        n_quantize: Number of quantization.
        upsampling_factor (int): Upsampling factor.
        use_speaker_code (bool): Whether to use speaker code.
        use_gpu (bool): Whether to use gpu

    Returns:
        generator: Generator instance.
    """

    with open(data_dir, 'r') as f:
        data_list = f.readlines()
        f.close()

    logging.info("Number of test data : %d" %(len(data_list)))
    
    feat_list = [data.strip().split('|')[1] for data in data_list]

    if batch_size == 1:
        for featfile in feat_list:
            x = np.zeros((n_quantize//2))
            h = np.load(featfile) # T x C

            # convert to torch variable
            x = torch.from_numpy(x).long()
            h = torch.from_numpy(h).float()

            x = x.unsqueeze(0)  # 1 => 1 x 1
            h = h.transpose(0, 1).unsqueeze(0)  # T x C => 1 x C x T

            if use_gpu:
                # send to cuda
                x = x.cuda()
                h = h.cuda()
            
            n_samples = h.size(2) * upsampling_factor - 1
            feat_name = os.path.basename(featfile).replace("-feats.npy", "")

            yield feat_name, (x, h, n_samples)
    else:
        feats = [ np.load(featfile) for featfile in feat_list ]
        feats = sorted(feats, key=lambda feat : len(feat))

        # divide into batch list
        n_batch = math.ceil(len(feats) / batch_size)
        batch_lists = np.array_split(feat_list, n_batch)
        batch_lists = [f.tolist() for f in batch_lists]

        for batch_list in batch_lists:
            batch_x = []
            batch_h_ = []
            n_samples_list = []
            feat_names = []

            maxlen = maxlen = max([batch.shape[0] for batch in batch_list])

            for featfile in batch_list:
                # make seed waveform and load aux feature
                x = np.zeros((1))
                h = np.array(featfile) # TxC

                # convert to torch variable
                # x = torch.from_numpy(x).long()
                # h = torch.from_numpy(h).float()

                # append to list
                batch_x += [x] 
                batch_h_ += [h] # TxC
                n_samples_list += [len(h) * upsampling_factor - 1]
                feat_names += [os.path.basename(featfile).replace("-feats.npy", "")]

            batch_x = np.stack(batch_x, axis=0) # B x 1

            batch_size = len(batch_list)
            n_feats = batch_h_[0].shape[-1]
            size = (batch_size, maxlen, n_feats)
            batch_h = np.full(size, n_quantize//2)
            for b in range(batch_size):
                t = batch_h_[b].shape[0]
                batch_h[b,:t] = batch_h_[b]
            
            # convert to torch variable
            batch_x = torch.from_numpy(batch_x).long() # B x 1
            batch_h = torch.from_numpy(batch_h).float().transpose(1, 2) # B x T_max

            if use_gpu:
                # send to cuda
                x = x.cuda()
                h = h.cuda()

            yield feat_names, (batch_x, batch_h, n_samples_list)


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