import argparse
import wave
import numpy as np
from scipy.io import wavfile
from glob import glob
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split as split

# The parameters are prerequisite information. More specifically,
def pcm2wav( pcm_file, channels=1, bit_depth=16, sampling_rate=16000 ):

    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth "+str(bit_depth)+" must be a multiple of 8.")

    wav_file = pcm_file.replace('.pcm', '.wav')
        
    # Read the .pcm file as a binary file and store the data to pcm_data
    with open( pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read()
        obj2write = wave.open( wav_file, 'wb')
        obj2write.setnchannels( channels )
        obj2write.setsampwidth( bit_depth // 8 )
        obj2write.setframerate( sampling_rate )
        obj2write.writeframes( pcm_data )
        obj2write.close()

    return wav_file

def read_wav_or_pcm(src_file, is_pcm):
    if is_pcm:
        src_file = pcm2wav(src_file)
    sr, x = wavfile.read(src_file)
    return sr, x, src_file

def write_wav(dst_path, sr, x):
    wavfile.write(dst_path, sr, x)

def train_test_split(data_dir, local_dir, train_dir, test_dir, test_size = None, random_state = 10):

    src_files = sorted(glob(join(data_dir, "*.wav")))
    pcm_files = sorted(glob(join(data_dir, "*.pcm")))

    is_pcm = len(src_files) == 0 and len(pcm_files) > 0

    if is_pcm:
        print("Assuming 16kHz /16bit audio data")
        src_files = pcm_files

    if len(src_files) == 0:
        raise RuntimeError("No files found in {}".format(data_dir))

    total_samples = 0
    indices = []
    signed_int16_max = 2**15
    wav_files = []

    print("Total number of utterances: {}".format(len(src_files)))

    with open(f'{local_dir}/wav.scp', 'wt') as f:
        for idx, src_file in tqdm(enumerate(src_files)):
            sr, x, src_file= read_wav_or_pcm(src_file, is_pcm)
            if x.dtype == np.int16:
                x = x.astype(np.float32) / signed_int16_max
            total_samples += len(x)
            indices.append(idx)
            wav_files.append(src_file)
            f.write(src_file+'\n')

        f.close()
    
    src_files = wav_files
        
    total_hours = float(total_samples) / sr / 3600.0
    print("Total hours of utterances: {:.3f}".format(total_hours))

    # train, test split

    if test_size is not None :
        train_indices, test_indices = split(indices, test_size = test_size, random_state = random_state)
    else :
        # test size : 0.25 by default
        train_indices, test_indices = split(indices, random_state = random_state)
    
    with open(f'{train_dir}/wav.scp', 'wt') as f:
        train_indices = sorted(train_indices)
        for i in train_indices:
            f.write(src_files[i]+'\n')
        f.close()
    print(f"Making wav list for training is successfully done. (#training = {len(train_indices)} )")

    with open(f'{test_dir}/wav.scp', 'wt') as f:
        test_indices = sorted(test_indices)
        for i in test_indices:
            f.write(src_files[i]+'\n')
        f.close()
    print(f"Making wav list for testing is successfully done. (#testing = {len(test_indices)} )")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="making train & test dataset")

    parser.add_argument("--data_dir", default=None,
        help="directory of data wavfiles")
    
    parser.add_argument("--local_dir", default=None,
        help="directory of local")

    parser.add_argument("--train_dir", default=None,
        help="directory to save list of train data wavfiles")

    parser.add_argument("--test_dir", default=None,
        help="directory to save list of test data wavfiles")

    parser.add_argument("--test_size", default=None, type=float,
        help="size of test data(int or float)")

    parser.add_argument("--random_state", default=10, type=int,
        help="random state of train test split")

    args = parser.parse_args()
    # print(args)
    
    train_test_split(str(args.data_dir), str(args.local_dir), str(args.train_dir), str(args.test_dir), 
        test_size = args.test_size, random_state = args.random_state)


