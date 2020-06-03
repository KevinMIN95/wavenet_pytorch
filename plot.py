import matplotlib.pyplot as plt
import numpy as np
from audio_utils import logmelspectrogram
from scipy.io import wavfile
import librosa.display as dis

def showMelspectogram(S, sr, hop_length=512, fmax=None, save_file=None, show=True) :
    """Show MElSPECTOGRAM Figure
    Arguments:
        S {np.array} -- melspectogram
        sr {int} -- sampling rate
    """

    plt.figure(figsize=(10, 4))
    dis.specshow(S, x_axis='time', y_axis='mel', sr=sr, hop_length= hop_length, fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
   


if __name__ == "__main__":
    data_path = 'data/local/wav.scp'    

    with open(data_path, 'r') as f:
        wav_files = f.readlines()
        for wav_file in wav_files:
            src = wav_file.rstrip()

            sr, x= wavfile.read(src)
            if x.dtype == np.int16:
                x = x.astype(np.float32) / np.iinfo(np.int16).max
            mspc = logmelspectrogram(x, sr)

            src_fig = 'mel_fig/'+src.split('/')[-1].replace('.wav','')
            showMelspectogram(mspc, sr, show=False, save_file=src_fig)
        f.close()
    


    

