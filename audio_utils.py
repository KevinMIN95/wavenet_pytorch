import numpy as np
import librosa
from scipy import signal

EPS = 1e-10

# if x is positive 1 otherwise -1
def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()

# log1p() : log(1 + x)
def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()

def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()

def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()

def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()

def mulaw(x, mu=256):
    """Mu-Law companding
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)

def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)
    .. math::
        f^{-1}(x) = sign(y) (1 / \mu) (1 + \mu)^{|y|} - 1)
    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    """
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)

def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    """
    y = mulaw(x, mu)

    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)

def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize
    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncompressed signal ([-1, 1])
    """
    # [0, m) to [-1, 1]
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)

def audio_trim(wav, top_db=60, frame_length=2048, hop_length=512):
    audio, _ = librosa.effects.trim(wav, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return audio

def low_cut_filter(x, fs, cutoff=70):
    """APPLY LOW CUT FILTER.
    Args:
        x (ndarray): Waveform sequence.
        fs (int): Sampling frequency.
        cutoff (float): Cutoff frequency of low cut filter.

    Return:
        ndarray: Low cut filtered waveform sequence.
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = signal.firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = signal.lfilter(fil, 1, x)

    return lcf_x

def stft(x, n_fft, hop_length, win_length, pad_mode = "reflect"):
    return librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length,
                        pad_mode=pad_mode)

def logmelspectrogram(y, sampling_rate, n_fft=2048, hop_length=512, n_mels=128, fmin=None, fmax=None, power=1.0):
    mspc = librosa.feature.melspectrogram(
        y, sampling_rate,
        n_fft = n_fft,
        hop_length = hop_length,            
        n_mels=n_mels,
        fmin=fmin if fmin is not None else 0,
        fmax=fmax if fmax is not None else sampling_rate // 2,
        power=power
    )
    mspc = librosa.core.power_to_db(mspc)
    
    # (n_mels, t)
    return mspc