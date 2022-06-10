import numpy as np

import librosa
from scipy import signal

from hparams import hparams as hps
# Conversions
_mel_basis = None
_inv_mel_basis = None


def start_and_end_indices(quantized, silence_threshold=2):
    start = max(filter(lambda s: abs(quantized[s] - 127) < silence_threshold + 1, range(quantized.size)))
    end = min(filter(lambda e: abs(quantized[e] - 127) < silence_threshold + 1, range(quantized.size - 1, 1, -1)))

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def mulaw_quantize(x, mu=256) -> np.ndarray:
    """ Mu-Law companding + quantize
    Mu-Law companding
    .. math::
        f(x) = sign(x) ln (1 + mu |x|) / ln (1 + mu)

    Args:
        x (array-like): Input signal. Each value of input signal must be in range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    """
    mu = mu-1
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # scale [-1, 1] to [0, mu]
    return ((y + 1) / 2 * mu).astype(np.int)


def _normalize(S):
    if hps.allow_clipping_in_normalization:
        if hps.symmetric_mels:
            return np.clip((2 * hps.max_abs_value) * (
                    (S - hps.min_level_db) / (-hps.min_level_db)) - hps.max_abs_value, -hps.max_abs_value, hps.max_abs_value)
        else:
            return np.clip(hps.max_abs_value * ((S - hps.min_level_db) / (-hps.min_level_db)), 0, hps.max_abs_value)

    assert S.max() <= 0 and S.min() - hps.min_level_db >= 0
    if hps.symmetric_mels:
        return (2 * hps.max_abs_value) * ((S - hps.min_level_db) / (-hps.min_level_db)) - hps.max_abs_value
    else:
        return hps.max_abs_value * ((S - hps.min_level_db) / (-hps.min_level_db))

def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    # num frame
    pad = (fsize - fshift)
    if len(x) % fshift == 0:
        M = (len(x) + pad * 2 - fsize) // fshift + 1
    else:
        M = (len(x) + pad * 2 - fsize) // fshift + 2

    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r

def get_hop_size():
    hop_size = hps.hop_size
    if hop_size is None:
        assert hps.frame_shift_ms is not None
        hop_size = int(hps.frame_shift_ms / 1000 * hps.sample_rate)
    return hop_size

def melspectrogram(wav):
    # preemphasis
    y = signal.lfilter([1, -hps.preemphasis], [1], wav) if hps.preemphasize else wav
    # ltft
    if hps.use_lws:
        import lws
        D = lws.lws(hps.fft_size, get_hop_size(), fftsize=hps.win_size, mode="speech").stft(y).T
    else:
        D = librosa.stft(y=y, n_fft=hps.fft_size, hop_length=get_hop_size(), win_length=hps.win_size)
    # linear_to_mel
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(hps.sample_rate, hps.fft_size, n_mels=hps.num_mels)  # fmin=0, fmax= sample_rate/2.0
    x = np.dot(_mel_basis, np.abs(D))
    # emp to db
    min_level = np.exp(hps.min_level_db / 20 * np.log(10))  # min_level_db = -100
    S = 20 * np.log10(np.maximum(min_level, x)) - hps.ref_level_db

    if hps.signal_normalization:
        return _normalize(S)
    else:
        return S

def linearspectrogram(wav):
    # preemphasis
    y = signal.lfilter([1, -hps.preemphasis], [1], wav) if hps.preemphasize else wav
    # ltft
    if hps.use_lws:
        import lws
        D = lws.lws(hps.fft_size, get_hop_size(), fftsize=hps.win_size, mode="speech").stft(y).T
    else:
        D = librosa.stft(y=y, n_fft=hps.fft_size, hop_length=get_hop_size(), win_length=hps.win_size)
    # emp to db
    min_level = np.exp(hps.min_level_db / 20 * np.log(10))  # min_level_db = -100
    S = 20 * np.log10(np.maximum(min_level, np.abs(D))) - hps.ref_level_db

    if hps.signal_normalization:
        return _normalize(S)
    else:
        return S
