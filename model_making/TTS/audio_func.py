import numpy as np
import torch


def mulaw_quantize(x, mu=256) -> torch.Tensor:
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
    return torch.from_numpy(((y + 1) / 2 * mu).astype(np.int))
