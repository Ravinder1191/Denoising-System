import numpy as np

def compute_si_sdr(target, estimate):
    """
    Technique: Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) using optimal scaling vectors.
    Reason: Ignores simple volume changes and isolates actual noise reduction, which standard MSE fails to do.
    """
    eps = 1e-8
    target_zero_mean = target - np.mean(target)
    estimate_zero_mean = estimate - np.mean(estimate)
    alpha = np.sum(estimate_zero_mean * target_zero_mean) / (np.sum(target_zero_mean ** 2) + eps)
    target_scaled = alpha * target_zero_mean
    noise = estimate_zero_mean - target_scaled
    val = 10 * np.log10((np.sum(target_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return val
