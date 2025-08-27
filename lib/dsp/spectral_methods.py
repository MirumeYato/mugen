import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.signal import find_peaks
from scipy.linalg import hankel

def estimate_trend_phase(vector, valid_indices=None, spacing=1.0, scale=1.0):
    """
    Estimate a phase-based slope/trend from a complex vector.
    
    Parameters
    ----------
    vector : ndarray (complex)
        Complex-valued input sequence (e.g., feature vector).
    valid_indices : array-like, optional
        Indices to use for fitting. If None, use all.
    spacing : float, optional
        Frequency/feature spacing between indices (default=1.0).
    scale : float, optional
        Scaling factor to map slope into desired units (default=1.0).

    Returns
    -------
    slope_est : float
        Estimated slope/trend value.
    """
    if valid_indices is None:
        valid_indices = np.arange(len(vector))

    phase = np.unwrap(np.angle(vector))
    coeffs = np.polyfit(valid_indices, phase[valid_indices], 1)
    slope = -coeffs[0]

    return slope / (2 * np.pi * spacing) * scale

def estimate_peak_music(vector, window_size=245, num_components=2, 
                        valid_indices=None, spacing=1.0, scale=1.0, 
                        peak_height=0.0):
    """
    Estimate a peak position (e.g., phase slope or delay proxy) 
    using MUSIC and a Hankel covariance approach.

    Parameters
    ----------
    vector : ndarray (complex)
        Input complex-valued feature vector.
    window_size : int, optional
        Window size for Hankel matrix construction (default: 245).
    num_components : int, optional
        Number of dominant sources/components to consider (default: 2).
    valid_indices : array-like, optional
        Indices to select from the input vector (default: all).
    spacing : float, optional
        Frequency/feature spacing for scaling the estimate (default: 1.0).
    scale : float, optional
        Additional scaling factor applied to the estimate (default: 1.0).
    peak_height : float, optional
        Minimum pseudospectrum peak height for detection (default: 0.0).

    Returns
    -------
    est_value : float
        Estimated and scaled peak location.
    """
    if valid_indices is None:
        valid_indices = np.arange(len(vector))

    y = vector[valid_indices]
    c = y[:window_size]
    r = y[window_size-1:]
    Y = hankel(c, r)
    cov_matrix = (Y @ Y.conj().T) / Y.shape[1]

    # Use MUSIC to compute pseudospectrum
    S, omega = estimate_pseudospectrum_music(
        cov_matrix, num_components, freq_range='whole'
    )

    # Peak picking
    pks, _ = find_peaks(S, height=peak_height)
    if len(pks) == 0:
        return None  # no valid peak

    # Select the strongest among top components
    sorted_idx = np.argsort(-S[pks])[:num_components]
    phi_est = -omega[pks[np.max(sorted_idx)]]
    phi_est_mod = np.mod(phi_est, 2 * np.pi)

    # Scale to desired units
    est_value = (phi_est_mod / (2 * np.pi * spacing)) * scale
    return est_value

def estimate_pseudospectrum_music(cov_matrix, num_components, 
                                  nfft=256, fs=None, freq_range='half',
                                  threshold=None, 
                                  return_outputs=True, 
                                  return_eigenvectors=False, 
                                  return_eigenvalues=False,
                                  plot=False):
    """
    Estimate a pseudospectrum using the MUSIC (Multiple Signal Classification) method.
    
    Parameters
    ----------
    cov_matrix : ndarray (2D, Hermitian)
        Covariance or correlation matrix of the data.
    num_components : int
        Assumed number of dominant components (signal subspace size).
    nfft : int, optional
        FFT length (default: 256).
    fs : float, optional
        Sampling frequency (if provided, frequencies are in Hz).
    freq_range : {'half', 'whole', 'centered'}, optional
        Range of the frequency axis (default: 'half').
    threshold : float, optional
        Relative threshold to estimate signal subspace dimension adaptively.
    return_outputs : bool, optional
        If True, return values. If False, only plot.
    return_eigenvectors : bool, optional
        If True, also return the noise subspace eigenvectors.
    return_eigenvalues : bool, optional
        If True, also return the eigenvalues.
    plot : bool, optional
        If True, plot the pseudospectrum.
    
    Returns
    -------
    outputs : tuple
        Always (Sxx, w). Optionally adds (noise_vectors, eigenvalues).
    """
    # Ensure Hermitian
    R = (cov_matrix + cov_matrix.T.conj()) / 2
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Determine signal/noise subspaces
    if threshold is not None:
        signal_dim = np.sum(eigenvalues > threshold * eigenvalues[-1])
        signal_dim = min(signal_dim, num_components)
    else:
        signal_dim = num_components
    
    noise_vectors = eigenvectors[:, signal_dim:]
    
    # Compute pseudospectrum
    w, Sxx = _compute_music_pseudospectrum(noise_vectors, nfft, fs, freq_range)
    
    # Plot if requested
    if plot:
        plt.plot(w, 10 * np.log10(Sxx))
        plt.title("MUSIC Pseudospectrum")
        plt.xlabel("Frequency (rad/sample)" if fs is None else "Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.grid(True)
        plt.show()
    
    if not return_outputs:
        return
    
    outputs = [Sxx, w]
    if return_eigenvectors:
        outputs.append(noise_vectors)
    if return_eigenvalues:
        outputs.append(eigenvalues)
    return tuple(outputs)


def _compute_music_pseudospectrum(noise_vectors, nfft, fs, freq_range):
    """
    Internal helper: compute MUSIC pseudospectrum from noise subspace eigenvectors.
    """
    # Compute denominator from projection of steering vectors
    w, h = freqz(noise_vectors[:, 0], worN=nfft, whole=True)
    den = np.abs(h) ** 2
    for i in range(1, noise_vectors.shape[1]):
        _, h = freqz(noise_vectors[:, i], worN=nfft, whole=True)
        den += np.abs(h) ** 2
    
    Sxx = 1 / den
    
    # Adjust frequency axis
    if freq_range == 'half':
        select = slice(0, nfft//2 + 1) if nfft % 2 == 0 else slice(0, (nfft+1)//2)
        w, Sxx = w[select], Sxx[select]
    elif freq_range == 'centered':
        w, Sxx = np.fft.fftshift(w), np.fft.fftshift(Sxx)
    
    # Convert to Hz if fs provided
    if fs is not None:
        w = w * fs / (2 * np.pi)
    
    return w, Sxx
