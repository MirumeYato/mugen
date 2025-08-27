import tensorflow as tf

import numpy as np
from scipy.linalg import hankel
# from sklearn.utils import shuffle

def reduplicate(arr):
    return np.concatenate([arr,arr], axis=0)

def to_mag_phase(array_2x256):
    """
    Convert a single (2, 256) array [real, imag] → [mag, phase]
    """
    re = array_2x256[0]
    im = array_2x256[1]
    mag = np.sqrt(re**2 + im**2)
    phase = np.arctan2(im, re)
    return np.stack([mag, phase], axis=0)  # shape (2, 256)

def preprocess_to_Ry_MEM_DIF(X, Ncov=64):
    """
    Converts input data X of shape (N, 2, 256) to Ry matrices of shape (N, Ncov, Ncov, 2)
    
    Args:
        X: numpy array (N, 2, 256), real & imaginary parts separately.
        Ncov: covariance matrix dimension.
        
    Returns:
        Ry_data: numpy array (N, Ncov, Ncov, 2), real and imaginary channels.
    """
    N = X.shape[0]
    Ry_data = np.zeros((N, Ncov, Ncov, 2), dtype=np.float32)
    validScIndex = np.arange(6, 251)
    
    for i in range(N):
        y_complex = X[i, 0, :] + 1j * X[i, 1, :]  # reconstruct complex y (256,)
        y_complex = y_complex[validScIndex]
        
        c = y_complex[:Ncov]
        r = y_complex[Ncov - 1:]
        Y = hankel(c, r)
        
        Ry = (Y @ Y.conj().T) / Y.shape[1]  # (Ncov, Ncov)
        
        Ry_data[i, :, :, 0] = Ry.real
        Ry_data[i, :, :, 1] = Ry.imag
        
    return Ry_data

def Ry_from_y(y, Ncov=64):
    c = y[:Ncov]
    r = y[Ncov - 1:]
    Y = hankel(c, r)
    Ry = tf.math.multiply(Y, tf.linalg.matrix_transpose(Y, conjugate=True))/ Y.shape[1]
    return tf.math.real(Ry), tf.math.imag(Ry) 

def generator(X, y, Ncov):
    for i in range(len(X)):
        validScIndex = np.arange(6, 251)
        y_complex = X[i, 0, :] + 1j * X[i, 1, :]
        y_complex = y_complex[validScIndex]
        Ry = Ry_from_y(y_complex, Ncov)
        yield Ry, y[i]
    
def augment(features, targets, p):
    """
    Perform data augmentation by injecting Cauchy noise into training features.

    This function generates additional training samples by adding Cauchy noise
    to each sample in the input feature set `features`. Each original sample is 
    augmented `num_points` times, resulting in an expanded dataset. The corresponding 
    target values from `targets` are duplicated accordingly.

    :param features: Input feature array of shape (n_samples, ...).
    :param targets: Corresponding target values of shape (n_samples,).
    :param p : 
        - num_points: Number of noisy copies to generate per sample (default: 20).
        - mu: Location parameter of the Cauchy distribution (default: 0).
        - sigma: Scale parameter of the Cauchy distribution (default: 0.02).

    :return: Tuple (aug_features, aug_targets), the augmented feature and target arrays.
    """
    if p is not None:
        print("DEBUG: Augmentation is used")
        t_exp = np.repeat(targets, p["num_points"][0], axis=0)
        noise_tar = np.random.standard_cauchy(t_exp.shape) * p["sigma"][0] + p["mu"][0]
        f_exp = np.repeat(features, p["num_points"][1], axis=0)
        noise_feat = np.random.standard_cauchy(f_exp.shape) * p["sigma"][1] + p["mu"][1]
        augmented_targets = t_exp + noise_tar
        augmented_features =  f_exp + noise_feat
        return augmented_features, augmented_targets
    else:
        print("DEBUG: No augmentation. Just shuffling")
        return features, targets