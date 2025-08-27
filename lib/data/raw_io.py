import os
import numpy as np
import scipy.io as scio
import re
from tqdm import trange

from lib import PATH
DATA_DIR = os.path.join(PATH, "data")

from lib.dsp.spectral_methods import estimate_trend_phase as slope, estimate_peak_music as music

# --- Data loading and feature extraction ---

def load_data(filename, name_pattern = None):
    """
    Loads the .mat file, extracts complex vector of features and 
    computes target estimates by the SLOPE and MUSIC methods.
    The target is inferred from the filename.
    """
    data = scio.loadmat(filename)
    # Define all possible name prefixes
    name_pattern = r'(Data|LoS|NLoS)'  # group of valid names
    
    # Regex: match name + digits (or "p" for point) + "m.mat"
    match = re.search(name_pattern + r'([\d\.p]+)m\.mat', filename)
    
    if match:
        target_str = match.group(2).replace('p', '.')
        gt_target = float(target_str)
    else:
        gt_target = None

    features = []
    targets = []
    slope_estimates = []
    music_estimates = []
    Nftm = data['data'].shape[0]
    Ncov = 150  # MUSIC covariance matrix size
    rtt = []
    time_stamp_1 = []
    time_stamp_2 = []
    time_stamp_3 = []
    time_stamp_4 = []

    for i in range(Nftm):
        t1 = data['data'][i][0][0].item()
        t2 = data['data'][i][1][0].item()
        t3 = data['data'][i][2][0].item()
        t4 = data['data'][i][3][0].item()

        feature_vectorT2 = data['data'][i][4][0]
        feature_vectorT4 = data['data'][i][5][0]

        # Here we extract features using the imaginary part as phase and the real part as magnitude.
        phase_T2 = np.imag(feature_vectorT2)
        mag_T2 = np.real(feature_vectorT2)
        phase_T4 = np.imag(feature_vectorT4)
        mag_T4 = np.real(feature_vectorT4)
        feat = np.stack([phase_T2, mag_T2, phase_T4, mag_T4], axis=0)  # (4, N)
        
        rtt_i = ((t4 - t1) - (t3 - t2)) / 2 * 0.3 # rtt/2 without counting effect from device delta_t [m]. rtt_i + 0.3 * summed_delta_t/2 = real_half_rtt [m] = target between devices

        # SLOPE method
        delta_tT2_Slope = slope(feature_vectorT2)
        delta_tT4_Slope = slope(feature_vectorT4)
        t2_hat_slope = t2 + delta_tT2_Slope
        t4_hat_slope = t4 + delta_tT4_Slope
        target_slope = ((t4_hat_slope - t1) - (t3 - t2_hat_slope)) / 2 * 0.3 # = rtt_i + 0.3 * sumed_delta_t_fromSlope/2

        # MUSIC method
        try:
            delta_tT2_Music = music(feature_vectorT2, Ncov)
            delta_tT4_Music = music(feature_vectorT4, Ncov)
        except Exception as error:
            print(f"WARNING: raw_io.py:75 in music.\nError message: {error}")
            print(f"Data for this point will be ignored\n")
            continue
        t2_hat_music = t2 + delta_tT2_Music
        t4_hat_music = t4 + delta_tT4_Music
        target_music = ((t4_hat_music - t1) - (t3 - t2_hat_music)) / 2 * 0.3 # = rtt_i + 0.3 * sumed_delta_t_fromMUSIC/2

        features.append(feat)
        slope_estimates.append(target_slope)
        music_estimates.append(target_music)
        targets.append(gt_target)
        rtt.append(rtt_i)
        time_stamp_1.append(t1)
        time_stamp_2.append(t2)
        time_stamp_3.append(t3)
        time_stamp_4.append(t4)

    features = np.array(features)
    targets = np.array(targets)
    slope_estimates = np.array(slope_estimates)
    music_estimates = np.array(music_estimates)
    rtt = np.array(rtt)
    return features, targets, slope_estimates, music_estimates, rtt, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4

from joblib import Parallel, delayed
from tqdm import trange
import numpy as np

class SynthDataset:
    def __init__(self, 
                 fpath, 
                 max_samples_num=float("inf"), 
                 channel_len=None, 
                 delta=0, 
                 dtype=np.float32):
        self.channel_len = channel_len
        self.fpath = fpath
        self.max_samples_num = max_samples_num
        self.samples_num = 0
        self.left_guard_pos = 0
        self.right_guard_pos = 255
        self.delta = delta
        lola = self.left_guard_pos + self.right_guard_pos
        self.left_mid_pos = lola // 2 - delta
        self.right_mid_pos = lola // 2 + 1 + delta
        self.use_channel_num = self.right_guard_pos - self.left_guard_pos - 2 * delta
        self.dtype = dtype

    def chunk_generator(self, chunk_size=10000):
        print(f"DEBUG: Loading .mat file header from {self.fpath} ...")
        raw_feature_vector_data = scio.loadmat(self.fpath)
        total_samples = min(raw_feature_vector_data["data"].shape[0], self.max_samples_num)
        print(f"DEBUG: Total samples: {total_samples}")

        feature_vector_data = raw_feature_vector_data["data"]

        feature_vector_target = np.concatenate([
            np.arange(self.left_guard_pos, self.left_mid_pos + 1),
            np.arange(self.right_mid_pos, self.right_guard_pos + 1)
        ])
        n_channels = len(feature_vector_target)

        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            n_samples = chunk_end - chunk_start
            print(f"DEBUG: chunk_end: {chunk_end}, n_samples: {n_samples}")

            # Preallocate output arrays
            features_chunk = np.empty((n_samples, n_channels, 2), dtype=self.dtype)
            cach_chunk_arr = np.empty((n_samples, n_channels), dtype=np.complex64)
            targets_chunk = np.empty(n_samples, dtype=np.float64)

            # Efficient extraction with minimal Python looping
            for i in range(n_samples):
                idx = chunk_start + i
                feature_vector_vector = feature_vector_data[idx][0]  # shape (257, 1)
                feature_vector_channels = feature_vector_vector[0]   # [feature_vector_vector[0][j] for j in feature_vector_target]
                cach_chunk_arr[i, :] = feature_vector_channels
                features_chunk[i, :, 0] = np.imag(feature_vector_channels)
                features_chunk[i, :, 1] = np.real(feature_vector_channels)
                targets_chunk[i] = feature_vector_data[idx][1].item()*10**9

            # Now parallel estimation
            music_chunk = np.empty(n_samples, dtype=np.float32)
            slope_chunk = np.empty(n_samples, dtype=np.float32)

            def safe_music(x):
                try:
                    return music(x)-87.5
                except Exception as e:
                    # We use nan for broken inputs
                    return np.nan

            def safe_slope(x):
                try:
                    return slope(x)-87.5
                except Exception as e:
                    return np.nan
                
            def process_both(feature_vector):
                return safe_music(feature_vector), safe_slope(feature_vector)
            
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_both)(feature_vector) for feature_vector in cach_chunk_arr
            )
            music_results, slope_results = zip(*results)

            music_chunk[:] = music_results
            slope_chunk[:] = slope_results

            yield features_chunk, targets_chunk, music_chunk, slope_chunk

class SynthExtDataset:
    def __init__(self, 
                 fpath, 
                 max_samples_num=float("inf"), 
                 channel_len=None, 
                 delta=0, 
                 dtype=np.float32):
        self.channel_len = channel_len
        self.fpath = fpath
        self.max_samples_num = max_samples_num
        self.samples_num = 0
        self.left_guard_pos = 0
        self.right_guard_pos = 255
        self.delta = delta
        lola = self.left_guard_pos + self.right_guard_pos
        self.left_mid_pos = lola // 2 - delta
        self.right_mid_pos = lola // 2 + 1 + delta
        self.use_channel_num = self.right_guard_pos - self.left_guard_pos - 2 * delta
        self.dtype = dtype

    def chunk_generator(self, chunk_size=10000):
        print(f"DEBUG: Loading .mat file header from {self.fpath} ...")
        raw_feature_vector_data = scio.loadmat(self.fpath)
        total_samples = min(raw_feature_vector_data["data"].shape[0], self.max_samples_num)
        print(f"DEBUG: Total samples: {total_samples}")

        feature_vector_data = raw_feature_vector_data["data"]

        feature_vector_target = np.concatenate([
            np.arange(self.left_guard_pos, self.left_mid_pos + 1),
            np.arange(self.right_mid_pos, self.right_guard_pos + 1)
        ])
        n_channels = len(feature_vector_target)

        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            n_samples = chunk_end - chunk_start
            print(f"DEBUG: chunk_end: {chunk_end}, n_samples: {n_samples}")

            # Preallocate output arrays
            features2_chunk = np.empty((n_samples, n_channels, 2), dtype=self.dtype)
            features4_chunk = np.empty((n_samples, n_channels, 2), dtype=self.dtype)
            cach_chunk_arr2 = np.empty((n_samples, n_channels), dtype=np.complex64)
            cach_chunk_arr4 = np.empty((n_samples, n_channels), dtype=np.complex64)
            targets_chunk = np.empty(n_samples, dtype=np.float64)
            rtt_chunk = np.empty(n_samples, dtype=np.float64)

            # Efficient extraction with minimal Python looping
            for i in range(n_samples):
                idx = chunk_start + i
                t1 = feature_vector_data[idx][0][0].item()
                t2 = feature_vector_data[idx][1][0].item()
                t3 = feature_vector_data[idx][2][0].item()
                t4 = feature_vector_data[idx][3][0].item()
                feature_vector_vector2 = feature_vector_data[idx][4] # shape (257, 1)
                feature_vector_vector4 = feature_vector_data[idx][5] 
                # feature_vector_channels = feature_vector_vector[0]#[feature_vector_vector[0][j] for j in feature_vector_target]
                cach_chunk_arr2[i, :] = feature_vector_vector2[0]
                cach_chunk_arr4[i, :] = feature_vector_vector4[0]
                features2_chunk[i, :, 0] = np.imag(feature_vector_vector2[0])
                features2_chunk[i, :, 1] = np.real(feature_vector_vector2[0])
                features4_chunk[i, :, 0] = np.imag(feature_vector_vector4[0])
                features4_chunk[i, :, 1] = np.real(feature_vector_vector4[0])
                targets_chunk[i] = feature_vector_data[idx][6].item()
                rtt_chunk[i] = ((t4 - t1) - (t3 - t2)) / 2 * 10**9

            # Now parallel estimation
            music_chunk = np.empty(n_samples, dtype=np.float32)
            slope_chunk = np.empty(n_samples, dtype=np.float32)

            def safe_music(x, y, z):
                try:
                    return z + (music(x) -87.5 + music(y) -87.5)/2 
                except Exception as e:
                    # We use nan for broken inputs
                    print("WARNING: MUSIC inf")
                    return np.nan

            def safe_slope(x, y, z):
                try:
                    return z + (slope(x) -87.5 + slope(y) -87.5)/2
                except Exception as e:
                    print("WARNING: SLOPE inf")
                    return np.nan
                
            def process_both(x, y, z):
                return safe_music(x, y, z), safe_slope(x, y, z)
            
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_both)(feature_vector2, feature_vector4, rtt) for feature_vector2, feature_vector4, rtt in zip(cach_chunk_arr2, cach_chunk_arr4, rtt_chunk)
            )
            music_results, slope_results = zip(*results)
            valid_mask = ~np.isnan(music_results) & ~np.isnan(slope_results)
            print("Valid are ", len(valid_mask), " from ", n_samples)

            music_chunk[:] = music_results
            slope_chunk[:] = slope_results

            yield features2_chunk[valid_mask], features4_chunk[valid_mask], targets_chunk[valid_mask], music_chunk[valid_mask], slope_chunk[valid_mask], rtt_chunk[valid_mask]

class RealStPDataset:
    def __init__(self, 
                 fpath, 
                 max_samples_num=float("inf"), 
                 channel_len=None, 
                 delta=0, 
                 dtype=np.float32):
        self.channel_len = channel_len
        self.fpath = fpath
        self.max_samples_num = max_samples_num
        self.samples_num = 0
        self.left_guard_pos = 0
        self.right_guard_pos = 255
        self.delta = delta
        lola = self.left_guard_pos + self.right_guard_pos
        self.left_mid_pos = lola // 2 - delta
        self.right_mid_pos = lola // 2 + 1 + delta
        self.use_channel_num = self.right_guard_pos - self.left_guard_pos - 2 * delta
        self.dtype = dtype

    def chunk_generator(self, chunk_size=10000):
        print(f"DEBUG: Loading .mat file header from {self.fpath} ...")
        raw_feature_vector_data = scio.loadmat(self.fpath)
        total_samples = min(raw_feature_vector_data["data"].shape[0], self.max_samples_num)
        print(f"DEBUG: Total samples: {total_samples}")

        feature_vector_data = raw_feature_vector_data["data"]

        feature_vector_target = np.concatenate([
            np.arange(self.left_guard_pos, self.left_mid_pos + 1),
            np.arange(self.right_mid_pos, self.right_guard_pos + 1)
        ])
        n_channels = len(feature_vector_target)

        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            n_samples = chunk_end - chunk_start
            print(f"DEBUG: chunk_end: {chunk_end}, n_samples: {n_samples}")

            # Preallocate output arrays
            features_chunk = np.empty((n_samples, n_channels, 2), dtype=self.dtype)
            cach_chunk_arr = np.empty((n_samples, n_channels), dtype=np.complex64)
            targets_chunk = np.empty(n_samples, dtype=np.float64)
            valid_mask = np.zeros(n_samples, dtype=bool)

            # Efficient extraction with minimal Python looping
            k1=0
            k2=0
            for i in range(n_samples):
                idx = chunk_start + i
                feature_vector_vector = feature_vector_data[idx][0][:,0] # shape (257, 1)

                if np.all(feature_vector_vector == feature_vector_vector[0]):
                    k1+=1
                    # print(f'WARNING: feature_vector vector {idx} has constant values: {feature_vector_vector[0]}')
                    continue

                unique, counts = np.unique(feature_vector_vector, return_counts=True)
                max_count_ratio = counts.max() / len(feature_vector_vector)

                if max_count_ratio >= 0.2:
                    k2+=1
                    # print(f"WARNING: feature_vector vector {idx} has ≥20% repeated value ({unique[counts.argmax()]})")
                    continue

                # feature_vector_channels = feature_vector_vector[0]#[feature_vector_vector[0][j] for j in feature_vector_target]
                cach_chunk_arr[i, :] = feature_vector_vector
                features_chunk[i, :, 0] = np.imag(feature_vector_vector)
                features_chunk[i, :, 1] = np.real(feature_vector_vector)
                targets_chunk[i] = feature_vector_data[idx][1].item()
                valid_mask[i] = True

            print(f"Same content are: {k1}. 20% and higher are same: {k2}. Finally {k1+k2} from ", n_samples)
            # print("Valid are ", len(valid_mask[valid_mask == True]), " from ", n_samples)

            yield features_chunk[valid_mask], targets_chunk[valid_mask]
        
if __name__ == "__main__":
    # fpath = os.path.join(DATA_DIR, "Intel_dataset", "RTT_data.csv")
    # intel_dataset = IntelDataset(fpath)

    # print("Try to call class: ", intel_dataset)

    fpath = os.path.join(DATA_DIR, "synth_dataset", "feature_vector_[25]_[200_600]_[40x10k]_[chanlen_aggr]_[model_A].mat")
    synth_dataset = SynthDataset(fpath)

    print("Try to call class: ", synth_dataset)