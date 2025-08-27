import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import pickle

from lib import PATH
from .raw_io import RealStPDataset, load_data, SynthDataset, SynthExtDataset

DATA_DIR = os.path.join(PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
INTEL_DATA_DIR = os.path.join(DATA_DIR, "Intel_dataset")
SYNTH_DATA_DIR = os.path.join(DATA_DIR, "synth_dataset")
NEW_SYNTH_DATA_DIR = os.path.join(DATA_DIR, "new_synth_dataset")
SYNTH_EXT_DATA_DIR = os.path.join(DATA_DIR, "unreal_synth_dataset")
REAL_STP_DATA_DIR = os.path.join(DATA_DIR, "raw_data_v2")

def process_real():
    # --- Process files and plot on the same canvas using subplots ---

    # files = [
    #     'data/0p35.mat', 'data/2.mat', 'data/5.mat', 'data/8.mat',
    #     'data/10.mat', 'data/13.mat', 'data/15.mat', 'data/18.mat'
    #      ] #['data/0p2.mat', 'data/3.mat', 'data/8.mat']

    files = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isfile(os.path.join(RAW_DATA_DIR, f))]

    all_features, all_targets = [], []
    all_slope, all_music = [], []
    all_rtt = []
    all_t1 = []
    all_t2 = []
    all_t3 = []
    all_t4 = []
    n_items_in_f = np.array([])

    for f in tqdm(files, desc="Processing files"):
        print(f"Processing {f}")
        feat, tar, slope_est, music_est, rtt_f, t1_f, t2_f, t3_f, t4_f = load_data(os.path.join(RAW_DATA_DIR, f))
        all_features.append(feat)
        all_targets.append(tar)
        all_slope.append(slope_est)
        all_music.append(music_est)
        all_rtt.append(rtt_f)
        all_t1.append(t1_f)
        all_t2.append(t2_f)
        all_t3.append(t3_f)
        all_t4.append(t4_f)
        n_items_in_f = np.append(n_items_in_f, len(tar))
        print(f"DEBUG: number of data points in file {f} = {n_items_in_f}")
        print(f"DEBUG: targets are = {tar}")

    features = np.concatenate(all_features, axis=0)  # (num_total_samples, 4, N)
    targets = np.concatenate(all_targets, axis=0)
    slope_estimates_all = np.concatenate(all_slope, axis=0)
    music_estimates_all   = np.concatenate(all_music, axis=0)
    rtt   = np.concatenate(all_rtt, axis=0)
    t1 = np.concatenate(all_t1, axis=0)
    t2 = np.concatenate(all_t2, axis=0)
    t3 = np.concatenate(all_t3, axis=0)
    t4 = np.concatenate(all_t4, axis=0)

    # --- Expand dims to add channel for CNN ---
    # New shape becomes: (num_samples, 4, N, 1)
    # features = np.expand_dims(features, axis=-1) #(sum of point in each file, 4, 256, 1)

    # --- Save data in DataFrame ---

    # Construct DataFrame
    df = pd.DataFrame({
        'features': list(features),  # store each feature array as a row
        'target': targets,
        'slope_est': slope_estimates_all,
        'music_est': music_estimates_all,
        'rtt' : rtt,
        'time_stamp_1': t1,
        'time_stamp_2': t2,
        'time_stamp_3': t3,
        'time_stamp_4': t4
    })

    # Optional: add file index if you want to trace back
    file_indices = []
    for i, count in enumerate(n_items_in_f.astype(int)):
        file_indices.extend([files[i]] * count)
    df['source_file'] = file_indices

    print("DEBUG: Here is final DataFrame of with processed data")
    print(df)

    # Save to disk — choose your preferred format:
    # df.to_csv("processed_data.csv", index=False)              # CSV format
    df.to_pickle(os.path.join(DATA_DIR, "processed_data_real_test.pkl"))  # Pickle format (for fast saving/loading)
    # df.to_parquet("processed_data.parquet")                   # Parquet format (best for large datasets)

def process_synth_chunked():    
    """
    This data do have some lables for type of measurment (condition of experiment), but for now we do not implement it, 
    because it is actually not informative. 
    """
    fpath = os.path.join(SYNTH_DATA_DIR, "feat_vec_[25]_[200_600]_[40x10k]_[chanlen_3-15]_[model_BCDE].mat")
    out_features_path = os.path.join(SYNTH_DATA_DIR, "features_synth.npy")
    out_targets_path = os.path.join(SYNTH_DATA_DIR, "targets_synth.npy")
    out_music_path = os.path.join(SYNTH_DATA_DIR, "music_synth.npy")
    out_slope_path = os.path.join(SYNTH_DATA_DIR, "slope_synth.npy")

    dataset = SynthDataset(fpath, dtype=np.float32) #, max_samples_num = 10
    features_list, targets_list, music_list, slope_list = [], [], [], []

    chunk_size=10000

    for features, targets, music, slope in tqdm(dataset.chunk_generator(chunk_size=chunk_size), desc="Processing chunks", total = 520000/chunk_size):
        features_list.append(features)
        targets_list.append(targets)
        music_list.append(music)
        slope_list.append(slope)

    # Efficient final concatenation
    features_arr = np.concatenate(features_list, axis=0).transpose(0, 2, 1)
    targets_arr = np.concatenate(targets_list, axis=0)
    music_arr = np.concatenate(music_list, axis=0)
    slope_arr = np.concatenate(slope_list, axis=0)
    print("DEBUG: Final features shape:", features_arr.shape)
    print("DEBUG: Final targets shape:", targets_arr.shape)
    print("DEBUG: Final music shape:", music_arr.shape)
    print("DEBUG: Final slope shape:", slope_arr.shape)
    np.save(out_features_path, features_arr)
    np.save(out_targets_path, targets_arr)
    np.save(out_music_path, music_arr)
    np.save(out_slope_path, slope_arr)

    # If you want a DataFrame (not recommended for this shape):
    # df = pd.DataFrame({'features': list(features_arr), 'target': targets_arr})
    # df.to_pickle(os.path.join(DATA_DIR, "processed_data_synth.pkl"))

def process_new_synth_chunked():    
    """
    This data do have some lables for type of measurment (condition of experiment), but for now we do not implement it, 
    because it is actually not informative. 
    """
    # fpath = os.path.join(SYNTH_DATA_DIR)
    files = [f for f in os.listdir(NEW_SYNTH_DATA_DIR) if os.path.isfile(os.path.join(NEW_SYNTH_DATA_DIR, f))]
    out_features_path = os.path.join(NEW_SYNTH_DATA_DIR,     'processed',"features_synth.npy")
    out_targets_path = os.path.join(NEW_SYNTH_DATA_DIR,      'processed',"targets_synth.npy")
    out_music_path = os.path.join(NEW_SYNTH_DATA_DIR,        'processed',"music_synth.npy")
    out_slope_path = os.path.join(NEW_SYNTH_DATA_DIR,        'processed',"slope_synth.npy")
    out_source_path = os.path.join(NEW_SYNTH_DATA_DIR,       'processed', "source_synth.npy")

    features_list, targets_list, music_list, slope_list, source_file = [], [], [], [], []

    chunk_size=2000
    for f in files:
        dataset = SynthDataset(os.path.join(NEW_SYNTH_DATA_DIR, f), dtype=np.float32) #, max_samples_num = 10
        for features, targets, music, slope in tqdm(dataset.chunk_generator(chunk_size=chunk_size), desc="Processing chunks", total = 10000/chunk_size):
            features_list.append(features)
            targets_list.append(targets)
            music_list.append(music)
            slope_list.append(slope)
            source_file.append([f]*len(targets))


    # Efficient final concatenation
    features_arr = np.concatenate(features_list, axis=0).transpose(0, 2, 1)
    targets_arr = np.concatenate(targets_list, axis=0)
    music_arr = np.concatenate(music_list, axis=0)
    slope_arr = np.concatenate(slope_list, axis=0)
    source_arr = np.concatenate(source_file, axis=0)
    print("DEBUG: Final features shape:", features_arr.shape)
    print("DEBUG: Final targets shape:", targets_arr.shape)
    print("DEBUG: Final music shape:", music_arr.shape)
    print("DEBUG: Final slope shape:", slope_arr.shape)
    np.save(out_features_path, features_arr)
    np.save(out_targets_path, targets_arr)
    np.save(out_music_path, music_arr)
    np.save(out_slope_path, slope_arr)
    np.save(out_source_path, source_arr)

def process_synth_ext_chunked():    
    """
    This data do have some lables for type of measurment (condition of experiment), but for now we do not implement it, 
    because it is actually not informative. 
    """
    # fpath = os.path.join(SYNTH_DATA_DIR)
    files = [f for f in os.listdir(SYNTH_EXT_DATA_DIR) if os.path.isfile(os.path.join(SYNTH_EXT_DATA_DIR, f))]
    out_features2_path = os.path.join(SYNTH_EXT_DATA_DIR,     'processed', "features_incended_synth.npy")
    out_features4_path = os.path.join(SYNTH_EXT_DATA_DIR,     'processed', "features_reflected_synth.npy")
    out_targets_path = os.path.join(SYNTH_EXT_DATA_DIR,      'processed', "targets_synth.npy")
    out_music_path = os.path.join(SYNTH_EXT_DATA_DIR,        'processed', "music_synth.npy")
    out_slope_path = os.path.join(SYNTH_EXT_DATA_DIR,        'processed', "slope_synth.npy")
    out_source_path = os.path.join(SYNTH_EXT_DATA_DIR,       'processed', "source_synth.npy")
    out_rtt_path = os.path.join(SYNTH_EXT_DATA_DIR,          'processed', "rtt_synth.npy")

    features2_list, features4_list, targets_list, music_list, slope_list, source_file, rtt = [], [], [], [], [], [], []

    chunk_size=2000
    for f in files:
        dataset = SynthExtDataset(os.path.join(SYNTH_EXT_DATA_DIR, f), dtype=np.float32) #, max_samples_num = 10
        for features2, features4, targets, music, slope, rtt_i in tqdm(dataset.chunk_generator(chunk_size=chunk_size), desc="Processing chunks", total = 10000/chunk_size):
            features2_list.append(features2)
            features4_list.append(features4)
            targets_list.append(targets)
            music_list.append(music)
            slope_list.append(slope)
            source_file.append([f]*len(targets))
            rtt.append(rtt_i)


    # Efficient final concatenation
    features2_arr = np.concatenate(features2_list, axis=0).transpose(0, 2, 1)
    features4_arr = np.concatenate(features4_list, axis=0).transpose(0, 2, 1)
    targets_arr = np.concatenate(targets_list, axis=0)
    music_arr = np.concatenate(music_list, axis=0)
    slope_arr = np.concatenate(slope_list, axis=0)
    source_arr = np.concatenate(source_file, axis=0)
    rtt_arr = np.concatenate(rtt, axis=0)
    print("DEBUG: Final features2 shape:", features2_arr.shape)
    print("DEBUG: Final features4 shape:", features4_arr.shape)
    print("DEBUG: Final targets shape:", targets_arr.shape)
    print("DEBUG: Final music shape:", music_arr.shape)
    print("DEBUG: Final slope shape:", slope_arr.shape)
    print("DEBUG: Final rtt shape:", rtt_arr.shape)
    np.save(out_features2_path, features2_arr)
    np.save(out_features4_path, features4_arr)
    np.save(out_targets_path, targets_arr)
    np.save(out_music_path, music_arr)
    np.save(out_slope_path, slope_arr)
    np.save(out_source_path, source_arr)
    np.save(out_rtt_path, rtt_arr)

def process_real_StP_chunked():    
    """
    This data do have some lables for type of measurment (condition of experiment), but for now we do not implement it, 
    because it is actually not informative. 
    """
    # fpath = os.path.join(SYNTH_DATA_DIR)
    files = [f for f in os.listdir(REAL_STP_DATA_DIR) if os.path.isfile(os.path.join(REAL_STP_DATA_DIR, f))]
    out_features_path = os.path.join(REAL_STP_DATA_DIR,     'processed', "features_stp.npy")
    out_targets_path = os.path.join(REAL_STP_DATA_DIR,      'processed', "targets_stp.npy")
    out_source_path = os.path.join(REAL_STP_DATA_DIR,       'processed', "source_stp.npy")

    features_list, targets_list, source_file = [], [], []
    
    chunk_size=5000
    for f in files:
        dataset = RealStPDataset(os.path.join(REAL_STP_DATA_DIR, f), dtype=np.float32) #, max_samples_num = 10
        for features, targets in tqdm(dataset.chunk_generator(chunk_size=chunk_size), desc="Processing chunks", total = 10000/chunk_size):
            features_list.append(features)
            targets_list.append(targets)
            source_file.append([f]*len(targets))


    # Efficient final concatenation
    features_arr = np.concatenate(features_list, axis=0).transpose(0, 2, 1)
    targets_arr = np.concatenate(targets_list, axis=0)
    source_arr = np.concatenate(source_file, axis=0)
    print("DEBUG: Final features shape:", features_arr.shape)
    print("DEBUG: Final targets shape:", targets_arr.shape)
    np.save(out_features_path, features_arr)
    np.save(out_targets_path, targets_arr)
    np.save(out_source_path, source_arr) 

def make_dummy_dataset(n_samples=1000, n_features=256, out_file=None):
    """
    Create a dummy dataset with the same structure as the original project
    and save it to a .pkl file.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Length of each feature vector (per branch).
    out_file : str
        Output .pkl filename.
    """
    if out_file is None: out_file = os.path.join(DATA_DIR, 'dummy_data.pkl')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
    rng = np.random.default_rng()

    # Possible source files (scenarios)
    source_files = [
        'Pure0p35', 'Pure2', 'Pure5', 'Pure8', 
        'Pure10', 'Pure13', 'Pure15', 'Pure18',  
        'Noisy3','Noisy5', 'Noisy8', 'Noisy12', 
        'Perturbation8', 'Perturbation9', 'Perturbation10', 'Perturbation11'
    ]

    data = []
    for _ in range(n_samples):
        # Random "features" (2 vectors of length n_features)
        features = rng.normal(size=(4, n_features)).round(3).tolist()

        # Random targets & estimates
        target = float(rng.uniform(0, 20))              # example: distance in [0,20]
        slope_est = float(rng.normal(10, 5))
        music_est = float(rng.normal(10, 5))
        rtt = float(rng.normal(0, 50))

        # Random timestamps (big integers ~10^12)
        time_stamps = rng.integers(6e12, 8e12, size=4).astype(np.int64)

        # Random source file
        source_file = random.choice(source_files)

        row = {
            "features": features,
            "target": target,
            "slope_est": slope_est,
            "music_est": music_est,
            "rtt": rtt,
            "time_stamp_1": time_stamps[0],
            "time_stamp_2": time_stamps[1],
            "time_stamp_3": time_stamps[2],
            "time_stamp_4": time_stamps[3],
            "source_file": source_file,
        }
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    print(df.head())

    # Save as pickle
    with open(out_file, "wb") as f:
        pickle.dump(df, f)

    print(f"Dummy dataset saved to {out_file} with {n_samples} samples.")
    