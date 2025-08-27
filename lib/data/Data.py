import os
import numpy as np
import pandas as pd

from lib import PATH
DATA_DIR = os.path.join(PATH, "data")

def l2_normalize_features(features):
    # features: (2, 256)
    norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12  # to avoid division by zero
    return features / norms

class Data:
    """
    Unified loader and preprocessor for datasets (synthetic and real).

    Provides a consistent interface to load, sample, and normalize multiple dataset
    variants used in 2 feature vector-based target estimation research. Internally, data is stored 
    in a `pandas.DataFrame` where each row corresponds to one sample and columns
    store features and auxiliary labels (target target, MUSIC/ slope estimates, RTT, etc.).

    The class can be called like a function (:meth:`__call__`) to directly load a
    dataset by name.

    Para
    ----------
    ``random`` : int, default 42
        Random seed for reproducibility when sampling subsets.
    ``num_samples`` : int or float, default np.inf
        Number of samples to keep. If larger than dataset size, keeps all.
    ``norm`` : bool, default False
        Whether to apply L2 feature normalization after loading.
    ``**kwargs`` :
        Reserved for future extensions (ignored currently).

    :meth:`__call__`
    ----------
    Load a dataset by name and return a preprocessed DataFrame.

        Args :
            dataset_name (str) : Dataset identifier. One of:

                - ``synth_v2``
                - ``synth_v3``
                - ``real`` - original Feiyu data
                - ``real_pure`` (real with excluded files)
                - ``real_stp`` - data from Saint Petersburg Datacom
            file_name (str, optional): Pickle file for real datasets. Defaults to "processed_data_real.pkl".

        Returns :
            pd.DataFrame: Loaded dataset with features and labels.

    Attributes
    ----------
    df : pandas.DataFrame
        Tabular dataset representation with different schema depending on dataset type.
        Typical columns include:
        
        - ``features`` : ndarray of shape (2, 256)
        - ``features_incended`` : ndarray of shape (2, 256)
        - ``features_reflected`` : ndarray of shape (2, 256)
        - ``target`` : float, target [m]
        - ``music_est`` : float, estimated target via MUSIC [m]
        - ``slope_est`` : float, estimated target via slope [m]
        - ``rtt`` : float, round-trip time [ns]
        - ``source_file`` : str, filename the sample originated from

    See Also
    --------
    pandas.DataFrame : Underlying tabular representation of the dataset.
    l2_normalize_features : Utility to normalize features.

    Notes
    -----
    - Synthetic v2 and v3 datasets are loaded from preprocessed NumPy arrays.
    - Real datasets are loaded from preprocessed pickle files or .npy dumps.
    - Sampling and normalization are optional but recommended for model training.

    Examples
    --------
    Load a synthetic dataset and inspect shape:

    >>> loader = Data(random=123, num_samples=500, norm=True)
    >>> df = loader("synth_v2")
    >>> df.head()
             features     target   music_est  slope_est source_file
    0  [[... 2x256 ...]]  1.23      1.25       1.20      'file1'
    1  [[... 2x256 ...]]  3.45      3.44       3.47      'file2'

    Load real dataset excluding specific files:

    >>> df = loader("real_pure", file_name="processed_data_real.pkl")
    >>> len(df)
    780

    Apply feature normalization automatically:

    >>> loader = Data(norm=True)
    >>> df = loader("real")
    >>> df["features_incended"].iloc[0].shape
    (2, 256)
    """
    def __init__(self, random: int = 42, num_samples: int = np.inf, norm: bool = False, **kwargs):     
        self.rand_state = random
        self.num_samples = num_samples
        self.norm = norm

    def __call__(self, dataset_name: str, file_name: str = "processed_data_real.pkl") -> pd.DataFrame:
        """
        Load a dataset by name and return a preprocessed DataFrame.

        Args:
            dataset_name (str): Dataset identifier. One of:
                - "synth_v2"
                - "synth_v3"
                - "real"
                - "real_pure" (real with excluded files)
                - "real_stp"
            file_name (str, optional): Pickle file for real datasets. Defaults to "processed_data_real.pkl".

        Returns:
            pd.DataFrame: Loaded dataset with features and labels.
        """        
        # Dataset INI
        if   dataset_name == 'synth_v2': self.synth_v2()
        elif dataset_name == 'synth_v3': self.synth_v3()
        # elif dataset_name == 'synth_v4': self.synth_v4()
        elif dataset_name == 'real':     self.real(file_name=file_name)
        elif dataset_name == 'real_pure':self.real(excluded_files=[
                'data0p2.mat', 'data3.mat', 'data8.mat', 
                'Noisy3','Noisy5', 'Noisy8', 'Noisy12',
                'Perturbation8', 'Perturbation9', 'Perturbation10', 'Perturbation11'
            ], file_name=file_name)
        elif dataset_name == 'real_stp': self.real_stp()
        elif dataset_name == 'real_intel': self.real_intel()

        self.samples_cut(self.num_samples) # also shaffles the data

        # Normalizae features
        if self.norm: self.normalize_feature(self, dataset_name)

        return self.df
    
    def synth_v3(self, synth_dir: str = None):
        """
        Load synthetic v3 dataset from .npy files.

        Args:
            synth_dir (str, optional): Directory containing processed files.
                Defaults to "data/unreal_synth_dataset/processed".
        
        Side Effects:
            Sets self.df with columns:
                - features_incended (np.ndarray of shape (2, 256))
                - features_reflected (np.ndarray of shape (2, 256))
                - target (float, )
                - music_est, slope_est (float, )
                - rtt (float, ns)
                - source_file (str)
        """        
        # Load data from disk
        if synth_dir is None: synth_dir = os.path.join(DATA_DIR, 'unreal_synth_dataset', 'processed')
        # Feature vector (num samples, [imag, real], 256 elements of 1 feature)
        features2:np.ndarray = np.load(os.path.join(synth_dir, 'features_incended_synth.npy'))  # shape: (40000, 2, 256) 
        features4:np.ndarray = np.load(os.path.join(synth_dir, 'features_reflected_synth.npy'))  # shape: (40000, 2, 256) 
        targets = np.load(os.path.join(synth_dir, 'targets_synth.npy'))    # shape: (40000,)  - real target
        music_est = np.load(os.path.join(synth_dir,  'music_synth.npy'))    # shape: (40000,)   rtt + ( delta_t(incended)+ delta_t(reflected))/2
        slope_est = np.load(os.path.join(synth_dir,  'slope_synth.npy'))    # shape: (40000,)   rtt + ( delta_t(incended)+ delta_t(reflected))/2
        source_file = np.load(os.path.join(synth_dir,  'source_synth.npy'))    # shape: (40000,)
        rtt = np.load(os.path.join(synth_dir,  'rtt_synth.npy'))    # shape: (40000,)   ((t4-t1)-(t3-t2))/2

        # Option 1: Each row is a feature array (good for simple row-wise operations)
        self.df = pd.DataFrame({
            'features_incended': list(features2),  # Each entry is a (256, 2) array
            'features_reflected': list(features4),  # Each entry is a (256, 2) array
            'target': targets,          # real target
            'music_est': music_est*0.3, # delta_t -> real target
            'slope_est': slope_est*0.3, # delta_t -> real target
            'rtt': rtt,
            'source_file': source_file
        })

        print(f'DEBUG: dataset was loaded:\n{self.df.head()}')

    def synth_v2(self, synth_dir: str = None):
        """
        Load synthetic v2 dataset from .npy files.

        Args:
            synth_dir (str, optional): Directory containing processed files.
                Defaults to "data/new_synth_dataset/processed".

        Side Effects:
            Sets self.df with columns:
                - features (np.ndarray of shape (2, 256))
                - target, music_est, slope_est (float, )
                - source_file (str)
        """        
        # Load data from disk
        if synth_dir is None: synth_dir = os.path.join(DATA_DIR, 'new_synth_dataset', 'processed')
        # Feature vector (num samples, [imag, real], 256 elements of 1 feature)
        features:np.ndarray = np.load(os.path.join(synth_dir, 'features_synth.npy'))  # shape: (40000, 2, 256) 
        targets = np.load(os.path.join(synth_dir, 'targets_synth.npy'))    # shape: (40000,) delta_t*0.3 = real target
        music_est = np.load(os.path.join(synth_dir,  'music_synth.npy'))    # shape: (40000,) delta_t*0.3 = real target
        slope_est = np.load(os.path.join(synth_dir,  'slope_synth.npy'))    # shape: (40000,) delta_t*0.3 = real target
        source_file = np.load(os.path.join(synth_dir,  'source_synth.npy'))    # shape: (40000,)

        # print(np.array(features).shape)

        # Option 1: Each row is a feature array (good for simple row-wise operations)
        self.df = pd.DataFrame({
            'features': list(features),  # Each entry is a (256, 2) array
            'target': targets*0.3,      # delta_t tranform to real target
            'music_est': music_est*0.3, # delta_t tranform to real target
            'slope_est': slope_est*0.3, # delta_t tranform to real target
            'source_file': source_file
        })

        print(f'DEBUG: dataset was loaded:\n{self.df.head()}')

    def real(self, 
            path: str = None, 
            excluded_files: list =['data0p2.mat', 'data3m.mat', 'data8.mat'], 
            file_name: str = "processed_data_real.pkl"
        ):
        """
        Load real dataset from pickle file.

        Args:
            path (str, optional): Directory containing pickle file. Defaults to DATA_DIR.
            excluded_files (list[str], optional): List of files to exclude. Defaults to some known files.
            file_name (str, optional): Pickle file name. Defaults to "processed_data_real.pkl".

        Side Effects:
            Sets self.df with additional columns:
                - features_incended (np.ndarray of shape (2, 256))
                - features_reflected (np.ndarray of shape (2, 256))
        """        
        # Load data from disk
        # target, slope and music is real target
        if path is None: path = DATA_DIR
        df = pd.read_pickle(os.path.join(path, file_name))
        if excluded_files is not None: 
            df = df[~df['source_file'].isin(excluded_files)].reset_index(drop=True)

        df['features_incended']  = list(np.stack(df['features'])[:,:2,:]) # shape: (800, 2, 256) 
        df['features_reflected'] = list(np.stack(df['features'])[:,2:,:]) # shape: (800, 2, 256) 

        self.df = df
        print(f'DEBUG: dataset was loaded:\n{self.df.head()}')

    def real_stp(self, path: str = None):
        """
        Load step dataset (real measurements).

        Args:
            path (str, optional): Directory containing processed .npy files.
                Defaults to "data/raw_data_v2/processed".

        Side Effects:
            Sets self.df with columns:
                - features (np.ndarray of shape (2, 256))
                - target (float, )
                - source_file (str)
        """
        # Load data from disk
        if path is None: path = os.path.join(DATA_DIR, 'raw_data_v2', 'processed')
        # Feature vector (num samples, [imag, real], 256 elements of 1 feature)
        features:np.ndarray = np.load(os.path.join(path, 'features_stp.npy'))  # shape: (103659, 2, 256) 
        targets = np.load(os.path.join(path, 'targets_stp.npy'))    # shape: (103659,)  - real target
        source_file = np.load(os.path.join(path,  'source_stp.npy'))    # shape: (103659,)

        # Option 1: Each row is a feature array (good for simple row-wise operations)
        self.df = pd.DataFrame({
            'features': list(features),  # Each entry is a (256, 2) array
            'target': targets,
            'source_file': source_file,
            'music_est': None, # no rtt provided
            'slope_est': None, # no rtt provided
        })

        print(f'DEBUG: dataset was loaded:\n{self.df.head()}')

    def samples_cut(self, num_samples: int = 1000):
        """
        Randomly sample a fixed number of rows from the dataset.

        Args:
            num_samples (int, optional): Number of samples to keep. Defaults to 1000.

        Side Effects:
            Updates self.df with a shuffled and truncated version.
        """
        num_samples = min(num_samples, len(self.df))
        self.df = self.df.sample(n=num_samples, random_state=self.rand_state).reset_index(drop=True)  # random_state ensures reproducibility
    
    def normalize_feature(self, dataset_type: str):
        """
        Apply L2 normalization to features depending on dataset type.

        Args:
            dataset_type (str): Dataset type ("real" or synthetic).
        
        Side Effects:
            Updates 'features' or 'features_incended'/'features_reflected' columns in self.df.
        """
        if dataset_type == 'real':
            self.df['features'] = self.df['features'].apply(l2_normalize_features)
        else:
            self.df['features_incended'] = self.df['features_incended'].apply(l2_normalize_features)
            self.df['features_reflected'] = self.df['features_reflected'].apply(l2_normalize_features)