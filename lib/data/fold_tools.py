from __future__ import annotations

from typing import Callable, Sequence, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

from lib.utils.tools import extract_sort_key

############################
#         Folds
############################

def fold_n_files(
    df: pd.DataFrame,
    extract_sort_key: Optional[Callable[[str], Any]] = extract_sort_key
) -> List[Tuple[List[int], List[int]]]:
    """
    Leave-one-file-out folds.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'source_file' column.
    extract_sort_key : callable, optional
        Key function to sort unique file names (e.g., extract numeric distance).

    Returns
    -------
    list[tuple[list[int], list[int]]]
        Each tuple is (train_idx, test_idx).
    """
    folds: List[Tuple[List[int], List[int]]] = []
    file_list = sorted(df['source_file'].unique(), key=extract_sort_key)
    # print(f'DEBUG: fold ordering is: {file_list}')
    for f in file_list:
        train_idx = df.index[df['source_file'] != f].tolist()
        test_idx = df.index[df['source_file'] == f].tolist()
        folds.append((train_idx, test_idx))
    return folds


def fold_n_targets(df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
    """
    Leave-one-target-out folds.

    Uses unique values of 'target' as held-out groups.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'target' column.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        Each tuple is (train_idx, test_idx).
    """
    folds: List[Tuple[List[int], List[int]]] = []
    target_list = sorted(df['target'].unique())  # float sorting by default
    for t in target_list:
        train_idx = df.index[df['target'] != t].tolist()
        test_idx = df.index[df['target'] == t].tolist()
        folds.append((train_idx, test_idx))
    return folds


def fold_n_files_exclude(
    df: pd.DataFrame,
    extract_sort_key: Optional[Callable[[str], Any]] = None,
    excluded_files: Sequence[str] = ('some_file_1.mat', 'some_file_2.mat'),
) -> List[Tuple[List[int], List[int]]]:
    """
    LOFO on non-excluded files + extra folds for excluded ones.

    Parameters
    ----------
    df : DataFrame
        Must contain 'source_file'.
    extract_sort_key : callable, optional
        Sort key for file names.
    excluded_files : sequence of str
        Files to treat specially (test-only folds).

    Returns
    -------
    list[tuple[list[int], list[int]]]
        (train_idx, test_idx) over original df indices.
    """
    if 'source_file' not in df.columns:
        raise KeyError("df must have a 'source_file' column")

    in_excl = df['source_file'].isin(excluded_files)
    df_pure = df[~in_excl]
    df_excl = df[in_excl]

    file_list = sorted(df_pure['source_file'].unique(), key=extract_sort_key)
    excl_list = sorted(df_excl['source_file'].unique(), key=extract_sort_key)

    folds: List[Tuple[List[int], List[int]]] = []

    # Folds for non-excluded files: leave-one-file-out on df_pure
    print(f"DEBUG: fold ordering is: {file_list}")
    for f in file_list:
        test_idx = df.index[df['source_file'] == f].tolist()                    # from original df
        train_idx = df.index[(~in_excl) & (df['source_file'] != f)].tolist()    # all pure except f
        print(f"{f} : {len(train_idx)}, {len(test_idx)}")
        folds.append((train_idx, test_idx))

    # Excluded: test = excluded file, train = all pure
    print(f"DEBUG: fold ordering excluded is: {excl_list}")
    for f in excl_list:
        test_idx = df.index[df['source_file'] == f].tolist()        # from original df
        train_idx = df.index[~in_excl].tolist()                     # all pure files
        print(f"{f} : {len(train_idx)}, {len(test_idx)}")
        folds.append((train_idx, test_idx))

    return folds


def fold_by_kmeans_target(
    df: pd.DataFrame,
    n_clusters: int = 5
) -> Tuple[List[Tuple[List[int], List[int]]], np.ndarray]:
    """
    Leave-one-cluster-out folds by k-means on 'target'.

    Parameters
    ----------
    df : DataFrame
        Must contain 'target'.
    n_clusters : int
        Number of k-means clusters.

    Returns
    -------
    (folds, labels)
        folds : list of (train_idx, test_idx),
        labels : np.ndarray of cluster mean labels like 'X.Ym' sorted by mean target.
    """
    df = df.copy()
    targets = df['target'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(targets)
    df['target_cluster'] = kmeans.labels_

    # Compute cluster means
    cluster_means = df.groupby('target_cluster')['target'].mean().sort_values()

    # Get cluster labels sorted by mean target value
    sorted_cluster_ids = cluster_means.index.tolist()

    folds: List[Tuple[List[int], List[int]]] = []
    for c in sorted_cluster_ids:
        train_idx = df.index[df['target_cluster'] != c].tolist()
        test_idx = df.index[df['target_cluster'] == c].tolist()
        folds.append((train_idx, test_idx))
    
    return folds, np.array([f'{val:.1f}m' for val in cluster_means])


def fold_10euqal(df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
    """
    10 stratified-by-file folds (per-file KFold combined).

    Returns 10 folds; each fold aggregates splits across files.

    Parameters
    ----------
    df : DataFrame
        Must contain 'source_file'.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        Ten (train_idx, test_idx) folds.
    """
    folds: List[Tuple[List[int], List[int]]] = []
    for f in df['source_file'].unique():
        indices = np.where(df['source_file'] == f)[0]
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
            if len(folds) <= fold_idx:
                folds.append(([], []))
            folds[fold_idx][0].extend(indices[train_idx])
            folds[fold_idx][1].extend(indices[test_idx])
    return folds


def fold_1equal(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Single stratified split per file, aggregated into one fold.

    Parameters
    ----------
    df : DataFrame
        Must contain 'source_file'.
    test_size : float
        Test proportion within each file.
    random_state : int
        RNG seed.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        One (train_idx, test_idx) fold.
    """
    train_indices: List[int] = []
    test_indices: List[int] = []
    for f in df['source_file'].unique():
        indices = np.where(df['source_file'] == f)[0]
        tr, te = train_test_split(indices, test_size=test_size,
                                  random_state=random_state, shuffle=True)
        train_indices.extend(tr)
        test_indices.extend(te)
    return [(train_indices, test_indices)]


def fold_10euqal_simple(df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
    """
    Plain 10-fold split over all rows (no stratification).

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    list[tuple[list[int], list[int]]]
        Ten (train_idx, test_idx) folds.
    """
    folds: List[Tuple[List[int], List[int]]] = []
    indices = np.arange(len(df))
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        if len(folds) <= fold_idx:
            folds.append(([], []))
        folds[fold_idx][0].extend(indices[train_idx])
        folds[fold_idx][1].extend(indices[test_idx])
    return folds


def fold_1simple(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Single random 90/10 split over all rows.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One (train_idx, test_idx) fold with ndarray indices.
    """
    train_ids, test_ids = train_test_split(
        np.arange(len(df)), test_size=0.1, random_state=42, shuffle=True
    )
    return [(train_ids, test_ids)]


def fold_overfit(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Overfit diagnostic split: train on full df, test on random 10%.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One (train_idx=all, test_idx=random 10%) fold.
    """
    _, test_ids = train_test_split(
        np.arange(len(df)), test_size=0.1, random_state=42, shuffle=True
    )
    return [(np.arange(len(df)), test_ids)]


def fold_overfit_full(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extreme overfit split: train == test == all rows.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One (all, all) fold.
    """
    idx = np.arange(len(df))
    return [(idx, idx)]


def fold_3groups(df: pd.DataFrame) -> List[Tuple[List[int], pd.Series]]:
    """
    3 group-based folds by prefix of 'source_file'.

    Groups: 'Pure_data', 'Minor_noise', 'Intense_disturbance'. For each group g, test is files
    starting with g; train is all others.

    Parameters
    ----------
    df : DataFrame
        Must contain 'source_file'.

    Returns
    -------
    list[tuple[list[int], pd.Series]]
        For each group: (train_idx, test_mask_series).
    """
    folds: List[Tuple[List[int], pd.Series]] = []
    types = ['Pure_data', 'Minor_noise', 'Intense_disturbance']
    for fold_idx, fold_name in enumerate(types):
        file_list_without_t = df['source_file'][~df['source_file'].str.startswith(fold_name)].unique()
        if len(folds) <= fold_idx:
            folds.append(([], df['source_file'].str.startswith(types[fold_idx])))
        for f in file_list_without_t:
            indices = np.where(df['source_file'] == f)[0]
            train_idx = np.arange(df.count().iloc[0])[indices]
            folds[fold_idx][0].extend(train_idx.tolist())
    return folds