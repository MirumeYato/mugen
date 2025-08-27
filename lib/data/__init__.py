# Usually empty, or you could re-export load_data here if you like.
import numpy as np

from .Data import Data
from .fold_tools import fold_1simple, fold_10euqal, fold_3groups, fold_n_files 

name_list_3groups = np.array(['Pure_data', 'Minor_noise', 'Intense_disturbance'],
        dtype=object)
name_list_n_files = np.array([
    'Pure0p35', 'Pure2', 'Pure5', 'pure8',
    'Pure10', 'Pure13', 'Pure15', 'Pure18', 
    'Noisy3','Noisy5', 'Noisy8', 'Noisy12',
    'Perturbation8', 'Perturbation9', 'Perturbation10', 'Perturbation11'],
    dtype=object)
name_list3_n_files_pure = np.array([
    '0p35', '2', '5', '8',
    '10', '13', '15', '18'],
    dtype=object)

__all__ = [name_list_3groups, name_list_n_files, name_list3_n_files_pure]