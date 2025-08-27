import numpy as np
from tqdm import tqdm
import pandas as pd

# Path settings
import os
import sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

from lib.data.cache import process_real, process_real_StP_chunked, \
    process_new_synth_chunked, process_synth_ext_chunked, make_dummy_dataset

if __name__ == "__main__":
    # Choose function for data extraction
    # process_real()
    # process_new_synth_chunked()
    # process_synth_ext_chunked()
    # process_real_StP_chunked()
    make_dummy_dataset(n_samples=5000, n_features=256)
    pass