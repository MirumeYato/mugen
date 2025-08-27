import math
import os
import seaborn as sns

# Suppress GPU initialization for pure CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# Optimize threading
num_threads = os.cpu_count() or 4
print(f"DEBUG: num threads = {num_threads}")
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from tensorflow.keras.models import load_model # type: ignore
from sklearn.utils import shuffle
import tensorflow.keras.backend as K # type: ignore
import tensorflow as tf
import pandas as pd

# Path settings
import sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

DATA_DIR = os.path.join(PATH, "data")

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.get_logger().setLevel('ERROR')

from lib.cross_validation import plot_ecdf, plot_hist
from lib.tools import Data
from lib.fold_tools import BaseDataUnpacker, RangeDataUnpacker, DelayDataUnpacker, MagPhaseRangeDataUnpacker, MagPhaseDelayDataUnpacker, MusicRangeDataUnpacker, MusicDelayDataUnpacker
from lib.fold_tools import BaseDataConstructor, RangeDataConstructor, DelayDataConstructor
from lib.fold_tools import fold_3groups, fold_10euqal, fold_n_files
# from lib.fold_tools import delay_construct as pack
# from lib.model_configurations import delay_model as create_model
from lib.model_configurations import CovarianceLikeLayer, Compute_fft, ExpandDim1Layer, FFTFeaturesLayer, PositionalEncodingLayer, TransformerBlock

def plot_combined_ecdf(path, nn, slope, music, flag_res_check, file_name="combined_ecdf.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ecdf_nn = ECDF(nn)
        ax.plot(ecdf_nn.x, ecdf_nn.y, label='CNN', linewidth=2)
        
        if flag_res_check: 
            ecdf_slope, ecdf_music = ECDF(slope), ECDF(music)
            ax.plot(ecdf_slope.x, ecdf_slope.y, label='SLOPE', linestyle='--')
            ax.plot(ecdf_music.x, ecdf_music.y, label='MUSIC', linestyle=':')

        val = []
        for p in [0.8, 0.9, 0.95, 1.0]:
            val.append(ecdf_nn.x[np.searchsorted(ecdf_nn.y, p, side='right') - 1])
        ax.text(
        0.98, 0.35,
        f'CNN @ 0.8 ≈ {val[0]:.2f}m\n'
        f'CNN @ 0.9 ≈ {val[1]:.2f}m\n'
        f'CNN @ 0.95 ≈ {val[2]:.2f}m\n'
        f'CNN @ 1 ≈ {val[3]:.2f}m',
        transform=ax.transAxes, fontsize=8, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

        ax.set_title("Combined ECDF of MAE")
        ax.set_xlabel("MAE [m]")
        ax.set_ylabel("ECDF")
        ax.set_ylim([0, 1]) 
        ax.set_xlim([0, 10]) # max([ecdf.x.max() for ecdf in ecdfs])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(path, file_name), dpi=300, bbox_inches='tight')

def check_model_on_real(model, create_folding, unpacker_class, constructor_class, path, run_name, name = None, norm = False):
    df:pd.DataFrame = Data(norm = norm)('real') # here is data for testing (in general you can change it)
    folds = create_folding(df)

    # Determine optimal grid size
    num_folds = len(folds)
    cols = min(4, num_folds)  # At most 4 columns
    rows = math.ceil(num_folds / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # 4x4 inches per plot
    axes = np.array(axes).reshape(-1) if num_folds > 1 else [axes]  # Flatten safely for all cases

    fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # 4x4 inches per plot
    axes_hist = np.array(axes_hist).reshape(-1) if num_folds > 1 else [axes_hist]  # Flatten safely for all cases
    
    all_mae_nn, all_mae_slope, all_mae_music = [], [], []
    name_i = None
    for fold_idx, (__, ids) in enumerate(folds): 
        if name is not None: name_i = name[fold_idx]
        Unpacker:BaseDataUnpacker = unpacker_class(df, np.array([0]), ids, None, None)
        data, additional = Unpacker.process()
        X_test, y_test = data[2:4]
        
        y_pred = model.predict(X_test).flatten() #+ rtt# * 0.3 # MAIN RESULT
            
        # Now we get all results in needed for us format. You can define this func as you wish
        Constructor:BaseDataConstructor = constructor_class(name_i, y_pred, data, additional, None) 
        mae, results_i, __ = Constructor.process() # flag_res_check = slope and music presented in results
        mae_nn, mae_slope, mae_music = mae
            
        all_mae_nn.extend(mae_nn)
        all_mae_slope.extend(mae_slope)
        all_mae_music.extend(mae_music)

        plot_ecdf([mae_nn, mae_slope, mae_music], name_i, axes[fold_idx])
        if any([x is None for  x in results_i["rtt"]]): 
            plot_hist([y_pred, additional[0][0], additional[0][1]], name_i, axes_hist[fold_idx], results_i['target'])
        else:
            plot_hist([y_pred+results_i["rtt"], additional[0][0], additional[0][1]], name_i, axes_hist[fold_idx], results_i['target']+results_i['rtt'])

    for i in range(num_folds, len(axes)):
        fig.delaxes(axes[i])

    for i in range(num_folds, len(axes_hist)):
        fig_hist.delaxes(axes_hist[i])

    fig.suptitle("ECDFs per Fold")
    fig.tight_layout()
    if len(folds)>1: 
        fig.savefig(os.path.join(path, f"fold_ecdfs_{run_name}.png"), dpi=300, bbox_inches='tight')
        fig_hist.savefig(os.path.join(path, f"fold_hist_{run_name}.png"), dpi=300, bbox_inches='tight')
    plot_combined_ecdf(path, all_mae_nn, all_mae_slope, all_mae_music, True, file_name=f"recheck_{run_name}.png")

from scipy.stats import gaussian_kde
# Function to find KDE peak
def find_peak_kde(values):
    values = values.dropna()
    if len(values) < 3:
        return values.mean()
    kde = gaussian_kde(values)
    x_vals = np.linspace(values.min(), values.max(), 1000)
    y_vals = kde(x_vals)
    return x_vals[np.argmax(y_vals)]

def peak_wide(values):
    values = values.dropna()
    if len(values) < 3:
        return values.mean()
    return abs(max(values)-min(values))/2

def find_mean(values):
    values = values.dropna()
    if len(values) < 3:
        return values.mean()
    return np.mean(values)

def target_and_peak_comp(path_in_output='MT_hyb_real_Delay_n_files_500'):
    # df = pd.read_pickle(os.path.join(PATH, 'output', path_in_output, "model_results.pkl"))
    df = pd.read_pickle(os.path.join(path_in_output, "model_results.pkl"))
    if df['rtt'].values[0] is None:
        # Grouped computation
        grouped = df.groupby('dataset')

        # Apply peak estimation
        result = grouped['nn_predict'].apply(find_peak_kde).reset_index()
        result.columns = ['dataset', 'peak_nn']
        # First value of target per group
        target_rtt = grouped['target'].first().reset_index()
        # Mean prediction per group
        mean_pred = grouped['nn_predict'].apply(find_mean).reset_index()
        mean_pred.columns = ['dataset', 'mean_nn']
        # Peak width per group
        wide = grouped['nn_predict'].apply(peak_wide).reset_index()
        wide.columns = ['dataset', 'wide_nn']

        # Merge all results
        result = result.merge(target_rtt, on='dataset')
        result = result.merge(mean_pred, on='dataset')
        result = result.merge(wide, on='dataset')
        # Compute absolute errors
        result['peak_diff'] = np.abs(result['peak_nn'] - result['target'])
        result['mean_diff'] = np.abs(result['mean_nn'] - result['target'])

        # Final output
        return result[['dataset', 'peak_nn', 'mean_nn', 'wide_nn', 'target', 'peak_diff', 'mean_diff']]

    else:
        # Create the column for nn_predict + rtt
        df['nn_plus_rtt'] = df['nn_predict'] + df['rtt']
        # Compute target + rtt (same across group)
        df['target_plus_rtt'] = df['target'] + df['rtt']

        # Grouped computation
        grouped = df.groupby('dataset')
        result = grouped['nn_plus_rtt'].apply(find_peak_kde).reset_index()
        result.columns = ['dataset', 'peak_nn_plus_rtt']
        # Get one row per dataset to fetch target_plus_rtt
        target_rtt = grouped['target_plus_rtt'].first().reset_index()
        # Mean prediction per group
        mean_pred = grouped['nn_plus_rtt'].apply(find_mean).reset_index()
        mean_pred.columns = ['dataset', 'mean_nn']
        # Peak width per group
        wide = grouped['nn_plus_rtt'].apply(peak_wide).reset_index()
        wide.columns = ['dataset', 'wide_nn']

        # Merge all results
        result = result.merge(target_rtt, on='dataset')
        result = result.merge(mean_pred, on='dataset')
        result = result.merge(wide, on='dataset')
        # Compute absolute errors
        result['peak_diff'] = np.abs(result['peak_nn_plus_rtt'] - result['target_plus_rtt'])
        result['mean_diff'] = np.abs(result['mean_nn'] - result['target_plus_rtt'])

        # Final output
        return result[['dataset', 'peak_nn_plus_rtt', 'mean_nn', 'wide_nn', 'target_plus_rtt', 'peak_diff', 'mean_diff']]
    
def performance_check():
    """
    Function for printing model accurecy (based on all data tetsing datam that was used during model training and testing)

    return parameters:
        avg peak wide - average (for all folds) wide of distribution
        avg peak-target - average (for all folds) accurecy of peak = peak-target
        avg mean-target - average (for all folds) accurecy of mean = mean-target

        max peak wide - max (for all folds) wide of distribution
        max peak-target - max (for all folds) accurecy of peak = peak-target
        max mean-target - max (for all folds) accurecy of mean = mean-target
    """
    hard_code = '/hdd/gregory/range-estimation/output/purest_experiment/nfiles' #hb_complex_research
    # flist = ['MT_conv5_1multikernel_att_real_Delay_n_files_500']
    flist = [f for f in os.listdir(hard_code) if os.path.isdir(os.path.join(hard_code, f))]
    for f in flist:
        if f == 'retry': continue
        df = target_and_peak_comp(path_in_output=os.path.join(hard_code, f))
        print(f"For file {f}: \navg peak wide = {np.mean(df['wide_nn'])}\navg peak-target = {np.mean(df['peak_diff'])}\navg mean-target = {np.mean(df['mean_diff'])}")
        print(f"max peak wide = {np.max(df['wide_nn'])}\nmax peak-target = {np.max(df['peak_diff'])}\nmax mean-target = {np.max(df['mean_diff'])}")

def main():
    """
    Example how to check any model on real data
    """
    run_name = 'test' # needed, as additional name to result pictures files
    project_name = "lol"  # name and path to directory with model. It will save pictures with results into it
    path =  os.path.join(PATH, "output", project_name)

    # You should check models layer structure and add custom layers as below if needed:

    # model = load_model(
    #     os.path.join(FPATH, 'model_0.keras'), # model_0.keras model_finetuned
    #     custom_objects={'CovarianceLikeLayer': CovarianceLikeLayer,
    #                     'Compute_fft': Compute_fft, 
    #                     'ExpandDim1Layer': ExpandDim1Layer}
    # )

    model = load_model(
        os.path.join(path, 'model_finetuned.keras'), # model_0.keras model_finetuned
        custom_objects={"PositionalEncodingLayer": PositionalEncodingLayer,
            "TransformerBlock": TransformerBlock,
            "FFTFeaturesLayer": FFTFeaturesLayer}
    )

    # model = load_model(os.path.join(FPATH,'model_0.h5')) # simple loading
    
    model.summary() # here you can see if all is right

    # Finally, start the testing model on real data
    # You should change RangeDataUnpacker and RangeDataConstructor if needed (it should correspond to data on whitch it was trained)
    # Also you change fold_n_files to any other folding method
    check_model_on_real(model, fold_n_files, RangeDataUnpacker, RangeDataConstructor, path, run_name)

if __name__ == "__main__":
    performance_check()
    # main()