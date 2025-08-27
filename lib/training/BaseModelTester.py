import os
from typing import Optional

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
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
# from tensorflow.keras.models import load_model # type: ignore
from sklearn.utils import shuffle
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import tensorflow as tf
import pandas as pd
import math

from lib import PATH
OUTPUT = os.path.join(PATH, 'output')

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.get_logger().setLevel('ERROR')

from lib.pipelines import BDC, BDU
from lib.utils.plotting import plot_learning_curves 
from lib.utils.tools import make_unique_output_dir

class BaseModelTester:
    """
    Class for testing models by different data, models arch e.t.c.
    """
    def __init__(
            self, 
            folds:list, 
            df:pd.DataFrame, 
            create_model_fn, 
            name:str,
            input_shape:np.ndarray=(2,256), 
            name_list:list = None, 
            output_dir:str=OUTPUT, 
            max_retries=3, 
            epochs=100, 
            aug_params: Optional[dict] = None
            ):
        self.folds = folds
        self.df = df
        self.name = name
        self.name_list = name_list
        self.create_model = create_model_fn
        self.input_shape = input_shape
        self.max_retries = max_retries
        self.epochs = epochs
        self.output_dir = os.path.join(output_dir, f'MT_{name}_{epochs}')
        self.aug_params = aug_params

        self.output_dir = make_unique_output_dir(self.output_dir)
        print(f"Output directory: {self.output_dir}")

        self.results_df = pd.DataFrame(columns=["dataset", "nn_predict", "target", "slope", "music", "rtt"])

    def is_model_stuck(self, predictions, threshold=0.01):
        """
        Simple check if predictions is not zero
        """
        print(f"DEBUG: std(predictions): {np.std(predictions)}")
        return np.std(predictions) < threshold

    def val_loss_stuck(self, history, threshold=0.0001):        
        """
        Simple check if predictions not become zero
        """
        if history is None: return False
        val_losses = history.history.get('val_loss', [])
        print(f"DEBUG: len(val_losses):{len(val_losses)}\nstd(val_losses[-5:]){np.std(val_losses[-5:])}")
        return len(val_losses) >= 5 and np.std(val_losses[-5:]) < threshold
        
    def _plot_preprocessing(self):
        """
        Some actions for prepairing plots
        """
        # Determine optimal grid size
        num_folds = len(self.folds)
        cols = min(4, num_folds)  # At most 4 columns
        rows = math.ceil(num_folds / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # 4x4 inches per plot
        axes = np.array(axes).reshape(-1) if num_folds > 1 else [axes]  # Flatten safely for all cases
        return fig, axes, num_folds
    
    def _iter_name(self, iter_idx: int) -> str:
        """
        Some simple naming of folds
        """
        if self.name_list is not None: name = self.name_list[iter_idx]
        else: name = iter_idx
        print(f"Running fold: {name}")
        return name

    def model_fit(self, train_ds, test_ds):
        model = self.create_model(self.input_shape)

        es = EarlyStopping(monitor="loss", patience=30, restore_best_weights=True, verbose=0, min_delta=0.0005) # val_loss

        history = model.fit(
            train_ds, 
            validation_data=test_ds, 
            epochs=self.epochs,
            batch_size=64, 
            callbacks=[es], 
            verbose=1
            )
        return history, model
    
    def model_pred(self, model, X_test, optional_data = None):
        y_pred = model.predict(X_test).flatten() # MAIN RESULT
        # y_pred_val = model.predict(X_val).flatten()
        print(len(y_pred))
        return y_pred
    
    def model_save(self, model, name):
        self.output_model_fpath = os.path.join(self.output_dir, f'model_{name}.keras')
        model.save(self.output_model_fpath)

    # example. Change to your purposes
    def run(self, fold_unpacking=None, post_processing=None, model_save_flag=False, optional_data = None):
        """
        MAIN LOOP for training and giving output results (model performance)
        """
        if fold_unpacking is None: 
            print("You must give function for interpretate fold data to traing and test data for model")
            return "Error"
        elif post_processing is None:
            print("You must give function for interpretate results")
            return "Error"

        for fold_idx, (train_ids, test_ids) in enumerate(self.folds): # Here is start of main loop
            name = self._iter_name(fold_idx)
            
            Unpacker:BDU = fold_unpacking(self.df, train_ids, test_ids, self.aug_params, optional_data)
            data, additional = Unpacker.process()
            train_ds, test_ds, X_test, y_test, X_val, y_val = data

            for retry in range(self.max_retries): # Loop for check if model stucked some how during current iteration of training. If stucked -> retry
                print(f"Attempt {retry+1}...")
                K.clear_session() # we do not want to flood models  by previos training attemptions
                history, model = self.model_fit(train_ds, test_ds)
                y_pred = self.model_pred(model, X_test, optional_data=y_val)
                if self.is_model_stuck(y_pred) or self.val_loss_stuck(history): # stuck chek
                    print("Model stuck or loss stagnant. Retrying...")
                    continue
                break
            else:
                print(f"Failed on {name} after {self.max_retries} retries")
                continue
            # y_pred = y_test+np.random.normal(0,0.2,y_test.shape) # true test
            if history is not None: plot_learning_curves(history, model_name=self.name+f'_{name}', save_dir=self.output_dir)
            Constructor:BDC = post_processing(name, y_pred, data, additional, optional_data) 
            mae, results_i, flag_res_check = Constructor.process() # flag_res_check = slope and music presented in results

            self.results_df = pd.concat([self.results_df, results_i], ignore_index=True)
            print(self.results_df.head(2))

            if model_save_flag == True: 
                self.model_save(model, name)
            del model

        self._save_df()

    def _save_df(self):
        self.results_df.to_pickle(
            os.path.join(self.output_dir, f"model_results.pkl")
            )  # Pickle format (for fast saving/loading)

    def _plot_combined_ecdf(self, nn, slope, music, flag_res_check, file_name="combined_ecdf.png"):
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
        # ax.set_xlim([0, max([ecdf.x.max() for ecdf in ecdfs])])
        ax.set_xlim([0, 10])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, file_name), dpi=300, bbox_inches='tight')

    def get_results_df(self) -> pd.DataFrame:
        return self.results_df
    
    def get_model_fpath(self) -> str:
        return self.output_model_fpath