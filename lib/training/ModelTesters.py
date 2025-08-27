import os
from typing import Optional

from sklearn.metrics import mean_absolute_error

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
from sklearn.linear_model import LinearRegression
import pandas as pd

from lib import PATH
OUTPUT = os.path.join(PATH, 'output')

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.get_logger().setLevel('ERROR')

from lib.pipelines import BDC, BDU
from .BaseModelTester import BaseModelTester
from lib.utils.plotting import plot_learning_curves, plot_ecdf, plot_hist

class SimpleModelTester(BaseModelTester):
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
        super().__init__(
            folds=folds,
            df=df,
            create_model_fn=create_model_fn,
            name=name,
            input_shape=input_shape,
            name_list=name_list,
            output_dir=output_dir,
            max_retries=max_retries,
            epochs=epochs,
            aug_params=aug_params
        )

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
        all_mae_nn, all_mae_slope, all_mae_music, all_mae_nn_val, all_mae_sl_val, all_mae_mu_val = [], [], [], [], [], []
        fig, axes, num_folds = self._plot_preprocessing()
        fig_hist, axes_hist, __ = self._plot_preprocessing()

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

            # Now we get all results in needed for us format. You can define this func as you wish
            # mae_nn_val = np.abs(y_pred_val - y_val)
            # mu = self.df.loc[test_ids, 'music_est'].values
            # sl = self.df.loc[test_ids, 'slope_est'].values
            # mae_sl_val = np.abs(reduplicate(sl) - y_val)
            # mae_mu_val = np.abs(reduplicate(mu) - y_val)
            if history is not None: plot_learning_curves(history, model_name=self.name+f'_{name}', save_dir=self.output_dir)
            Constructor:BDC = post_processing(name, y_pred, data, additional, optional_data) 
            mae, results_i, flag_res_check = Constructor.process() # flag_res_check = slope and music presented in results
            mae_nn, mae_slope, mae_music = mae
            
            all_mae_nn.extend(mae_nn)
            # all_mae_nn_val.extend(mae_nn_val)
            # all_mae_sl_val.extend(mae_sl_val)
            # all_mae_mu_val.extend(mae_mu_val)
            if flag_res_check: 
                all_mae_slope.extend(mae_slope)
                all_mae_music.extend(mae_music)
            
            if flag_res_check:
                plot_ecdf([mae_nn, mae_slope, mae_music], name, axes[fold_idx])
                if any([x is None for  x in results_i["rtt"]]): 
                    plot_hist([y_pred, additional[0][0], additional[0][1]], name, axes_hist[fold_idx], results_i["target"])
                else:
                    plot_hist([y_pred+results_i["rtt"], additional[0][0], additional[0][1]], name, axes_hist[fold_idx], results_i["target"]+results_i["rtt"])
            else: 
                plot_ecdf([mae_nn], name, axes[fold_idx])
                if any([x is None for  x in results_i["rtt"]]): 
                    plot_hist([y_pred], name, axes_hist[fold_idx], results_i["target"])
                else:
                    plot_hist([y_pred+results_i["rtt"]], name, axes_hist[fold_idx], results_i["target"]+results_i["rtt"])

            self.results_df = pd.concat([self.results_df, results_i], ignore_index=True)
            print(self.results_df.head(2))

            if model_save_flag == True and model is not None: 
                self.model_save(model, name)
            del model

        for i in range(num_folds, len(axes)):
            fig.delaxes(axes[i])
        
        for i in range(num_folds, len(axes_hist)):
            fig_hist.delaxes(axes_hist[i])

        fig.suptitle("ECDFs per Fold")
        fig.tight_layout()
        if len(self.folds)>1: 
            fig.savefig(os.path.join(self.output_dir, "fold_ecdfs.png"), dpi=300, bbox_inches='tight')
            fig_hist.savefig(os.path.join(self.output_dir, "fold_hists.png"), dpi=300, bbox_inches='tight')

        self._plot_combined_ecdf(all_mae_nn, all_mae_slope, all_mae_music, flag_res_check)
        # self._plot_combined_ecdf(all_mae_nn_val, all_mae_sl_val, all_mae_mu_val, flag_res_check, file_name="validation.png")
        self._save_df()

class DummyTester(SimpleModelTester):
    """
    Class for testing models by different data, models arch e.t.c.

    Here are no model training. All predicts are:

    >>> y_pred = y_true + np.random.normal(0, 0.2, optional_data.shape)

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
        super().__init__(
            folds=folds,
            df=df,
            create_model_fn=create_model_fn,
            name=name,
            input_shape=input_shape,
            name_list=name_list,
            output_dir=output_dir,
            max_retries=max_retries,
            epochs=epochs,
            aug_params=aug_params
        )
    def is_model_stuck(self, predictions, threshold=0.01):
        return False
    
    def model_fit(self, train_ds, test_ds):
        return None, None
    
    def model_pred(self, model, X_test, optional_data = None):
        y_pred = optional_data+np.random.normal(0, 0.2, optional_data.shape) # true test
        print(len(y_pred))
        return y_pred            

class LinearTester(SimpleModelTester):      
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
        super().__init__(
            folds=folds,
            df=df,
            create_model_fn=create_model_fn,
            name=name,
            input_shape=input_shape,
            name_list=name_list,
            output_dir=output_dir,
            max_retries=max_retries,
            epochs=epochs,
            aug_params=aug_params
        )
    def model_fit(self, train_ds, test_ds):
        y1 = tf.concat([b[0] for b in train_ds], axis=0).numpy()
        y2 = tf.concat([b[1] for b in train_ds], axis=0).numpy()

        lin = LinearRegression().fit(y1.reshape(-1,1), y2)

        pred = lin.predict(y1.reshape(-1,1))
        print("MAE baseline (y2 ≈ a*y1+b):", mean_absolute_error(y2, pred))
        return None, lin
    
    def model_pred(self, model, X_test, optional_data = None):
        y_pred = model.predict(X_test.reshape(-1,1)).flatten() # MAIN RESULT
        # y_pred_val = model.predict(X_val).flatten()
        print(len(y_pred))
        return y_pred

class FusionTester(SimpleModelTester):      
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
        super().__init__(
            folds=folds,
            df=df,
            create_model_fn=create_model_fn,
            name=name,
            input_shape=input_shape,
            name_list=name_list,
            output_dir=output_dir,
            max_retries=max_retries,
            epochs=epochs,
            aug_params=aug_params
        )
    def model_fit(self, train_ds, test_ds):
        model = self.create_model(train_ds)

        es = EarlyStopping(monitor="loss", patience=30, restore_best_weights=True, verbose=0, min_delta=0.0005) # val_loss

        history = model.fit(
            train_ds, 
            validation_data=test_ds, 
            epochs=self.epochs,
            batch_size=32, 
            callbacks=[es], 
            verbose=1
            )
        return history, model
    
    def model_pred(self, model, X_test, optional_data = None):
        y_pred = model.predict((X_test[0],X_test[1])).flatten() # MAIN RESULT
        print(len(y_pred))
        return y_pred