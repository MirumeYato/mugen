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

# Path settings
import sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

OUTPUT = os.path.join(PATH, 'output')

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.get_logger().setLevel('ERROR')

from datetime import datetime
from lib.fold_tools import BaseDataConstructor, BaseDataUnpacker, RangeDataUnpacker, RangeDataConstructor, reduplicate
from lib.autoencoder import build_autoencode_model, build_Regression_model
# from lib.fold_tools import AutoencoderDataUnpacker
from lib.tools import Data

"""
AutoencoderTester: adapted from BaseModelTester
- Phase 1: pretraining CSI Autoencoder
- Phase 2: loading encoder for downstream tasks (regression, classification, etc.)
"""

from lib.BaseModelTester import BaseModelTester, plot_learning_curves, make_unique_output_dir
from lib.cross_validation import ModelTester
from datetime import datetime

class AutoencoderTester(BaseModelTester):
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
    def plot_test(self, autoencoder, X_val, fold_idx):
        # --- Evaluate reconstruction on validation set ---
        X_val_pred = autoencoder.predict(X_val, batch_size=64)
        recon_errors = np.mean(np.square(X_val - X_val_pred), axis=(1, 2))  # shape: (N_val,)

        # Save or log metrics
        print(f"[Fold {fold_idx}] Validation MSE: {np.mean(recon_errors):.6f} ± {np.std(recon_errors):.6f}")

        # Plot histogram of reconstruction error
        plt.figure(figsize=(6, 4))
        sns.histplot(recon_errors, bins=50, kde=True)
        plt.title(f"Reconstruction error (fold {fold_idx})")
        plt.xlabel("MSE per sample")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"recon_error_fold{fold_idx}.png"))
        plt.close()

        # ECDF
        ecdf = ECDF(recon_errors)
        plt.figure()
        plt.plot(ecdf.x, ecdf.y, label='ECDF')
        plt.axhline(0.9, color='r', linestyle='--', label='90% threshold')
        plt.xlabel('MSE')
        plt.ylabel('Cumulative Probability')
        plt.title(f"ECDF of Reconstruction Error (fold {fold_idx})")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"ecdf_fold{fold_idx}.png"))
        plt.close()

    def run(self, fold_unpacking, b_size):
        for fold_idx, (train_ids, test_ids) in enumerate(self.folds):
            print(f"[Fold {fold_idx}] Starting training")
            name = f"fold{fold_idx}"

            Unpacker: BaseDataUnpacker = fold_unpacking(self.df, train_ids, test_ids)
            data, _ = Unpacker.process()
            X_train, _, X_test, _, X_val, _ = data

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(len(X_train)).batch(b_size)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val)).batch(b_size)

            K.clear_session()
            autoencoder = self.create_model(self.input_shape)
            es = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

            # tf.profiler.experimental.start('logdir')
            history = autoencoder.fit(train_ds, validation_data=val_ds,
                                      epochs=self.epochs, verbose=1,
                                      batch_size=b_size, callbacks=[es])
            # tf.profiler.experimental.stop()

            plot_learning_curves(history, model_name=f"{self.name}_fold{fold_idx}", save_dir=self.output_dir)
            self.plot_test(autoencoder, X_val, fold_idx)

            # Save encoder
            if hasattr(autoencoder, 'encode'):
                encoder_output = autoencoder.encode(X_test)
                np.save(os.path.join(self.output_dir, f"z_fold{fold_idx}.npy"), encoder_output)
                print(f"[✓] Saved encoder output for fold {fold_idx} to {self.output_dir}")
            else:
                print("[!] Autoencoder does not support encode() method. Skipping encoder saving.")

            # Save full model (optional)
            autoencoder.save_weights(os.path.join(self.output_dir, f"autoencoder_fold{fold_idx}.weights.h5"))
            print(f"[✓] Saved autoencoder model weights.")

class RegressionTester(ModelTester):
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

    def model_fit(self, train_ds, test_ds):
        model, unfreeze_cb = self.create_model(self.input_shape, '/hdd/gregory/range-estimation/output/MT_autoencoder_pretrain_100_ver250806_1049_synth')

        es = EarlyStopping(monitor="loss", patience=30, restore_best_weights=True, verbose=0, min_delta=0.0005) # val_loss

        history = model.fit(
            train_ds, 
            validation_data=test_ds, 
            epochs=self.epochs,
            batch_size=64, 
            callbacks=[es, unfreeze_cb], 
            verbose=1
            )
        return history, model

# Example usage
def run_autoencoder_pretraining():
    df = Data()('synth_v3')  # real or 'synth_v3'
    from lib.fold_tools import fold_1simple
    folds = fold_1simple(df)

    tester = AutoencoderTester(
        folds=folds,
        df=df,
        create_model_fn=build_autoencode_model,
        name="autoencoder_pretrain",
        epochs=100
    )

    tester.run(fold_unpacking=RangeDataUnpacker, b_size = 64)



from lib.fold_tools import fold_1simple, extract_sort_key

def run_regression(run_name, create_model, unpack, pack, create_folding, name_list, flag):
    eph = 100

    # Here you cna choose data for train and test
    df:pd.DataFrame = Data()('synth_v3') # real synth_v3

    # here you choose folding method
    if flag: folds = create_folding(df, extract_sort_key) # extract_sort_key print(len(folds),len(folds[0]),len(folds[0][1]),len(folds[0][0]))
    else: folds = create_folding(df)
    
    # here we define model tester
    tester = RegressionTester(
        folds=folds,
        df=df,
        name_list=name_list, # needs if you want correct naming incide folding plots
        create_model_fn=create_model, # your model configuration
        input_shape=(2, 256), # All models should work with this shape. If not fix shape incide create_model function (NOT HERE!)
        name=run_name, # output directory name
        epochs=eph,
        aug_params=None#{"num_points": 1, "mu": 0, "sigma": 0.2} # if you do not want to do augmentation choose None in opposite use {"num_points": 10, "mu": 0, "sigma": 0.02}
    )

    print(f"DEBUG: model parameters are:\npath = {tester.output_dir},\nepochs = {tester.epochs}")
    os.makedirs(tester.output_dir, exist_ok=True)

    tester.run(fold_unpacking=unpack, post_processing = pack, model_save_flag=True) # main loop of training
    print(tester.get_model_fpath())

def main():
    name_list1 = np.array(['LoS_no_human', 'LoS_is_human', 'NLoS'],
        dtype=object)
    name_list3 = np.array([
        'LoS_NoHuman0p35m', 'LoS_NoHuman2m', 'LoS_NoHuman5m', 'LoS_NoHuman8m',
        'LoS_NoHuman10m', 'LoS_NoHuman13m', 'LoS_NoHuman15m', 'LoS_NoHuman18m', 
        'LoS_IsHuman3m','LoS_IsHuman5m', 'LoS_IsHuman8m', 'LoS_IsHuman12m',
        'NLoS8m', 'NLoS9m', 'NLoS10m', 'NLoS11m'],
        dtype=object)
    # name_list = [None, name_list1, None, name_list3] # correct names of foldings for plots
    # flag_list = [False, False, False, True] # flag for using sorting in fold for n_files folding
    name_list = [None] # correct names of foldings for plots
    flag_list = [False] # flag for using sorting in fold for n_files folding

    for i, cm in enumerate([fold_1simple]): #fold_1simple, fold_3groups, fold_10euqal, fold_n_files # loop for folding metheds
        for model_arch in [build_Regression_model]: # loop for model configurations
            for unpack_i in [RangeDataUnpacker]: # loop unpacking types
                run_regression(model_arch.__name__[3:]+'_real_'+unpack_i.__name__[:-12]+'_'+cm.__name__[5:], model_arch, unpack_i, RangeDataConstructor, cm, name_list[i], flag_list[i])
            # for unpack_i in [DelayDataUnpacker]: # loop unpacking types
            #     tester(model_arch.__name__[3:]+'_real_'+unpack_i.__name__[:-12]+'_'+cm.__name__[5:], model_arch, unpack_i, DelayDataConstructor, cm, name_list[i], flag_list[i])


if __name__ == '__main__':
    # run_autoencoder_pretraining()
    main()    