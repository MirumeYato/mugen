import os

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
import tensorflow as tf
import pandas as pd

from lib import PATH
DATA_DIR = os.path.join(PATH, "data")

tf.config.optimizer.set_jit(True)

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.get_logger().setLevel('ERROR')

from lib.data import Data
from lib.pipelines import BDU
from models.model_configurations import FFTFeaturesLayer, PositionalEncodingLayer, TransformerBlock
from models.model_configurations import CovarianceLikeLayer, Compute_fft, ExpandDim1Layer

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

class BaseFineTune(ABC):
    def __init__(self, path, epochs, norm=False, aug_params=None, batch_size=32):
        self.df:pd.DataFrame = Data(norm = norm)('real')
        self.aug_params = aug_params
        self.path = path
        self.epochs = epochs
        self.batch_size = batch_size
    
    def __call__(self, unpacker_class, n_trainable_last):
        self.model = self.load_custom_model()
        self.freeze_layers(n_trainable_last)
        self.compile()
        train_ds, test_ds = self.get_data(unpacker_class)
        self.custom_fit(train_ds, test_ds)
        self.model.save(os.path.join(self.path, 'model_finetuned.keras'))
        
    def get_data(self, unpacker_class):
        train_ids, test_ids = self.data_split()
        Unpacker:BDU = unpacker_class(self.df, train_ids, test_ids, self.aug_params, None, batch_size = self.batch_size)
        data, __ = Unpacker.process()
        train_ds, test_ds, X_test, y_test, __, __ = data

        return (train_ds, test_ds)
    
    def freeze_layers(self, n_trainable_last=3):
        for layer in self.model.layers[:-n_trainable_last]:
            layer.trainable = False
    
    @abstractmethod
    def custom_fit(self, model, train_ds, test_ds):
        pass

    @abstractmethod
    def data_split(self):
        pass

    @abstractmethod
    def load_custom_model(self):
        pass

    @abstractmethod
    def compile(self):
        pass

class SimpleFineTune(BaseFineTune):
    def data_split(self):
        train_ids, test_ids = train_test_split(
            np.arange(len(self.df)),
            test_size=0.9, random_state=42, shuffle=True)
        return train_ids, test_ids
        
    def load_custom_model(self):
        # --- Load your model with custom objects ---
        model = load_model(
            os.path.join(self.path, 'model_0.keras'),
        )
        return model

    def compile(self):
        # --- Compile with lower learning rate for fine-tuning ---
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=tf.keras.losses.Huber(),
            metrics=['mae']
        )
    
    def custom_fit(self, train_ds, test_ds):
        # --- Fine-tune ---
        history = self.model.fit(
            train_ds, 
            validation_data=test_ds, 
            epochs=self.epochs,
            batch_size=self.batch_size, 
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0, min_delta=0.0005), # val_loss
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ], 
            verbose=1
            )

class ExtSimpleFineTune(SimpleFineTune):   
    def load_custom_model(self):
        # --- Load your model with custom objects ---
        model = load_model(
            os.path.join(self.path, 'model_0.keras'),
            custom_objects={"ExpandDim1Layer": ExpandDim1Layer}
        )
        return model

class CovarFineTune(SimpleFineTune):        
    def load_custom_model(self):
        # --- Load your model with custom objects ---
        model = load_model(
            os.path.join(self.path, 'model_0.keras'),
            custom_objects={"Compute_fft": Compute_fft,
                "ExpandDim1Layer": ExpandDim1Layer,
                "CovarianceLikeLayer": CovarianceLikeLayer}
        )
        return model
    
    def compile(self):
        # --- Compile with lower learning rate for fine-tuning ---
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='log_cosh',
            metrics=['mae']
        )
        
class TransformerFineTune(SimpleFineTune):        
    def load_custom_model(self):
        # --- Load your model with custom objects ---
        model = load_model(
            os.path.join(self.path, 'model_0.keras'),
            custom_objects={"PositionalEncodingLayer": PositionalEncodingLayer,
                "TransformerBlock": TransformerBlock,
                "FFTFeaturesLayer": FFTFeaturesLayer}
        )
        return model
    
    def compile(self):
        # --- Compile with lower learning rate for fine-tuning ---
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='log_cosh',
            metrics=['mae']
        )

