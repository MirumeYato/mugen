
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Layer, Normalization, Concatenate, GlobalAveragePooling1D, Input, Conv2D, Conv1D, Flatten, Dense, Dropout, MaxPool1D, BatchNormalization, Reshape, SeparableConv2D, GlobalAveragePooling2D, Activation # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts # type: ignore

class ExpandDim1Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return tf.expand_dims(x, axis=-1)

# Model definition
def oneway_model(input_shape=(2,256)):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(32, (2, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),

        Flatten(),
        Dense(64, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def optimal_model_IR(input_shape):
    """
    Incedent + Reflected waves 
    """
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(32, (4, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),

        Flatten(),
        Dense(64, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def new_model_IR(input_shape):
    """
    Incedent + Reflected waves 
    """
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(128, (4, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003),
        # tf.keras.layers.BatchNormalization(),
        ),
        Dropout(0.1),
        
        Conv2D(256, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003),
        # tf.keras.layers.BatchNormalization(),
        ),
        Dropout(0.1),

        Flatten(),
        Dense(256, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def full_vec_filter_model_IR(input_shape):
    """
    Incedent + Reflected waves 
    """
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(32, (1, 256), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)
               ),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)
               ),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),

        Flatten(),
        Dense(64, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def oneway_BIG_model(input_shape):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(32, (2, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        Dropout(0.1),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def Dense_model(input_shape):
    """
    Dense-only model for CSI (2, 256, 1) input: real + imag parts.
    """
    tf.keras.backend.clear_session()
    
    model = Sequential([
        Input(shape=input_shape),          # (256, 2)
        Flatten(),                         # → (512,)

        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dropout(0.1),

        Dense(64, activation='relu'),
        Dropout(0.1),

        Dense(1, activation='relu')        # output distance or delay (positive)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),  # good for outliers
        metrics=['mae']
    )
    
    return model

def delay_BIG_model(input_shape):
    """
    Incedent + Reflected waves 
    """
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        ExpandDim1Layer(),
        Conv2D(32, (4, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),
        
        Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        # tf.keras.layers.BatchNormalization(),
        Dropout(0.1),

        Flatten(),
        Dense(128, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(64, activation='relu'),
            #   kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.1),
        Dense(1, activation='relu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
        # loss='mse'
    )
    return model

def marat_model(input_shape):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.BatchNormalization(axis=1),
        Conv1D(filters=16, kernel_size=32, strides=1, padding="valid", activation="relu", data_format="channels_last"),
        MaxPool1D(2, data_format="channels_last"),
        Conv1D(filters=32, kernel_size=16, strides=1, padding="valid", activation="relu", data_format="channels_last"),
        MaxPool1D(2, data_format="channels_last"),
        Conv1D(filters=64, kernel_size=8, strides=1, padding="valid", activation="relu", data_format="channels_last"),
        MaxPool1D(2, data_format="channels_last"),
        Conv1D(filters=64, kernel_size=4, strides=1, padding="valid", activation="relu", data_format="channels_last"),
        MaxPool1D(2, data_format="channels_last"),
        Conv1D(filters=64, kernel_size=2, strides=1, padding="valid", activation="relu", data_format="channels_last"),
        MaxPool1D(2, data_format="channels_last"),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(1, activation="relu")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
        # loss=tf.keras.losses.Huber(),
        # metrics=['mae']
        # loss='mse'
    )
    return model

from tensorflow.keras.models import Model  # type: ignore

# ---------- 1. Custom preprocessing layer ----------
class HankelCovLayer(tf.keras.layers.Layer):
    def __init__(self, Ncov=2, validScIndex=None, **kwargs):
        super().__init__(**kwargs)
        self.Ncov = Ncov
        self.validScIndex = list(validScIndex) if validScIndex is not None else []

    def call(self, inputs):
        validScIndex_tensor = tf.constant(self.validScIndex, dtype=tf.int32)

        real = inputs[:, 0, :]
        imag = inputs[:, 1, :]
        complex_vec = tf.complex(real, imag)

        complex_valid = tf.gather(complex_vec, validScIndex_tensor, axis=1)
        M = tf.shape(complex_valid)[1]

        c = complex_valid[:, :self.Ncov]
        r = complex_valid[:, self.Ncov - 1:]

        def build_hankel(c_i, r_i):
            vals = tf.concat([c_i, r_i[1:]], axis=0)
            Y = tf.stack([vals[i:i + M - self.Ncov + 1] for i in range(self.Ncov)], axis=0)
            return Y

        Y_batch = tf.map_fn(lambda x: build_hankel(x[0], x[1]), (c, r), dtype=tf.complex64)
        Y_H_batch = tf.transpose(tf.math.conj(Y_batch), perm=[0, 2, 1])
        Ry_batch = tf.matmul(Y_batch, Y_H_batch) / tf.cast(tf.shape(Y_batch)[2], tf.complex64)

        Ry_real_imag = tf.stack([tf.math.real(Ry_batch), tf.math.imag(Ry_batch)], axis=-1)
        return Ry_real_imag

    def get_config(self):
        config = super().get_config()
        config.update({
            "Ncov": self.Ncov,
            "validScIndex": self.validScIndex  # now it's a Python list, JSON-serializable
        })
        return config
    
def Hankel_BIG_model(input_shape, Ncov=100, validScIndex=np.arange(6, 251)):
    tf.keras.backend.clear_session()
    
    inputs = Input(shape=input_shape)  # shape: (2, 256)
    x = HankelCovLayer(Ncov=Ncov, validScIndex=validScIndex)(inputs)  # → (Ncov, Ncov, 2)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )

    return model

class CovarianceLikeLayer(tf.keras.layers.Layer):
    def __init__(self, L=64, **kwargs):
        super().__init__(**kwargs)
        self.L = L

    def call(self, inputs):
        # inputs: (batch, 2, 256)
        real = inputs[:, 0, :]  # (batch, 256)
        imag = inputs[:, 1, :]  # (batch, 256)
        x = tf.complex(real, imag)  # (batch, 256)
        
        # Build Hankel matrix Y manually in a sliding window way
        Ys = []
        for i in range(self.L):
            Ys.append(x[:, i:i + (256 - self.L + 1)])  # shape (batch, 256 - L + 1)
        Y = tf.stack(Ys, axis=1)  # (batch, L, 256 - L + 1)

        Y_H = tf.transpose(tf.math.conj(Y), perm=[0, 2, 1])
        Ry = tf.matmul(Y, Y_H) / tf.cast(tf.shape(Y)[-1], tf.complex64)  # (batch, L, L)

        return tf.concat([tf.math.real(Ry), tf.math.imag(Ry)], axis=-1)  # (batch, L, 2L)    

class Compute_fft(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        real = x[:, 0, :]
        imag = x[:, 1, :]
        fft = tf.signal.fft(tf.complex(real, imag))
        mag = tf.abs(fft)
        phase = tf.math.angle(fft)
        fft_feat = tf.stack([mag, phase], axis=-1)  # (batch, 256, 2)
        fft_feat = tf.expand_dims(fft_feat, axis=1)  # (batch, 1, 256, 2)
        return fft_feat
        
def hybrid_model(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # Learn covariance-like features
    cov = CovarianceLikeLayer(L=128)(inp)
    # cov = Flatten()(cov)
    cov = GlobalAveragePooling1D()(cov)  # Instead of Flatten
    cov = Dense(64, activation='relu')(cov)

    if use_fft: 
        fft_feat = Compute_fft()(inp)
        fft_feat = GlobalAveragePooling2D()(fft_feat)
        combined = Concatenate(axis=-1)([x, cov, fft_feat]) #tf.concat([x, cov, fft_feat], axis=-1)
    else:
        combined = Concatenate(axis=-1)([x, cov])

    out = Dense(128, activation='relu')(combined)
    out = Dropout(0.1)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(1, activation='relu')(out)

    model = tf.keras.Model(inputs=inp, outputs=out)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='log_cosh', metrics=["mae"]) # tf.keras.losses.Huber()
    return model

from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention # type: ignore

class PositionalEncodingLayer(Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.pos_encoding = self.get_positional_encoding(sequence_length, embed_dim)

    def get_positional_encoding(self, seq_len, d_model):
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]  # cast to float here

        angle_rates = 1 / tf.pow(10000.0, (2 * tf.floor(i / 2)) / d_model)
        angle_rads = pos * angle_rates

        # Interleave sines and cosines
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        return {
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        }

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        }

class FFTFeaturesLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        real = x[:, 0, :]
        imag = x[:, 1, :]
        fft = tf.signal.fft(tf.complex(real, imag))
        mag = tf.abs(fft)
        phase = tf.math.angle(fft)
        return tf.stack([mag, phase], axis=-1)  # (batch, 256, 2)

    def get_config(self):
        return {}


def transformer_model(input_shape=(2, 256), depth=4, heads=4, embed_dim=64, ff_dim=128):
    tf.keras.backend.clear_session()
    inp = Input(shape=input_shape)  # (2, 256)

    # Apply FFT and extract [mag, phase]
    fft_feats = FFTFeaturesLayer()(inp)  # (batch, 256, 2)
    x = Dense(embed_dim)(fft_feats)  # project (batch, 256, embed_dim)

    x = PositionalEncodingLayer(sequence_length=256, embed_dim=embed_dim)(x)

    for _ in range(depth):
        x = TransformerBlock(embed_dim=embed_dim, num_heads=heads, ff_dim=ff_dim)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="relu")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="log_cosh", metrics=["mae"])
    return model