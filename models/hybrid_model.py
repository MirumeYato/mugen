import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Layer, Add, Concatenate, GlobalAveragePooling1D, Input, Conv2D, Conv1D, Flatten, Dense, Dropout, MaxPool1D, BatchNormalization, Reshape, SeparableConv2D, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

class ExpandDim1Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return tf.expand_dims(x, axis=-1)

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
        
def hb_orig_model(input_shape=(2, 256), use_fft=True):
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

def hb_cov_only(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Learn covariance-like features
    cov = CovarianceLikeLayer(L=128)(inp)
    cov = GlobalAveragePooling1D()(cov)  # Instead of Flatten
    cov = Dense(64, activation='relu')(cov)

    out = Dense(128, activation='relu')(cov)
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

def hb_fft_only(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    fft_feat = Compute_fft()(inp)
    fft_feat = GlobalAveragePooling2D()(fft_feat)

    out = Dense(128, activation='relu')(fft_feat)
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

def hb_cov_cnn(input_shape=(2, 256), use_fft=True):
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

def hb_fft_cnn(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    fft_feat = Compute_fft()(inp)
    fft_feat = GlobalAveragePooling2D()(fft_feat)
    combined = Concatenate(axis=-1)([x, fft_feat]) #tf.concat([x, cov, fft_feat], axis=-1)
    
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

def hb_cov_fft(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Learn covariance-like features
    cov = CovarianceLikeLayer(L=128)(inp)
    # cov = Flatten()(cov)
    cov = GlobalAveragePooling1D()(cov)  # Instead of Flatten
    cov = Dense(64, activation='relu')(cov)

    fft_feat = Compute_fft()(inp)
    fft_feat = GlobalAveragePooling2D()(fft_feat)
    combined = Concatenate(axis=-1)([cov, fft_feat]) #tf.concat([x, cov, fft_feat], axis=-1)

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

# =========================================================================================
# CNN testing

# def hb_cov_flat(input_shape=(2, 256), use_fft=True):
#     tf.keras.backend.clear_session()

#     inp = Input(shape=input_shape)  # (2, 256)

#     # Low-level CNN on raw data
#     expanded = ExpandDim1Layer()(inp)
#     x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
#     x = BatchNormalization()(x)
#     x = Conv2D(64, (1, 5), activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = GlobalAveragePooling2D()(x)

#     # Learn covariance-like features
#     cov = CovarianceLikeLayer(L=128)(inp)
#     # cov = Flatten()(cov)
#     cov = Flatten()(cov)  # instead of GAP
#     cov = Dense(128, activation='relu')(cov)

#     if use_fft: 
#         fft_feat = Compute_fft()(inp)
#         fft_feat = GlobalAveragePooling2D()(fft_feat)
#         combined = Concatenate(axis=-1)([x, cov, fft_feat]) #tf.concat([x, cov, fft_feat], axis=-1)
#     else:
#         combined = Concatenate(axis=-1)([x, cov])

#     out = Dense(128, activation='relu')(combined)
#     out = Dropout(0.1)(out)
#     out = Dense(64, activation='relu')(out)
#     out = Dropout(0.1)(out)
#     out = Dense(1, activation='relu')(out)

#     model = tf.keras.Model(inputs=inp, outputs=out)

#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=1e-3,
#         decay_steps=1000,
#         decay_rate=0.96,
#         staircase=True,
#     )
#     optimizer = Adam(learning_rate=lr_schedule)

#     model.compile(optimizer=optimizer, loss='log_cosh', metrics=["mae"]) # tf.keras.losses.Huber()
#     return model

def hb_conv5(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_opt(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_opt_f(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_opt_f_norm(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 1), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv_opt(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(expanded)
    x = Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv_opt_flat(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(expanded)
    x = Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)

    out = Dense(128, activation='relu')(x)
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









############################################################




def hb_conv5_Huber(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

def hb_conv5_no_norm(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_flat(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_3layers(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = Conv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 3), activation='relu')(x)  # NEW layer
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_separable(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    x = SeparableConv2D(32, (2, 5), activation='relu', padding='valid')(expanded)
    x = BatchNormalization()(x)
    x = SeparableConv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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

def hb_conv5_1multikernel(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    conv1 = Conv2D(32, (2, 3), activation='relu', padding='same')(expanded)
    conv2 = Conv2D(32, (2, 5), activation='relu', padding='same')(expanded)
    conv3 = Conv2D(32, (2, 7), activation='relu', padding='same')(expanded)
    x = Concatenate()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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


def deep_rescnn_model(input_shape=(2, 256)):
    inp = tf.keras.Input(shape=input_shape)
    x = ExpandDim1Layer()(inp)  # (batch, 2, 256, 1)

    # Block 1
    y = Conv2D(32, (2, 5), padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = Conv2D(32, (2, 3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    x1 = Add()([x, y])  # Residual connection

    # Block 2
    y = Conv2D(64, (2, 3), padding='same', activation='relu')(x1)
    y = BatchNormalization()(y)
    y = Conv2D(64, (2, 3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    shortcut = Conv2D(64, (1, 1), padding='same')(x1)  # Match channel dims
    x2 = Add()([shortcut, y])

    # Final Conv
    x2 = Conv2D(128, (2, 3), padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling2D()(x2)

    x2 = Dense(128, activation='relu')(x2)
    x2 = Dropout(0.1)(x2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dropout(0.1)(x2)
    out = Dense(1)(x2)

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

class AttentionFusionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = None  # We'll initialize it in `build` based on number of inputs

    def build(self, input_shape):
        num_inputs = len(input_shape)
        self.num_inputs = num_inputs
        self.concat_channels = sum([shape[-1] for shape in input_shape])
        self.global_pool = GlobalAveragePooling2D()
        self.dense = Dense(num_inputs, activation='softmax')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: list of tensors with shape (B, H, W, C)
        concat = Concatenate(axis=-1)(inputs)  # shape: (B, H, W, C_total)
        pooled = self.global_pool(concat)      # shape: (B, C_total) → flattened

        # Attention scores for each branch
        attention_scores = self.dense(pooled)  # shape: (B, num_inputs)

        # Apply attention to each input
        weighted = []
        for i in range(self.num_inputs):
            alpha = tf.expand_dims(tf.expand_dims(attention_scores[:, i], 1), 1)  # shape: (B, 1, 1)
            alpha = tf.expand_dims(alpha, axis=-1)  # shape: (B, 1, 1, 1)
            weighted_input = inputs[i] * alpha
            weighted.append(weighted_input)

        return Add()(weighted)  # shape: (B, H, W, C) — same as individual input

    def get_config(self):
        config = super().get_config()
        return config
    
def hb_conv5_1multikernel_att(input_shape=(2, 256), use_fft=True):
    tf.keras.backend.clear_session()

    inp = Input(shape=input_shape)  # (2, 256)

    # Low-level CNN on raw data
    expanded = ExpandDim1Layer()(inp)
    conv1 = Conv2D(32, (2, 3), activation='relu', padding='same')(expanded)
    conv2 = Conv2D(32, (2, 5), activation='relu', padding='same')(expanded)
    conv3 = Conv2D(32, (2, 7), activation='relu', padding='same')(expanded)
    x = AttentionFusionLayer()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Dense(128, activation='relu')(x)
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