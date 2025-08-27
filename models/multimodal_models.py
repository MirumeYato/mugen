import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Normalization, Concatenate, GlobalAveragePooling1D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Activation # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# -------------------- Custom layers --------------------
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, LearningRateSchedule # type: ignore

class ComputeFFT(tf.keras.layers.Layer):
    """From (batch, 2, L) real/imag -> (batch, K, 2) mag+phase."""
    def __init__(self, take_k=None, **kwargs):
        super().__init__(**kwargs)
        self.take_k = take_k

    def call(self, x):
        # x: (batch, 2, L) => complex (batch, L)
        real = x[:, 0, :]
        imag = x[:, 1, :]
        c = tf.complex(real, imag)
        fft = tf.signal.fft(c)                        # (batch, L)
        if self.take_k is not None:
            fft = fft[:, :self.take_k]
        mag   = tf.abs(fft)
        phase = tf.math.angle(fft)
        return tf.stack([mag, phase], axis=-1)        # (batch, K or L, 2)

class ExpandDim(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, x):
        return tf.expand_dims(x, axis=self.axis)

class Squeeze(tf.keras.layers.Layer):
    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, x):
        return tf.squeeze(x, axis=self.axis)

# -------------------- Model factory --------------------

class X1X2FusionModel(tf.keras.Model):
    def __init__(
        self,
        L=256,
        k_fft=None,
        l2w=3e-3,
        dropout=0.10,
        loss="huber",
        metrics=("mae",),
        learning_rate=3e-4,#1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.L = L
        self.k_fft = k_fft
        self.loss_cfg = loss
        self.metrics_cfg = metrics
        self.dropout_rate = dropout

        reg = tf.keras.regularizers.l2(l2w)

        # --- X1 branch ---
        self.x1_norm = Normalization(name="x1_norm")
        self.x1_dense1 = Dense(32, activation="relu", kernel_regularizer=reg)
        self.x1_dense2 = Dense(16, activation="relu", kernel_regularizer=reg)

        # --- X2 RAW branch ---
        self.x2_expand = ExpandDim(axis=-1, name="x2_expand_hw")
        self.x2_conv2d = Conv2D(
            32, (2, 1), padding="valid", activation=None, kernel_regularizer=reg
        )
        self.x2_bn1 = BatchNormalization()
        self.x2_relu1 = Activation("relu")
        self.x2_squeeze = Squeeze(axis=1, name="squeeze_height")
        self.x2_conv1d = Conv1D(
            64, 5, padding="same", activation=None, kernel_regularizer=reg
        )
        self.x2_bn2 = BatchNormalization()
        self.x2_relu2 = Activation("relu")
        self.x2_gap1 = GlobalAveragePooling1D()

        # --- X2 FFT branch ---
        self.fft = ComputeFFT(take_k=k_fft, name="fft_mag_phase")
        self.fft_conv1d = Conv1D(
            32, 5, padding="valid", activation=None, kernel_regularizer=reg
        )
        self.fft_bn = BatchNormalization()
        self.fft_relu = Activation("relu")
        self.fft_gap = GlobalAveragePooling1D()

        # --- Fusion head ---
        self.concat = Concatenate()
        self.head_dense1 = Dense(64, activation="relu", kernel_regularizer=reg)
        self.head_dropout = Dropout(dropout)
        self.head_dense2 = Dense(32, activation="relu", kernel_regularizer=reg)
        self.out_layer = Dense(1, name="y")
       # Save optimizer config for later compilation
        self.learning_rate = learning_rate

    def adapt_normalization(self, train_ds):
        """Adapt the x1 Normalization layer on training dataset."""
        x1_only_ds = train_ds.map(lambda inputs, y: tf.cast(inputs[0], tf.float32))
        self.x1_norm.adapt(x1_only_ds)

    def call(self, inputs, training=False):
        x1_in, x2_in = inputs

        # --- X1 branch ---
        b1 = self.x1_norm(x1_in)
        b1 = self.x1_dense1(b1)
        b1 = self.x1_dense2(b1)

        # --- X2 RAW branch ---
        b2 = self.x2_expand(x2_in)
        b2 = self.x2_conv2d(b2)
        b2 = self.x2_bn1(b2, training=training)
        b2 = self.x2_relu1(b2)
        b2 = self.x2_squeeze(b2)
        b2 = self.x2_conv1d(b2)
        b2 = self.x2_bn2(b2, training=training)
        b2 = self.x2_relu2(b2)
        b2 = self.x2_gap1(b2)

        # --- X2 FFT branch ---
        b3 = self.fft(x2_in)
        b3 = self.fft_conv1d(b3)
        b3 = self.fft_bn(b3, training=training)
        b3 = self.fft_relu(b3)
        b3 = self.fft_gap(b3)

        # --- Fusion head ---
        h = self.concat([b1, b2, b3])
        h = self.head_dense1(h)
        h = self.head_dropout(h, training=training)
        h = self.head_dense2(h)
        y = self.out_layer(h)

        return y

    def compile_with_schedule(self, steps_per_epoch):
        """Compile with cosine decay schedule and chosen loss/metrics."""
        if self.loss_cfg == "huber":
            loss_obj = tf.keras.losses.Huber()
        elif self.loss_cfg == "mae":
            loss_obj = tf.keras.losses.MeanAbsoluteError()
        else:
            loss_obj = self.loss_cfg

        # lr_schedule = CosineDecayRestarts(
        #     initial_learning_rate=self.learning_rate,
        #     first_decay_steps=500*steps_per_epoch,
        #     t_mul=2,
        #     m_mul=1,
        #     alpha=1e-6
        # )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=3*steps_per_epoch,
            decay_rate=0.95,
            staircase=True,
        )

        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        # opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3, clipnorm=1.0)


        super().compile(optimizer=opt, loss=loss_obj, metrics=list(self.metrics_cfg))

def build_compile_x1x2_model(
    train_ds,                 # tf.data.Dataset yielding ((X1, X2), y)
    L=256,                    # CFR length for X2: shape (2, L)
    k_fft=None,               # take first K FFT bins (None -> use all L)
    l2w=3e-3,
    dropout=0.10,
    learning_rate=1e-3,
    loss="huber",             # "huber" | "mae" | tf.keras.losses.Huber(delta=...) | "log_cosh"
    metrics=("mae",),
):
    """
    Returns a compiled model. Adapts the X1 Normalization on train_ds internally.
    Assumes each dataset element looks like: ((X1_batch, X2_batch), y_batch)
      - X1_batch: (batch, 2)
      - X2_batch: (batch, 2, L)
    """
    # Create model
    model = X1X2FusionModel(L=L, k_fft=k_fft, l2w=l2w, dropout=dropout, loss=loss, metrics=metrics, learning_rate=learning_rate)

    # Adapt Normalization on train dataset
    model.adapt_normalization(train_ds)

    # 1) оцениваем steps_per_epoch (для tf.data.Dataset это работает)
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    print('steps_per_epoch: ', steps_per_epoch)

    # Compile
    model.compile_with_schedule(steps_per_epoch)
    
    return model
