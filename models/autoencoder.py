import os
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np

# ---------- Custom Layers ----------

class ComputeFFT(layers.Layer):
    def call(self, inputs):
        real = inputs[:, 0, :]
        imag = inputs[:, 1, :]
        x_complex = tf.complex(real, imag)
        fft = tf.signal.fft(x_complex)
        fft_mag = tf.abs(fft)
        return tf.expand_dims(fft_mag, axis=-1)

class CovarianceLayer(layers.Layer):
    def __init__(self, L=128):
        super().__init__()
        self.L = L

    def call(self, inputs):
        real = inputs[:, 0, :]
        imag = inputs[:, 1, :]
        x = tf.complex(real, imag)
        hankel = tf.stack([tf.roll(x, shift=-i, axis=-1) for i in range(self.L)], axis=1)
        cov = tf.matmul(hankel, hankel, adjoint_b=True) / tf.cast(self.L, tf.complex64)
        cov_real = tf.math.real(cov)
        return tf.expand_dims(cov_real, axis=-1)

class CovGraphBranch(layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv1D(32, 3, padding='same', activation='gelu')
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, cov_input):
        cov_matrix = tf.squeeze(cov_input, axis=-1)  # (None, L, L)
        eigvals, _ = tf.linalg.eigh(cov_matrix)      # (None, L)
        eigvals = tf.sort(eigvals, direction='DESCENDING')  # (None, L)
        eigvals = tf.expand_dims(eigvals, axis=-1)   # (None, L, 1)
        x = self.conv(eigvals)                       # (None, L, 32)
        x = self.pool(x)                             # (None, 32)
        return x


class FusionEncoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)
        self.proj = layers.Dense(64, activation='gelu')

    def call(self, fft_feat, cov_feat):
        x = tf.concat([fft_feat, cov_feat], axis=-1)
        x = tf.expand_dims(x, axis=1)
        x = self.attn(x, x)
        x = tf.squeeze(x, axis=1)
        return self.proj(x)

# ---------- Models ----------

class AutoencoderModel(Model):
    def __init__(self, input_shape=(2, 256), L=64):
        super().__init__()
        self.fft_layer = ComputeFFT()
        self.cov_layer = CovarianceLayer(L=L)
        self.fft_branch = tf.keras.Sequential([
            layers.Conv1D(32, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('gelu'),
            layers.GlobalAveragePooling1D()
        ])
        self.cov_branch = CovGraphBranch()
        self.encoder = FusionEncoder()
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='gelu'),
            layers.Dense(np.prod(input_shape), activation=None),
            layers.Reshape(input_shape)
        ])

    def call(self, inputs):
        fft_input = self.fft_layer(inputs)
        cov_input = self.cov_layer(inputs)
        fft_feat = self.fft_branch(fft_input)
        cov_feat = self.cov_branch(cov_input)
        z = self.encoder(fft_feat, cov_feat)
        recon = self.decoder(z)
        return recon

    def encode(self, inputs):
        fft_input = self.fft_layer(inputs)
        cov_input = self.cov_layer(inputs)
        fft_feat = self.fft_branch(fft_input)
        cov_feat = self.cov_branch(cov_input)
        return self.encoder(fft_feat, cov_feat)

class RegressionModel(Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.regressor = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        z = self.encoder.encode(inputs)
        return self.regressor(z)

# ---------- Custom Loss Function ----------

class CSICompositeLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 alpha=1.0,      # raw MSE
                 beta=1.0,       # cosine
                 gamma=0.0,      # FFT MSE
                 delta=0.0,      # Cov MSE
                 lam=0.0,        # Contrastive (optional)
                 use_contrastive=False,
                 name='feat_composite_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lam = lam
        self.use_contrastive = use_contrastive

        self.mse = tf.keras.losses.MeanSquaredError()
        self.cosine = tf.keras.losses.CosineSimilarity(axis=-1)

    def compute_fft_mse(self, y_true, y_pred):
        x_true = tf.complex(y_true[:, 0, :], y_true[:, 1, :])
        x_pred = tf.complex(y_pred[:, 0, :], y_pred[:, 1, :])

        fft_true = tf.signal.fft(x_true)
        fft_pred = tf.signal.fft(x_pred)

        mag_true = tf.abs(fft_true)
        mag_pred = tf.abs(fft_pred)

        return self.mse(mag_true, mag_pred)

    def compute_cov_mse(self, y_true, y_pred, L=192):
        def covariance_matrix(x):
            x_c = tf.complex(x[:, 0, :], x[:, 1, :])
            hankel = tf.stack([tf.roll(x_c, -i, axis=-1) for i in range(L)], axis=1)
            cov = tf.matmul(hankel, hankel, adjoint_b=True) / tf.cast(L, tf.complex64)
            return tf.math.real(cov)  # (batch, L, L)

        cov_true = covariance_matrix(y_true)
        cov_pred = covariance_matrix(y_pred)

        return self.mse(cov_true, cov_pred)

    def compute_contrastive_loss(self, z, temperature=0.1):
        # Placeholder NT-Xent (needs positive/negative pairs or augmentations)
        # You can plug SimCLR-style setup here if needed
        return 0.0

    def call(self, y_true, y_pred):
        # y_true, y_pred: shape (batch, 2, 256)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss_mse = self.mse(y_true, y_pred)
        loss_cos = (1+self.cosine(tf.reshape(y_true, [tf.shape(y_true)[0], -1]),
                               tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])))/2
        loss_fft = self.compute_fft_mse(y_true, y_pred)
        loss_cov = self.compute_cov_mse(y_true, y_pred)

        total_loss = self.alpha * loss_mse + \
                     self.beta * loss_cos + \
                     self.gamma * loss_fft + \
                     self.delta * loss_cov

        if self.use_contrastive:
            z = None  # <- you should pass latent z if you want this
            contrastive = self.compute_contrastive_loss(z)
            total_loss += self.lam * contrastive

        return total_loss

# ---------- Build Function ----------
def build_autoencode_model(input_shape):
    model = AutoencoderModel()

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=3e-4,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9
    )

    # loss_fn = CSICompositeLoss(alpha=0.5, beta=10, gamma=0, delta=0, lam=0.0)

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model

class UnfreezeEncoderCallback(tf.keras.callbacks.Callback):
    def __init__(self, encoder, unfreeze_epoch=5):
        super().__init__()
        self.encoder = encoder
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch:
            print(f"\n[INFO] Unfreezing encoder at epoch {epoch}")
            self.encoder.trainable = True

def build_Regression_model(input_shape, path):
    # Step 1: Instantiate AutoencoderModel
    autoencoder  = AutoencoderModel()

    # Step 2: Build the model by calling it once (needed to create weights)
    dummy_input = tf.random.normal((1, 2, 256))
    _ = autoencoder(dummy_input)

    # Step 3: Load pretrained weights
    autoencoder.load_weights(os.path.join(path, "autoencoder_fold0.weights.h5"))
    autoencoder.encoder.trainable = False

        # Step 4: Create RegressionModel using pretrained encoder
    regression_model = RegressionModel(encoder=autoencoder)

    # Step 5: Build the regression model (trigger weight creation)
    _ = regression_model(dummy_input)


    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=3e-4,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9
    )

    unfreeze_cb = UnfreezeEncoderCallback(encoder=autoencoder.encoder, unfreeze_epoch=80)

    # loss_fn = CSICompositeLoss(alpha=0.5, beta=10, gamma=0, delta=0, lam=0.0)

    regression_model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return regression_model, unfreeze_cb