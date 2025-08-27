import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, GlobalMaxPooling1D, Dropout, Concatenate

# ========== Custom Layers ==========

class ExpandDim1Layer(Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=-1)  # (batch, 2, 256, 1)

class CSI_PointNet(tf.keras.Model):
    def __init__(self, dropout_rate=0.3):
        super(CSI_PointNet, self).__init__()

        # Shared MLP layers (applied to each "point")
        self.mlp1 = Dense(64)
        self.bn1 = BatchNormalization()
        self.mlp2 = Dense(128)
        self.bn2 = BatchNormalization()
        self.mlp3 = Dense(256)
        self.bn3 = BatchNormalization()

        # Aggregation
        self.pool = GlobalMaxPooling1D()

        # Dense head
        self.head1 = Dense(128, activation='relu')
        self.drop1 = Dropout(dropout_rate)
        self.head2 = Dense(64, activation='relu')
        self.drop2 = Dropout(dropout_rate)
        self.out = Dense(1)

    def call(self, inputs):
        # inputs: (batch, 2, 256)
        real = inputs[:, 0, :]  # (batch, 256)
        imag = inputs[:, 1, :]  # (batch, 256)
        pos = tf.linspace(0.0, 1.0, num=256)  # shape (256,)
        pos = tf.reshape(pos, (1, 256))
        pos = tf.tile(pos, [tf.shape(inputs)[0], 1])  # (batch, 256)

        # Combine features: (batch, 256, 3)
        x = tf.stack([real, imag, pos], axis=-1)

        # Shared MLP layers
        x = self.bn1(ReLU()(self.mlp1(x)))
        x = self.bn2(ReLU()(self.mlp2(x)))
        x = self.bn3(ReLU()(self.mlp3(x)))

        # Symmetric aggregation
        x = self.pool(x)

        # Dense head
        x = self.drop1(self.head1(x))
        x = self.drop2(self.head2(x))
        out = self.out(x)

        return out

def build_pointnet_model(input_shape):
    model = CSI_PointNet()

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=3e-4,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model

# class CovarianceLikeLayer(Layer):
#     def __init__(self, L=64, **kwargs):
#         super().__init__(**kwargs)
#         self.L = L

#     def call(self, inputs):
#         real = inputs[:, 0, :]  # (batch, 256)
#         imag = inputs[:, 1, :]
#         x = tf.complex(real, imag)  # (batch, 256)

#         Y = tf.signal.frame(x, frame_length=self.L, frame_step=1, axis=1)  # (batch, frames, L)
#         Y = tf.transpose(Y, perm=[0, 2, 1])  # (batch, L, frames)
#         Y_H = tf.transpose(tf.math.conj(Y), perm=[0, 2, 1])
#         Ry = tf.matmul(Y, Y_H) / tf.cast(tf.shape(Y)[-1], tf.complex64)
#         return tf.concat([tf.math.real(Ry), tf.math.imag(Ry)], axis=-1)  # (batch, L, 2L)

# class PositionalEncodingLayer(tf.keras.layers.Layer):
#     def __init__(self, max_len, d_model):
#         super().__init__()
#         self.max_len = max_len
#         self.d_model = d_model
#         self.pos_encoding = self._positional_encoding(max_len, d_model)

#     def _positional_encoding(self, length, d_model):
#         position = tf.cast(tf.range(length)[:, tf.newaxis], tf.float32)
#         div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
#         pe_sin = tf.sin(position * div_term)
#         pe_cos = tf.cos(position * div_term)
#         pe = tf.concat([pe_sin, pe_cos], axis=-1)[:, :d_model]
#         return pe[tf.newaxis, ...]

#     def call(self, x):
#         return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    
# class CovarianceEncoder(Layer):
#     def __init__(self, embed_dim=64, use_positional_encoding=True, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.use_pos_enc = use_positional_encoding

#         self.proj = Dense(embed_dim)
#         self.attn = MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4)
#         self.ffn = tf.keras.Sequential([
#             Dense(embed_dim * 2, activation='relu'),
#             Dense(embed_dim)
#         ])
#         self.norm1 = LayerNormalization()
#         self.norm2 = LayerNormalization()
#         self.output_proj = Dense(embed_dim)  # <--- NEW: unify output shape

#         if use_positional_encoding:
#             self.pos_enc = PositionalEncodingLayer(256, embed_dim)

#     def call(self, x):
#         x = self.proj(x)
#         if self.use_pos_enc:
#             x = self.pos_enc(x)

#         attn_out = self.attn(x, x)
#         x = self.norm1(x + attn_out)

#         ffn_out = self.ffn(x)
#         x = self.norm2(x + ffn_out)

#         # Top-k mean pooling across tokens (sequence axis)
#         scores = tf.reduce_mean(x, axis=-1)  # (batch, seq_len)
#         values, indices = tf.math.top_k(scores, k=8)  # (batch, 8)
#         topk_tokens = tf.gather(x, indices, batch_dims=1)  # (batch, 8, embed_dim)
#         pooled = tf.reduce_mean(topk_tokens, axis=1)  # (batch, embed_dim)

#         return self.output_proj(pooled)  # (batch, embed_dim)

# class FFTFeatureExtractor(tf.keras.layers.Layer):
#     def __init__(self, k=128, use_log=True, **kwargs):
#         super().__init__(**kwargs)
#         self.k = k
#         self.use_log = use_log
#         self.conv1 = Conv1D(32, 3, padding='same', activation='relu')
#         self.bn1 = BatchNormalization()
#         self.conv2 = Conv1D(64, 3, padding='same', activation='relu')
#         self.gap = GlobalAveragePooling1D()

#     def call(self, x):
#         real = x[:, 0, :]  # (batch, 256)
#         imag = x[:, 1, :]
#         fft = tf.signal.fft(tf.complex(real, imag))[:, :self.k]

#         mag = tf.abs(fft)
#         if self.use_log:
#             mag = tf.math.log1p(mag)
#         phase = tf.math.angle(fft)

#         feat = tf.stack([mag, phase], axis=-1)  # (batch, k, 2)
#         x = self.conv1(feat)
#         x = self.bn1(x)
#         x = self.conv2(x)
#         return self.gap(x)

# # ========== Loss & Metric ==========

# def nll_gaussian_with_kl(y_true, y_pred, kl_weight=0.2):
#     y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

#     mu = y_pred[:, 0]
#     log_sigma = y_pred[:, 1]
#     sigma = tf.math.softplus(log_sigma) + 1e-6  # ensure positivity

#     # Main negative log-likelihood (Gaussian)
#     dist = tfp.distributions.Normal(loc=mu, scale=sigma)
#     nll = -dist.log_prob(y_true)  # shape: (batch,)
#     nll_loss = tf.reduce_mean(nll)

#     # KL divergence to N(0, 1)
#     kl_div = -tf.math.log(sigma) + 0.5 * (tf.square(mu) + tf.square(sigma)) - 0.5
#     kl_div = tf.reduce_mean(kl_div)

#     # Final loss
#     return nll_loss + kl_weight * kl_div

# def mae_mu(y_true, y_pred):
#     y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # ensure float32
#     mu = y_pred[:, 0]
#     return tf.reduce_mean(tf.abs(y_true - mu))

# def mean_log_sigma(y_true, y_pred):
#     return tf.reduce_mean(y_pred[:, 1])

# # ========== Model Builder ==========

# def build_csi_model_refined(input_shape=(2, 256), L=128):
#     inp = Input(shape=input_shape)

#     # Branch 1: Raw CNN (ResNet style)
#     x = ExpandDim1Layer()(inp)
#     x = Conv2D(64, (2, 5), padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     res1 = x

#     x = Conv2D(64, (1, 5), padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Add()([x, res1])
#     x = Conv2D(64, (1, 3), padding='same', kernel_initializer='he_normal')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = GlobalAveragePooling2D()(x)

#     # Branch 2: Covariance + Transformer
#     cov = CovarianceLikeLayer(L=L)(inp)
#     cov_feat = CovarianceEncoder(embed_dim=64)(cov)

#     # Branch 3: FFT
#     fft_feat = FFTFeatureExtractor(k=128)(inp)

#     # Fusion
#     combined = Concatenate(axis=-1)([x, cov_feat, fft_feat])
#     hidden = Dense(128, activation='relu', kernel_initializer='he_normal')(combined)
#     hidden = Dropout(0.1)(hidden)
#     hidden = Dense(64, activation='relu', kernel_initializer='he_normal')(hidden)
#     hidden = Dropout(0.1)(hidden)

#     mu = Dense(1)(hidden)
#     log_sigma = Dense(1)(hidden)
#     output = Concatenate()([mu, log_sigma])

#     model = Model(inputs=inp, outputs=output)

#     lr_schedule = CosineDecayRestarts(
#         initial_learning_rate=3e-4,
#         first_decay_steps=40,
#         t_mul=2.0,
#         m_mul=0.9
#     )

#     model.compile(
#         optimizer=Adam(learning_rate=lr_schedule),
#         loss=nll_gaussian_with_kl,
#         metrics=[mae_mu, mean_log_sigma]
#     )

#     return model

# def plot_learning_curves(history, model_name="model", save_dir="plots"):
#     """
#     Plots and saves the learning curves of val_loss and val_mae_mu,
#     and displays std of the last 5 validation losses.

#     Args:
#         history: Keras History object from model.fit()
#         model_name: Name used for saving the plot
#         save_dir: Directory to save plots
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Extract values safely
#     val_loss = history.history.get('val_loss')
#     loss = history.history.get('loss')
#     val_mae = history.history.get('val_mae_mu', history.history.get('val_mae'))
#     mae = history.history.get('mae_mu', history.history.get('mae'))
#     epochs = np.arange(1, len(val_loss) + 1)

#     # Stability metric: std of last 5 val_losses
#     std_val_loss_5 = np.std(val_loss[-5:]) if val_loss else float('nan')
#     mean_val_loss_5 = np.mean(val_loss[-5:]) if val_loss else float('nan')

#     plt.figure(figsize=(10, 6))

#     # Plot val_loss and loss
#     plt.subplot(2, 1, 1)
#     if loss and val_loss:
#         plt.semilogy(epochs, loss, label='Train Loss', linestyle='--',)
#         plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
#     plt.title(f"{model_name} - Loss\nStability (std last 5 val_loss): {std_val_loss_5:.4f}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.grid(True)
#     plt.legend()

#     # Plot val_mae and mae
#     plt.subplot(2, 1, 2)
#     if mae and val_mae:
#         plt.plot(epochs, mae, label='Train MAE', linestyle='--')
#         plt.plot(epochs, val_mae, label='Val MAE', linewidth=2)
#     plt.xlabel("Epoch")
#     plt.ylabel("MAE")
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
#     plt.savefig(save_path)
#     plt.close()
    
#     print(f"[✓] Saved learning curves for {model_name} to: {save_path}")
#     print(f"    ↪ std(last 5 val_loss): {std_val_loss_5:.4f}, mean: {mean_val_loss_5:.4f}")