import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde

def plot_learning_curves(history, model_name="model", save_dir="plots"):
    """
    Plots and saves the learning curves of val_loss and val_mae, 
    and displays std of the last 5 validation losses.
    
    Args:
        history: Keras History object from model.fit()
        model_name: Name used for saving the plot
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract values
    val_loss = history.history.get('val_loss')
    val_mae = history.history.get('val_mae')
    loss = history.history.get('loss')
    mae = history.history.get('mae')
    epochs = np.arange(1, len(val_loss) + 1)

    # Stability metric: std of last 5 val_losses
    std_val_loss_5 = np.std(val_loss[-5:])
    mean_val_loss_5 = np.mean(val_loss[-5:])

    plt.figure(figsize=(10, 6))
    
    # Plot val_loss and loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, label='Train Loss', linestyle='--')
    plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.title(f"{model_name} - Loss\nStability (std last 5 val_loss): {std_val_loss_5:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot val_mae and mae
    plt.subplot(2, 1, 2)
    plt.plot(epochs, mae, label='Train MAE', linestyle='--')
    plt.plot(epochs, val_mae, label='Val MAE', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Saved learning curves for {model_name} to: {save_path}")
    print(f"    ↪ std(last 5 val_loss): {std_val_loss_5:.4f}, mean: {mean_val_loss_5:.4f}")

def plot_ecdf(maes, name, ax):
    """
    Common plot ECDF
    """
    ecdfs = [ECDF(m) for m in maes]
    styles = [("CNN", "-", 2), ("SLOPE", "--", 1.5), ("MUSIC", ":", 1.5)]

    for ecdf, (label, style, lw) in zip(ecdfs, styles):
        ax.plot(ecdf.x, ecdf.y, label=label, linestyle=style, linewidth=lw)
    ax.set_title(name)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, max([ecdf.x.max() for ecdf in ecdfs])])
    ax.set_xlim([0, 10])
    ax.legend(loc='lower right')

    cnn_val = ecdfs[0].x[np.searchsorted(ecdfs[0].y, 0.8, side='right') - 1]
    ax.text(0.95, 0.25, f'CNN @ 0.8 ≈ {cnn_val:.2f}m',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

def plot_hist(predictions, name, ax, target):
    """
    Plot KDE of multiple predictions, show target and annotate peaks.
    
    Parameters:
    - predictions: list of 1D arrays
    - name: plot title
    - ax: matplotlib Axes object
    - target: ground truth value (float)
    """
    styles = [("CNN", "blue"), ("SLOPE", "orange"), ("MUSIC", "green")]

    for prediction, (label, color) in zip(predictions, styles):
        prediction = np.asarray(prediction)
        sns.kdeplot(data=prediction, ax=ax, color=color, label=label)

        # Compute KDE peak
        if len(prediction) >= 3:
            kde = gaussian_kde(prediction)
            x_vals = np.linspace(prediction.min(), prediction.max(), 1000)
            y_vals = kde(x_vals)
            peak_x = x_vals[np.argmax(y_vals)]
            peak_y = np.max(y_vals)

            # Mark the peak
            ax.plot(peak_x, peak_y, 'o', color=color)
            ax.text(peak_x, peak_y, f'{peak_x:.2f}', color=color,
                    ha='left', va='bottom', fontsize=9, fontweight='bold')
            
    # Add vertical line for mean prediction
    ax.axvline(np.mean(predictions[0]), color='blue', linestyle='--', label=f'{np.mean(predictions[0]):.2f}', alpha = 0.5)

    # Add vertical line for ground truth target
    ax.axvline(np.mean(target), color='black', linestyle='--', label=f'Target')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(name)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='lower right')
