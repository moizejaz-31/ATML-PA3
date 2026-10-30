import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Configure matplotlib with clean publication-style defaults."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def plot_learning_curves(history_dict, x_key="step", y_key="loss", title="Learning Curves", save_path=None):
    """Plot multi-run learning curves."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, hist in history_dict.items():
        ax.plot(hist[x_key], hist[y_key], label=name)
    ax.set_xlabel(x_key.capitalize())
    ax.set_ylabel(y_key.capitalize())
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    return fig, ax
