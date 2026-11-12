"""Shared paths, result loading, image display, and plotting style for the Task 2 notebooks."""
import json
import os
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from .data import image_filename

_TASK2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.environ.get("TASK2_OUTPUT_DIR", _TASK2_DIR)
RESULTS_DIR = os.environ.get("TASK2_RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))
MANIFEST_PATH = os.environ.get("TASK2_MANIFEST", os.path.join(_TASK2_DIR, "splits", "manifest_seed6304.json"))
FEATURE_DIR = os.environ.get("TASK2_FEATURE_DIR", os.path.join(OUTPUT_DIR, "features"))
IMAGE_DIR = os.environ.get("TASK2_IMAGE_DIR", os.path.join(_TASK2_DIR, "data", "images"))
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CONDITION_LABELS = {"correct": "Correct image", "swap": "Complementary swap", "mismatched": "Mismatched", "no_image": "No image"}
CONDITION_COLORS = {"correct": "#2ecc71", "swap": "#e74c3c", "mismatched": "#f39c12", "no_image": "#95a5a6"}
STAGE_COLORS = {1: "#d6eaf8", 2: "#fdebd0", 3: "#d5f5e3"}

def setup_notebook() -> None:
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 13, "axes.labelsize": 11})
    os.makedirs(FIG_DIR, exist_ok=True)

def save_figure(name: str) -> None:
    """Save the current figure with the task's other figures, then show it inline."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    plt.show()

def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_run(run: str = "main") -> Dict[str, Any]:
    """Saved training logs of a run. A missing run stops the notebook: figures never use placeholder values."""
    run_dir = os.path.join(RESULTS_DIR, run)
    result = {}
    for name, key in [("training_history.json", "history"), ("development_history.json", "dev"), ("summary.json", "summary")]:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist. Run the training step of kaggle_task2.ipynb first.")
        with open(path, "r", encoding="utf-8") as f:
            result[key] = json.load(f)
    result["history"] = pd.DataFrame(result["history"])
    result["dev"] = pd.DataFrame(result["dev"])
    return result

def load_final(name: str, run: str = "main") -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, run, f"final_{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run the final evaluation step of kaggle_task2.ipynb first.")
    return pd.read_csv(path)

def open_image(key: str) -> Optional[Image.Image]:
    """The image for a key, or None where images were not kept (later sessions keep features only)."""
    path = os.path.join(IMAGE_DIR, image_filename(key))
    return Image.open(path).convert("RGB") if os.path.exists(path) else None

def show_image(ax, key: Optional[str], title: str = "") -> None:
    image = open_image(key) if key else None
    if image is not None:
        ax.imshow(image)
    else:
        ax.text(0.5, 0.5, "No image" if key is None else "Image not on this machine", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def shade_stages(ax, summary_boundaries=(2250, 3750, 4500)) -> None:
    """Background bands for the three curriculum stages."""
    start = 0
    for stage, end in enumerate(summary_boundaries, start=1):
        ax.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.5, zorder=0, label=f"Stage {stage}")
        start = end
