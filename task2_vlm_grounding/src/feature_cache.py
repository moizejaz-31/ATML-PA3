"""
SigLIP visual feature extraction and FP16 caching for Task 2.
Model: google/siglip-base-patch16-224. 196 patch features (14x14, spatial order) of dimension 768.
The vision transformer's last_hidden_state already has the final layer norm applied, exactly once.
All features live in one FP16 array, with an index from image key to row.
"""
import json
import os
import time
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .data import image_filename

def pool_features_2x2(features_196: torch.Tensor) -> torch.Tensor:
    """
    Compress 196 patch tokens into 49 tokens by 2x2 spatial average pooling.
    Input: (B, 196, 768) or (196, 768)
    Output: (B, 49, 768) or (49, 768)
    """
    is_batched = features_196.dim() == 3
    if not is_batched:
        features_196 = features_196.unsqueeze(0)

    B, N, D = features_196.shape
    assert N == 196, f"Expected 196 tokens, got {N}"

    # Reshape to spatial grid: (B, 14, 14, D) -> (B, D, 14, 14)
    spatial = features_196.view(B, 14, 14, D).permute(0, 3, 1, 2)
    # Average pool 2x2 with stride 2 -> (B, D, 7, 7)
    pooled = nn.functional.avg_pool2d(spatial, kernel_size=2, stride=2)
    # Reshape back to sequence: (B, 49, D)
    pooled_tokens = pooled.permute(0, 2, 3, 1).contiguous().view(B, 49, D)

    return pooled_tokens if is_batched else pooled_tokens.squeeze(0)

class FeatureStore:
    """Read-only cached features. An image that was never cached is an error, never a zero vector."""
    def __init__(self, feature_dir: str, visual_tokens: int = 196):
        with open(os.path.join(feature_dir, "index.json"), "r", encoding="utf-8") as f:
            self.index: Dict[str, int] = json.load(f)["index"]
        self.array = np.load(os.path.join(feature_dir, "features.npy"), mmap_mode="r")
        self.visual_tokens = visual_tokens

    def get(self, key: str) -> torch.Tensor:
        features = torch.from_numpy(np.array(self.array[self.index[key]]))
        return pool_features_2x2(features.float()).half() if self.visual_tokens == 49 else features

def cache_features(keys: List[str], image_dir: str, output_dir: str, model_name: str = "google/siglip-base-patch16-224", batch_size: int = 64) -> Dict[str, float]:
    """Encode every selected image once, then release the encoder before training."""
    index_path = os.path.join(output_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if set(saved["index"]) == set(keys):
            print(f"[Feature Cache] Reusing {len(keys)} cached images")
            return saved["stats"]

    from transformers import AutoImageProcessor, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).vision_model.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False

    start = time.perf_counter()
    array = None
    for first in range(0, len(keys), batch_size):
        batch = keys[first:first + batch_size]
        images = [Image.open(os.path.join(image_dir, image_filename(k))).convert("RGB") for k in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            patches = encoder(**inputs).last_hidden_state.half().cpu().numpy()
        if array is None:
            array = np.lib.format.open_memmap(os.path.join(output_dir, "features.tmp.npy"), mode="w+", dtype=np.float16, shape=(len(keys),) + patches.shape[1:])
        array[first:first + len(batch)] = patches
        if first % (batch_size * 20) == 0:
            print(f"[Feature Cache] {first + len(batch)}/{len(keys)}", flush=True)
    array.flush()
    del array
    os.replace(os.path.join(output_dir, "features.tmp.npy"), os.path.join(output_dir, "features.npy"))

    stats = {"images": len(keys), "seconds": time.perf_counter() - start, "peak_gpu_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
             "size_mb": os.path.getsize(os.path.join(output_dir, "features.npy")) / (1024 ** 2)}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"index": {k: i for i, k in enumerate(keys)}, "stats": stats, "model": model_name}, f)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


if __name__ == "__main__":
    import argparse
    from .data import all_images
    parser = argparse.ArgumentParser(description="Cache SigLIP features for the selected images")
    parser.add_argument("--manifest", type=str, default="task2_vlm_grounding/splits/manifest_seed6304.json")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(cache_features(all_images(manifest), args.image_dir, args.output_dir, args.model_name, args.batch_size))
