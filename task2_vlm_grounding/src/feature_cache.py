"""
SigLIP visual feature extraction and FP16 disk caching for Task 2.
Model: google/siglip-base-patch16-224 (86M parameters).
Extracts 196 patch features (14x14 grid) of dimension 768.
Applies SigLIP's final layer normalization exactly once.
Optionally compresses features to 49 tokens via 2x2 average spatial pooling.
"""
import os
import torch
import torch.nn as nn
from typing import List, Optional, Union
from PIL import Image

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

def extract_and_cache_siglip_features(
    image_paths: List[str],
    output_dir: str,
    model_name: str = "google/siglip-base-patch16-224",
    compress_49: bool = False,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> int:
    """
    Extracts 196 patch features (or 49 if compressed) from images in FP16 and caches to disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoProcessor, AutoModel
    print(f"[Feature Cache] Loading vision encoder: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    vision_model = AutoModel.from_pretrained(model_name).vision_model.to(device).eval()

    os.makedirs(output_dir, exist_ok=True)
    cached_count = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        valid_paths = []
        for p in batch_paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Skipping corrupted image {p}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vision_model(**inputs)
            # Patch features (B, 196, 768)
            patch_feats = outputs.last_hidden_state

            if compress_49:
                patch_feats = pool_features_2x2(patch_feats)

            patch_feats_fp16 = patch_feats.half().cpu()

        for path, feat in zip(valid_paths, patch_feats_fp16):
            base_id = os.path.splitext(os.path.basename(path))[0]
            save_path = os.path.join(output_dir, f"{base_id}.pt")
            torch.save(feat, save_path)
            cached_count += 1

    print(f"[Feature Cache] Successfully cached {cached_count} image features in FP16 to {output_dir}")
    return cached_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and cache SigLIP visual features in FP16")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cached .pt features")
    parser.add_argument("--compress_49", action="store_true", help="Apply 2x2 spatial pooling to 49 tokens")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    import glob
    imgs = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + glob.glob(os.path.join(args.image_dir, "*.png")))
    print(f"Found {len(imgs)} images to process...")
    extract_and_cache_siglip_features(
        image_paths=imgs,
        output_dir=args.output_dir,
        compress_49=args.compress_49,
        batch_size=args.batch_size,
    )
