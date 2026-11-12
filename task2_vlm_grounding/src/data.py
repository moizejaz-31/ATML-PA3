"""
Data selection for Task 2, from COCO 2014 captions and instances and VQAv2 complementary pairs:
  1. COCO captions (6,000 train images x 3 = 18,000 examples; 96 dev images x 1)
  2. Annotation-derived grounding pairs (presence and counting; pools for training, 24 dev, 64 test pairs)
  3. Natural VQAv2 complementary pairs (agreement >= 8/10 per image; 1,536 train, 32 dev, 96 test pairs)
Train2014 images serve training; val2014 images serve image-disjoint development and final test.
A missing quota is an error: selection never silently shrinks.
"""
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

SEED = 6304
SIZES = {
    "caption_train_images": 6000, "captions_per_image": 3, "caption_dev_images": 96,
    "grounding_dev": {"presence": 12, "counting": 12}, "grounding_test": {"presence": 32, "counting": 32},
    "vqa_train": {"yes/no": 640, "number": 256, "other": 640}, "vqa_dev": {"yes/no": 12, "number": 8, "other": 12},
    "vqa_test": {"yes/no": 36, "number": 24, "other": 36},
    "pool_pairs_per_category": 100,
}
RAW_FILES = {
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip": ["annotations/captions_train2014.json", "annotations/captions_val2014.json", "annotations/instances_train2014.json", "annotations/instances_val2014.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip": ["v2_OpenEnded_mscoco_train2014_questions.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip": ["v2_OpenEnded_mscoco_val2014_questions.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip": ["v2_mscoco_train2014_annotations.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip": ["v2_mscoco_val2014_annotations.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip": ["v2_mscoco_train2014_complementary_pairs.json"],
    "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip": ["v2_mscoco_val2014_complementary_pairs.json"],
}
IRREGULAR_PLURALS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors", "knife": "knives", "bus": "buses", "couch": "couches", "toothbrush": "toothbrushes", "wine glass": "wine glasses", "dining table": "dining tables"}
DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

def normalize_vqa_answer(text: Optional[str]) -> str:
    """VQA-style normalization: lowercase, drop punctuation and articles, standardize number words."""
    if not text:
        return ""
    text = text.lower().strip().replace("\n", " ")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s.]", "", text)
    words = [DIGITS.get(w, w) for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)

def plural(category: str) -> str:
    return IRREGULAR_PLURALS.get(category, f"{category}s")

def image_key(split: str, image_id: int) -> str:
    return f"{split}/{int(image_id)}"

def download_raw(raw_dir: str) -> None:
    """Fetch annotation and VQA archives. Images are fetched individually later, once selected."""
    os.makedirs(raw_dir, exist_ok=True)
    for url, members in RAW_FILES.items():
        if all(os.path.exists(os.path.join(raw_dir, m)) for m in members):
            continue
        archive = os.path.join(raw_dir, os.path.basename(url))
        print(f"[Data] Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(raw_dir)
        os.remove(archive)

def _load(raw_dir: str, name: str) -> Any:
    with open(os.path.join(raw_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

def object_counts(instances: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Set[str]], List[str]]:
    """Per image: non-crowd counts per category, and categories with a crowd annotation (ambiguous counts)."""
    names = {c["id"]: c["name"] for c in instances["categories"]}
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    crowded: Dict[int, Set[str]] = defaultdict(set)
    for ann in instances["annotations"]:
        if ann.get("iscrowd"):
            crowded[ann["image_id"]].add(names[ann["category_id"]])
        else:
            counts[ann["image_id"]][names[ann["category_id"]]] += 1
    return {k: dict(v) for k, v in counts.items()}, dict(crowded), sorted(names.values())

def presence_pair(category: str, split: str, with_object: int, without_object: int) -> Dict[str, Any]:
    return {"type": "presence", "category": category, "question": f"Does this image contain {category}?",
            "a": {"image": image_key(split, with_object), "target": "yes", "answers": ["yes"] * 10},
            "b": {"image": image_key(split, without_object), "target": "no", "answers": ["no"] * 10}}

def counting_pair(category: str, split: str, image_a: int, count_a: int, image_b: int, count_b: int) -> Dict[str, Any]:
    return {"type": "counting", "category": category, "question": f"How many {plural(category)} can you see?",
            "a": {"image": image_key(split, image_a), "target": str(count_a), "answers": [str(count_a)] * 10},
            "b": {"image": image_key(split, image_b), "target": str(count_b), "answers": [str(count_b)] * 10}}

def build_grounding_candidates(split: str, image_ids: List[int], counts: Dict[int, Dict[str, int]], crowded: Dict[int, Set[str]],
                               categories: List[str], per_category: int, rng: np.random.Generator) -> Dict[str, List[Dict[str, Any]]]:
    """Generate presence and counting pairs per category, restricted to the given images."""
    pools: Dict[str, List[Dict[str, Any]]] = {"presence": [], "counting": []}
    ids = np.array(sorted(image_ids))
    for category in categories:
        has = [int(i) for i in ids if counts.get(int(i), {}).get(category, 0) > 0 and category not in crowded.get(int(i), set())]
        lacks = [int(i) for i in ids if category not in counts.get(int(i), {}) and category not in crowded.get(int(i), set())]
        if has and lacks:
            for _ in range(per_category):
                pools["presence"].append(presence_pair(category, split, int(rng.choice(has)), int(rng.choice(lacks))))
        by_count = defaultdict(list)
        for i in has:
            if 1 <= counts[i][category] <= 4:
                by_count[counts[i][category]].append(i)
        levels = sorted(by_count)
        if len(levels) >= 2:
            for _ in range(per_category):
                first, second = rng.choice(levels, size=2, replace=False)
                pools["counting"].append(counting_pair(category, split, int(rng.choice(by_count[int(first)])), int(first), int(rng.choice(by_count[int(second)])), int(second)))
    return pools

def take_disjoint(candidates: List[Dict[str, Any]], quota: int, used: Set[str], rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Take shuffled pairs whose two images are not used elsewhere, marking them used."""
    chosen = []
    for index in rng.permutation(len(candidates)):
        pair = candidates[int(index)]
        if len(chosen) == quota:
            break
        if pair["a"]["image"] not in used and pair["b"]["image"] not in used and pair["a"]["image"] != pair["b"]["image"]:
            chosen.append(pair)
            used.update([pair["a"]["image"], pair["b"]["image"]])
    if len(chosen) < quota:
        raise ValueError(f"Only {len(chosen)} of {quota} disjoint pairs available")
    return chosen

def consensus(answers: List[str]) -> Tuple[str, int, str]:
    """Normalized consensus answer, its count, and an original answer string from that group."""
    normalized = [normalize_vqa_answer(a) for a in answers]
    top, count = Counter(normalized).most_common(1)[0]
    return top, count, answers[normalized.index(top)]

def filter_vqa_pairs(raw_dir: str, split: str, min_agreement: int = 8) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Keep complementary pairs whose normalized consensus answers differ, each with >= 8/10 agreement."""
    questions = {q["question_id"]: q for q in _load(raw_dir, f"v2_OpenEnded_mscoco_{split}_questions.json")["questions"]}
    annotations = {a["question_id"]: a for a in _load(raw_dir, f"v2_mscoco_{split}_annotations.json")["annotations"]}
    raw_pairs = _load(raw_dir, f"v2_mscoco_{split}_complementary_pairs.json")
    stats = {"complementary_pairs": len(raw_pairs), "low_agreement": 0, "same_consensus": 0, "retained": 0}
    pairs = []
    for qid_a, qid_b in raw_pairs:
        qa, qb = questions[qid_a], questions[qid_b]
        if qa["image_id"] == qb["image_id"] or qa["question"] != qb["question"]:
            continue
        answers_a = [x["answer"] for x in annotations[qid_a]["answers"]]
        answers_b = [x["answer"] for x in annotations[qid_b]["answers"]]
        top_a, count_a, target_a = consensus(answers_a)
        top_b, count_b, target_b = consensus(answers_b)
        if count_a < min_agreement or count_b < min_agreement:
            stats["low_agreement"] += 1
            continue
        if top_a == top_b:
            stats["same_consensus"] += 1
            continue
        stats["retained"] += 1
        pairs.append({"type": "natural", "answer_type": annotations[qid_a]["answer_type"], "question": qa["question"], "question_ids": [qid_a, qid_b],
                      "a": {"image": image_key(split, qa["image_id"]), "target": target_a, "answers": answers_a},
                      "b": {"image": image_key(split, qb["image_id"]), "target": target_b, "answers": answers_b}})
    return pairs, stats

def select_vqa(pairs: List[Dict[str, Any]], quotas: Dict[str, int], rng: np.random.Generator, used: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Per-answer-type selection. With `used`, also keeps images disjoint from earlier sets."""
    selected = []
    for answer_type, quota in quotas.items():
        candidates = [p for p in pairs if p["answer_type"] == answer_type]
        if used is not None:
            selected += take_disjoint(candidates, quota, used, rng)
        else:
            if len(candidates) < quota:
                raise ValueError(f"Only {len(candidates)} of {quota} {answer_type} pairs available")
            selected += [candidates[int(i)] for i in rng.choice(len(candidates), size=quota, replace=False)]
    return selected

def prepare_manifest(raw_dir: str, seed: int = SEED, sizes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Select every example, pair, and image. The manifest is saved so that all later steps reuse it."""
    sizes = sizes or SIZES
    rng = np.random.default_rng(seed)

    # Training captions, then annotation pools within those images
    train_captions = defaultdict(list)
    for ann in sorted(_load(raw_dir, "annotations/captions_train2014.json")["annotations"], key=lambda a: a["id"]):
        train_captions[ann["image_id"]].append(ann["caption"].strip())
    eligible = sorted(i for i, c in train_captions.items() if len(c) >= sizes["captions_per_image"])
    train_images = [int(i) for i in rng.choice(eligible, size=sizes["caption_train_images"], replace=False)]
    caption_train = [{"image": image_key("train2014", i), "caption": c} for i in train_images for c in train_captions[i][:sizes["captions_per_image"]]]

    counts, crowded, categories = object_counts(_load(raw_dir, "annotations/instances_train2014.json"))
    pools = build_grounding_candidates("train2014", train_images, counts, crowded, categories, sizes["pool_pairs_per_category"], rng)

    # Development and final test come from val2014, all image-disjoint
    used: Set[str] = set()
    val_captions = defaultdict(list)
    for ann in sorted(_load(raw_dir, "annotations/captions_val2014.json")["annotations"], key=lambda a: a["id"]):
        val_captions[ann["image_id"]].append(ann["caption"].strip())
    val_images = sorted(val_captions)
    caption_dev_ids = [int(i) for i in rng.choice(val_images, size=sizes["caption_dev_images"], replace=False)]
    caption_dev = [{"image": image_key("val2014", i), "caption": val_captions[i][0]} for i in caption_dev_ids]
    used.update(x["image"] for x in caption_dev)

    counts, crowded, categories = object_counts(_load(raw_dir, "annotations/instances_val2014.json"))
    val_candidates = build_grounding_candidates("val2014", val_images, counts, crowded, categories, sizes["pool_pairs_per_category"], rng)
    grounding = {}
    for name in ["grounding_dev", "grounding_test"]:
        grounding[name] = [pair for kind, quota in sizes[name].items() for pair in take_disjoint(val_candidates[kind], quota, used, rng)]

    vqa_val, vqa_val_stats = filter_vqa_pairs(raw_dir, "val2014")
    vqa_dev = select_vqa(vqa_val, sizes["vqa_dev"], rng, used)
    vqa_test = select_vqa(vqa_val, sizes["vqa_test"], rng, used)
    vqa_train_all, vqa_train_stats = filter_vqa_pairs(raw_dir, "train2014")
    vqa_train = select_vqa(vqa_train_all, sizes["vqa_train"], rng)

    manifest = {"seed": seed, "sizes": sizes, "caption_train": caption_train, "caption_dev": caption_dev,
                "presence_pool": pools["presence"], "counting_pool": pools["counting"],
                "grounding_dev": grounding["grounding_dev"], "grounding_test": grounding["grounding_test"],
                "vqa_train": vqa_train, "vqa_dev": vqa_dev, "vqa_test": vqa_test,
                "filter_stats": {"train2014": vqa_train_stats, "val2014": vqa_val_stats}}
    check_split_isolation(manifest)
    return manifest

def pair_images(pairs: Iterable[Dict[str, Any]]) -> Set[str]:
    return {p[side]["image"] for p in pairs for side in ("a", "b")}

def check_split_isolation(manifest: Dict[str, Any]) -> None:
    """Evaluation images never enter training or replay, and evaluation sets are mutually disjoint."""
    training = {x["image"] for x in manifest["caption_train"]} | pair_images(manifest["presence_pool"]) | pair_images(manifest["counting_pool"]) | pair_images(manifest["vqa_train"])
    sets = {"caption_dev": {x["image"] for x in manifest["caption_dev"]}, "grounding_dev": pair_images(manifest["grounding_dev"]), "grounding_test": pair_images(manifest["grounding_test"]),
            "vqa_dev": pair_images(manifest["vqa_dev"]), "vqa_test": pair_images(manifest["vqa_test"])}
    names = list(sets)
    for i, name in enumerate(names):
        if sets[name] & training:
            raise ValueError(f"{name} shares images with training")
        for other in names[i + 1:]:
            if sets[name] & sets[other]:
                raise ValueError(f"{name} and {other} share images")

def all_images(manifest: Dict[str, Any]) -> List[str]:
    keys = {x["image"] for name in ["caption_train", "caption_dev"] for x in manifest[name]}
    for name in ["presence_pool", "counting_pool", "grounding_dev", "grounding_test", "vqa_train", "vqa_dev", "vqa_test"]:
        keys |= pair_images(manifest[name])
    return sorted(keys)

def image_filename(key: str) -> str:
    split, image_id = key.split("/")
    return f"COCO_{split}_{int(image_id):012d}.jpg"

def download_images(keys: List[str], image_dir: str, workers: int = 16) -> int:
    """Fetch only the selected COCO images. Any failure after retries is an error."""
    os.makedirs(image_dir, exist_ok=True)
    def fetch(key: str) -> int:
        split = key.split("/")[0]
        name = image_filename(key)
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return 0
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/{split}/{name}", path + ".tmp")
                os.replace(path + ".tmp", path)
                return 1
            except OSError:
                if attempt == 3:
                    raise
        return 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(fetch, keys))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Select Task 2 data and download the selected images")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="task2_vlm_grounding/splits/manifest_seed6304.json")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--sizes_json", type=str, default=None, help="Tests only: override the manual's sizes")
    parser.add_argument("--skip_downloads", action="store_true", help="Tests only: raw files and images already exist")
    args = parser.parse_args()

    if not args.skip_downloads:
        download_raw(args.raw_dir)
    if os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        sizes = json.load(open(args.sizes_json)) if args.sizes_json else None
        manifest = prepare_manifest(args.raw_dir, sizes=sizes)
        os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    keys = all_images(manifest)
    print(f"[Data] {len(keys)} selected images | captions {len(manifest['caption_train'])} | pools {len(manifest['presence_pool'])}/{len(manifest['counting_pool'])} | "
          f"grounding {len(manifest['grounding_dev'])}/{len(manifest['grounding_test'])} | VQA {len(manifest['vqa_train'])}/{len(manifest['vqa_dev'])}/{len(manifest['vqa_test'])}", flush=True)
    if not args.skip_downloads:
        print(f"[Data] Downloaded {download_images(keys, args.image_dir)} new images", flush=True)
