"""
Data preparation for Task 2:
1. COCO 2014 Captions (3 per train image, 1 per dev image)
2. Annotation-derived grounding pairs (presence yes/no, counting 1-4)
3. Natural VQAv2 complementary pairs (consensus from >= 8 of 10 annotators)
"""
from typing import Dict, Any, List

def generate_annotation_grounding_pair(coco_annotation_data, object_category: str):
    """
    Generate presence or counting complementary pair from COCO annotations.
    """
    pass

def filter_vqav2_complementary_pairs(vqav2_data, min_agreement: int = 8):
    """
    Filter VQAv2 pairs where normalized consensus answers differ and have >=8 agreement.
    """
    pass
