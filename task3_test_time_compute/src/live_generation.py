"""
Live generation harness for Qwen2.5-3B-Instruct on 50 MATH evaluation problems.
Compares:
1. Single longer response (3,072 new tokens)
2. Parallel sampling (3 x 1,024 new tokens + self-consistency)
3. Sequential refinement (1,024 tokens + 2,048 tokens critique and revision)
"""
import re
from typing import Optional

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract last \boxed{...} answer using balanced braces.
    """
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None
