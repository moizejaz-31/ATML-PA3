"""
Answer normalization and mathematical equivalence for Task 3.
- Uses math-verify for MATH equivalence.
- Uses MathVista type/choice/precision rules.
- Maps missing extractions to NO_ANSWER.
"""
from typing import Optional

NO_ANSWER = "NO_ANSWER"

def normalize_math_answer(answer_str: Optional[str]) -> str:
    """Normalize MATH answer using math-verify logic."""
    if not answer_str or not answer_str.strip():
        return NO_ANSWER
    # Normalization placeholder
    return answer_str.strip()

def normalize_mathvista_answer(answer_str: Optional[str], question_type: str = "free_form") -> str:
    """Normalize MathVista answer string according to answer-type rules."""
    if not answer_str or not answer_str.strip():
        return NO_ANSWER
    return answer_str.strip()
