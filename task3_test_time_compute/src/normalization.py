"""
Answer normalization and equivalence for Task 3.
- MATH: math-verify equivalence (with exact-string agreement first).
- MathVista: answer-type, choice, and precision rules from the released question metadata.
- Missing extractions map to one NO_ANSWER group, always graded incorrect.
Answer groups are formed without consulting gold answers; grading uses the same rules.
"""
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

NO_ANSWER = "NO_ANSWER"
# Tests may run without math-verify. Real analysis must never silently lose equivalence.
ALLOW_STRING_EQUIVALENCE_ENV = "TASK3_ALLOW_STRING_EQUIVALENCE"

def normalize_math_answer(answer_str: Optional[str]) -> str:
    """Strip a stored MATH prediction. Equivalence itself is decided by math_equivalent."""
    if answer_str is None or not str(answer_str).strip():
        return NO_ANSWER
    return str(answer_str).strip()

def scorer_version() -> str:
    try:
        from importlib.metadata import version
        return f"math-verify {version('math-verify')}"
    except Exception:
        return "string equality (test mode)"

class MathEquivalence:
    """math-verify comparison with parse and pair caches."""
    def __init__(self):
        self.parsed: Dict[str, Any] = {}
        self.pairs: Dict[Tuple[str, str], bool] = {}
        try:
            from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
            self._parse, self._verify = parse, verify
            self._config = [LatexExtractionConfig(), ExprExtractionConfig()]
        except ImportError:
            if not os.environ.get(ALLOW_STRING_EQUIVALENCE_ENV):
                raise
            self._parse = None

    def _parsed(self, answer: str):
        if answer not in self.parsed:
            self.parsed[answer] = self._parse(f"${answer}$", extraction_config=self._config)
        return self.parsed[answer]

    def __call__(self, gold_like: str, candidate: str) -> bool:
        if gold_like == candidate:
            return True
        if NO_ANSWER in (gold_like, candidate) or self._parse is None:
            return False
        key = (gold_like, candidate)
        if key not in self.pairs:
            try:
                first, second = self._parsed(gold_like), self._parsed(candidate)
                self.pairs[key] = bool(first and second and self._verify(first, second))
            except Exception:
                self.pairs[key] = False
        return self.pairs[key]

def group_answers(answers: List[str], equivalent: Callable[[str, str], bool]) -> Dict[str, str]:
    """Map each answer string to the first-seen representative of its equivalence group."""
    representatives: List[str] = []
    mapping: Dict[str, str] = {}
    for answer in answers:
        if answer in mapping:
            continue
        match = next((rep for rep in representatives if equivalent(rep, answer)), None)
        if match is None:
            representatives.append(answer)
            match = answer
        mapping[answer] = match
    return mapping

def _option_letter(index: int) -> str:
    return chr(ord("A") + index)

def normalize_mathvista_answer(answer_str: Optional[str], metadata: Dict[str, Any]) -> str:
    """
    MathVista rules: multiple-choice answers become option letters; integers and floats use
    the question's type and precision; other answers compare as lowercase stripped text.
    """
    if answer_str is None or not str(answer_str).strip():
        return NO_ANSWER
    text = str(answer_str).strip()

    if metadata.get("question_type") == "multi_choice":
        choices = [str(c).strip() for c in (metadata.get("choices") or [])]
        letter = re.fullmatch(r"\(?([A-Za-z])\)?\.?", text)
        if letter:
            index = ord(letter.group(1).upper()) - ord("A")
            if 0 <= index < len(choices):
                return _option_letter(index)
        lowered = [c.lower() for c in choices]
        if text.lower() in lowered:
            return _option_letter(lowered.index(text.lower()))
        return text.lower()

    answer_type = metadata.get("answer_type")
    number = re.sub(r"[,$%\s]", "", text)
    try:
        if answer_type == "integer":
            return str(int(round(float(number))))
        if answer_type == "float":
            return f"{round(float(number), int(metadata.get('precision') or 0)):.{int(metadata.get('precision') or 0)}f}"
    except ValueError:
        return text.lower()
    if answer_type == "list":
        return re.sub(r"\s+", "", text).lower()
    return text.lower()
