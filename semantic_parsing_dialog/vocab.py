import re
import os
import json

from datasets import DatasetDict
from datasets import Dataset

from typing import List

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


def _extract_all(pattern: str, x: str) -> List[str]:
    """Extracts all pattern instances from an input string x"""
    matches = re.finditer(pattern, x)
    return [m.group(1) for m in matches]


def get_intents(x: str) -> List[str]:
    """Extracts all intents from the serialized representation"""
    return _extract_all(r"IN:(.+?)\s", x)


def get_slots(x: str) -> List[str]:
    """Extracts all slots from the serialized representation"""
    return _extract_all(r"SL:(.+?)\s", x)
