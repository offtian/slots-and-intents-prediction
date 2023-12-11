import os
import json
import re
import pandas as pd

from typing import List
from datasets import Dataset
from datasets import DatasetDict


def postprocess_text(preds: List[str]) -> List[str]:
    preds = [
        re.sub(" +", " ", pred.replace("]", " ] ").replace("[", " [")).strip()
        for pred in preds
    ]
    return preds


def load_dataset(filename: str) -> Dataset:
    return Dataset.from_pandas(pd.read_csv(filename, sep='\t'))


def load_datasets(datapath: str) -> DatasetDict:
    """Loads all the splits"""

    splits = ["train", "eval", "test"]

    dataset = DatasetDict(
        {
            split: load_dataset(os.path.join(datapath, f"{split}.tsv"))
            for split in splits
        }
    )

    return dataset


def load_semantic_vocab(datapath: str) -> dict:
    """Loads the semantic vocabulary of intents and slots"""

    with open(os.path.join(datapath, "slots_and_intents.json")) as f:
        data = json.load(f)

    return data
