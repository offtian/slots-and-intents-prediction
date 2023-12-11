import os
import torch

from .evaluate import evaluate_predictions
from .utils import load_dataset, postprocess_text


ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
