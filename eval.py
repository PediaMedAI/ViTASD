import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datasets import AutismDatasetModule
from train import ViTASDLM

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from torchmetrics import Accuracy, ConfusionMatrix, AUROC
from torch.optim import Optimizer

from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from pathlib import Path
from typing import Optional


auroc = AUROC(num_classes=2)
accuracy = Accuracy()


model = ViTASDLM.load_from_checkpoint(
    checkpoint_path="",
    hparams_file="",
    map_location=None,
)


def get_predictions(model):

    softmax = nn.Softmax(dim=1)
    dataset_module = AutismDatasetModule()
    model.eval()

    predictions = []
    labels = []

    for data, label in iter(dataset_module.test_dataloader()):

        prediction = model(data)
        predictions.append(softmax(prediction))
        labels.append(label)

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    true_predictions = [max(a,b) for a,b in predictions.tolist()]
    return predictions, labels, true_predictions

in_preds, in_labels, in_true_preds = get_predictions(model)

print("in auroc:", auroc(in_preds, in_labels))
print("in accuracy:", accuracy(in_preds, in_labels))



