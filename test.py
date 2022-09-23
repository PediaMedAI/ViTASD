import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from datasets import AutismDatasetModule
from models import ViTASD

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from pytorch_lightning.callbacks import Callback

from torchmetrics import Accuracy, ConfusionMatrix
from torch.optim import Optimizer

from timm.data import Mixup
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from pathlib import Path
from typing import Optional


class ViTASDLM(LightningModule):
    def __init__(self,
                 batch_size: int = 256,
                 num_classes: int = 2,
                 epochs: int = 200,
                 attn_only: bool = True,
                 smoothing: float = 0.0,  # Label smoothing
                 vis_path: str = "./runs/vis",

                 # Model parameters
                 model: str = "deit3_base_patch16_224",  # Name of model to train
                 input_size: int = 224,  # images input size
                 drop: float = 0.0,  # Dropout rate
                 drop_path: float = 0.05,  # Drop path rate

                 # Optimizer parameters
                 opt: str = "adamw",
                 weight_decay: float = 0.05,

                 # Learning rate schedule parameters
                 sched: str = "cosine",
                 lr: float = 4e-3,
                 warmup_lr: float = 1e-6,
                 min_lr: float = 1e-5,
                 warmup_epochs: int = 5,  # epochs to warmup LR, if scheduler supports
                 cooldown_epochs: int = 0,  # epochs to cooldown LR at min_lr, after cyclic schedule ends

                 # Mixup parameters
                 mixup: float = 0.8,  # mixup alpha, mixup enabled if > 0
                 cutmix: float = 1.0,  # cutmix alpha, cutmix enabled if > 0.
                 mixup_prob: float = 1.0,  # Prob of performing mixup or cutmix when either/both is enabled
                 mixup_switch_prob: float = 0.5,  # Prob of switching to cutmix when both mixup and cutmix enabled
                 mixup_mode: str = "batch",  # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
                 ):

        super(ViTASDLM, self).__init__()
        self.save_hyperparameters()

        self.model: torch.nn.Module = create_model(
            self.hparams.model,
            pretrained=True,
            num_classes=self.hparams.num_classes,
            drop_rate=self.hparams.drop,
            drop_path_rate=self.hparams.drop_path,
            drop_block_rate=None,
            img_size=self.hparams.input_size
        )

        state_dict = torch.load("/home/xucao/ASD/ViTASD/lightning_logs/ViTASD/version_1/checkpoints/epoch=14-step=1185.ckpt")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[0:5]
            if name == "model":
                new_state_dict[k[6:]] = v

        self.model.load_state_dict(new_state_dict)

        self._init_mixup()
        self._init_frozen_params()
        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.valid_criterion = torch.nn.CrossEntropyLoss()
        self.valid_acc = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes, normalize='true')

    def _init_mixup(self):
        self.mixup_fn = None
        mixup_active = self.hparams.mixup > 0 or self.hparams.cutmix > 0.
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.hparams.mixup,
                cutmix_alpha=self.hparams.cutmix,
                cutmix_minmax=None,
                prob=self.hparams.mixup_prob,
                switch_prob=self.hparams.mixup_switch_prob,
                mode=self.hparams.mixup_mode,
                label_smoothing=self.hparams.smoothing,
                num_classes=self.hparams.num_classes
            )

    def _init_frozen_params(self):
        if self.hparams.attn_only:
            for name_p, p in self.model.named_parameters():
                if '.attn.' in name_p:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            self.model.head.weight.requires_grad = True
            self.model.head.bias.requires_grad = True
            self.model.pos_embed.requires_grad = True
            for p in self.model.patch_embed.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        samples, targets = batch
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.forward(samples)
        loss = self.train_criterion(outputs, targets)
        loss_value = loss.item()
        self.log('Loss/train', loss_value, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        samples, targets = batch
        outputs = self.forward(samples)
        loss = self.valid_criterion(outputs, targets)
        loss_value = loss.item()
        self.valid_acc.update(outputs, targets)
        self.log("Accuracy/val", self.valid_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("Loss/val", loss_value, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        samples, targets = batch
        outputs = self.forward(samples)
        self.confusion_matrix.update(outputs, targets)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        opt: Optimizer = self.optimizers()
        self.log("LR", opt.param_groups[0]["lr"], on_epoch=True, sync_dist=True)

    def on_test_end(self) -> None:
        self.visualize_confusion_matrix()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt=self.hparams.opt,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler, _ = create_scheduler(self.hparams, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, optimizer_idx, metric) -> None:
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def visualize_confusion_matrix(self):
        cf_matrix = self.confusion_matrix.compute().cpu()
        categories = [f'C{i}' for i in range(self.hparams.num_classes)]
        fig, ax = plt.subplots(1)
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.2f', xticklabels=categories, yticklabels=categories)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True Label')
        vis_path = Path(self.hparams.vis_path)
        fig.savefig(str(vis_path / f"cf_matrix.png"), dpi=200)
    

def cli_main():
    cli = LightningCLI(ViTASDLM,
                       AutismDatasetModule,
                       seed_everything_default=42,
                       trainer_defaults=dict(accelerator='gpu', devices=1),
                       save_config_overwrite=True,
    )


if __name__ == "__main__":
    cli_main()