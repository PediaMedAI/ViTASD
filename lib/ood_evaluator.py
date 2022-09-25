import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from datasets import get_dataset


class OODEvaluator:
    def __init__(self, in_dataset, out_dataset, model, data_root, num_workers, batch_size):
        self.in_ds_name = in_dataset
        self.out_ds_name = out_dataset
        in_dataset = get_dataset(in_dataset + "/test", root=data_root)
        out_dataset = get_dataset(out_dataset + "/test", root=data_root)
        out_dataset.transform = in_dataset.transform
        datasets = [in_dataset, out_dataset]
        self.targets = torch.cat([torch.zeros(len(in_dataset)), torch.ones(len(out_dataset))])
        self.concat_dataset = torch.utils.data.ConcatDataset(datasets)
        self.dataloader = torch.utils.data.DataLoader(
            self.concat_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        self.model: torch.nn.Module = model

    def loop_over_datasets(self):
        self.model.eval()
        with torch.no_grad():
            scores = []
            for x, _ in self.dataloader:
                x = x.cuda()
                y_pred = F.softmax(self.model(x), dim=1)
                uncertainty = -(y_pred * y_pred.log()).sum(1)
                scores.append(uncertainty.detach().cpu().numpy())

        scores = np.concatenate(scores)
        return scores

    def get_ood_metrics(self):
        scores = self.loop_over_datasets()
        auroc = roc_auc_score(y_true=self.targets, y_score=scores)
        precision, recall, _ = precision_recall_curve(y_true=self.targets, probas_pred=scores)
        aupr = auc(recall, precision)
        return auroc, aupr