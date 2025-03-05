from __future__ import annotations

import numpy as np
import torch, warnings
from torch.nn.modules.loss import _Loss
from monai.losses import DiceCELoss
from monai.networks import one_hot
from typing import Union
from scipy.ndimage import distance_transform_edt as dist

class RegionWiseLoss(_Loss):

    def __init__(self, rwmap_type) -> None:
        super().__init__()
        if rwmap_type == "rrwmap":
            self.compute_rwmap = self._compute_rrwmap
        elif rwmap_type == "rwmap_ci_095005":
            self.compute_rwmap = self._compute_rwmap_ci
            self.ratio = [0.95, 0.05] # Yes, it is not [0.05, 0.95]
        else:
            raise ValueError(f"Unknown rwmap: `{rwmap_type}`")

    def _compute_rwmap_acloss(self, y_true: np.array):
        return 1 - y_true

    def _compute_rwmap_ci(self, y_true: np.array):

        rrwmap = np.zeros_like(y_true) # Y: one-hot encoded ground truth
        for b in range(rrwmap.shape[0]): # Batch dim
            for c in range(rrwmap.shape[1]): # Channel dim
                rrwmap[b, c] = dist(y_true[b, c])
                rrwmap[b, c] = -1 * (rrwmap[b, c] / (np.max(rrwmap[b, c] + 1e-15)))
                rrwmap[b, rrwmap[b]==0] = self.ratio[c]

        return rrwmap

    def _compute_rrwmap(self, y_true: np.array):

        rrwmap = np.zeros_like(y_true) # Y: one-hot encoded ground truth
        for b in range(rrwmap.shape[0]): # Batch dim
            for c in range(rrwmap.shape[1]): # Channel dim
                rrwmap[b, c] = dist(y_true[b, c])
                rrwmap[b, c] = -1 * (rrwmap[b, c] / (np.max(rrwmap[b, c] + 1e-15)))
        rrwmap[rrwmap==0] = 1
        return rrwmap

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = torch.softmax(input, 1)
        gt = target[:,0].cpu().detach().numpy() # it's non-hot-encoded
        gt = np.stack([gt==i for i in range(input.shape[1])], axis=1)*1.0
        rwmap = torch.tensor(self.compute_rwmap(gt), device=input.device)
        loss = torch.mean(input * rwmap)
        return loss

class MyDiceCELoss(_Loss):
    """Own implementation of Dice/CE loss.
    """
    def __init__(self, lambda_ce: float=1,
            lambda_dice: float=1, weights: List[float]=None):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        #from IPython import embed; embed(); asd
        input = torch.softmax(input, 1)
        target = one_hot(target, num_classes=input.shape[1])

        # Dice loss
        dice_loss = 0
        if self.lambda_dice > 0:
            smooth = 1
            axis = list([i for i in range(2, len(target.shape))])
            num = 2 * torch.sum(input * target, axis=axis) + smooth
            denom = torch.sum(target, axis=axis) + torch.sum(input, axis=axis) + smooth
            dice_loss = torch.mean(1 - (num / denom))

        ce_loss = 0
        if self.lambda_ce > 0:
            # CE loss
            if self.weights:
                w = torch.Tensor(np.array(self.weights)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(target.device)
                if len(target.shape) == 5:
                    w = w.unsqueeze(-1)
                ce = torch.sum(w*target * torch.log(input + 1e-15), axis=1)
            else:
                ce = torch.sum(target * torch.log(input + 1e-15), axis=1)
            ce_loss = -torch.mean(ce)

        return dice_loss*self.lambda_dice + ce_loss*self.lambda_ce

