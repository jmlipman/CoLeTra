from __future__ import annotations
import torch, os, subprocess
from monai.metrics import CumulativeIterationMetric
from monai.metrics import SurfaceDistanceMetric as MONAI_SDM
from pathlib import Path
from monai.utils import MetricReduction
import numpy as np
import time, random
import nibabel as nib
import gudhi as gd
from skimage.morphology import skeletonize, skeletonize_3d
from medpy import metric
from monai.networks import one_hot
import itertools

class HD95Metric(CumulativeIterationMetric):

    def __init__(self, voxres) -> None:
        super().__init__()
        self._buffers = None
        self.voxres = voxres

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        hd95_foreground = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                #from IPython import embed; embed(); asd
                hd95_foreground.append( metric.hd95(y_pred[b,c], y_true[b,0]==c,
                                                    self.voxres) )
        return torch.tensor([hd95_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self._buffers[0], dim=0)


class DiceMetric(CumulativeIterationMetric):
    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        dice_foreground = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                dice_foreground.append( metric.dc(y_pred[b, c], y_true[b,0]==c) )
        return torch.tensor([dice_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self._buffers[0], dim=0)
        #return self.get_buffer()

class clDiceMetric(CumulativeIterationMetric):

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _cl_score(self, segmentation, skeleton):
        """Computes the skeleton volume overlap.
        """
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().detach().numpy()
        return np.sum(segmentation*skeleton)/np.sum(skeleton)

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        clDice_foreground = []
        for b in range(y_pred.shape[0]): # For each image
            for c in range(1, y_pred.shape[1]): # For each class

                if len(y_pred.shape) == 4:
                    tprec = self._cl_score(y_pred[b,c],
                                           skeletonize(y_true[b,0]==c))
                    tsens = self._cl_score(y_true[b,0]==c,
                                           skeletonize(y_pred[b,c]))

                elif len(y_pred.shape) == 5:
                    tprec = self._cl_score(y_pred[b,c],
                                           skeletonize_3d(y_true[b,0]==c))
                    tsens = self._cl_score(y_true[b,0]==c,
                                           skeletonize_3d(y_pred[b,c]))

                clDice_foreground.append( 2*tprec*tsens/(tprec+tsens) )

        return torch.tensor([clDice_foreground])

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self._buffers[0], dim=0)

class BettiErrorMetric(CumulativeIterationMetric):
    # Local only. The global betti error is computed separately with another script.
    max_patches_3d = 100
    max_patches_2d = 500
    patch_2d = [64, 64]
    patch_3d = [48, 48, 48]

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        if len(y_pred.shape) == 4:
            self.max_patches = self.max_patches_2d
        else:
            self.max_patches = self.max_patches_3d

        all_betti_err = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                differences = []
                patch_coor = self._get_patches(y_true[b, 0]==c)
                for coor in patch_coor:
                    betti_true = self._compute_betti(y_true[b, 0][coor]==c)
                    betti_pred = self._compute_betti(y_pred[b, c][coor])
                    diff_tmp = np.abs(np.array(betti_true[:-1]) - np.array(betti_pred[:-1]))
                    differences.append(diff_tmp)
            all_betti_err.append( differences )
        all_betti_err = np.array([all_betti_err])
        all_betti_err = np.mean(all_betti_err, axis=2)

        return torch.tensor(all_betti_err)

    def _compute_betti(self, patch: np.array) -> List[int]:
        cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
        cc.compute_persistence()
        bnum = cc.persistent_betti_numbers(np.inf, -np.inf)
        return bnum

    def _get_patches(self, y_true: np.array):

        all_slices = []
        if ( (len(y_true.shape) == 3 and
             np.prod(y_true.shape)/np.prod(self.patch_3d) > self.max_patches) or
             (len(y_true.shape) == 2 and
             np.prod(y_true.shape)/np.prod(self.patch_2d) > self.max_patches) ):
             # The image is very large, so, there would be more patches
             # than self.max_patches. Thus, I will get random patches around
             # the pixels/voxels with value "1" in the ground truth.

             ones_idx = np.where(y_true)
             random_idx = random.sample(list(np.arange(len(ones_idx[0]))),
                                        self.max_patches)
             if len(y_true.shape) == 3:
                 for i in random_idx:
                     ys = np.max([ones_idx[0][i]-self.patch_3d[0], 0])
                     ye = ones_idx[0][i]+self.patch_3d[0]
                     xs = np.max([ones_idx[1][i]-self.patch_3d[1], 0])
                     xe = ones_idx[1][i]+self.patch_3d[1]
                     zs = np.max([ones_idx[2][i]-self.patch_3d[2], 0])
                     ze = ones_idx[2][i]+self.patch_3d[2]
                     all_slices.append((slice(ys, ye), slice(xs, xe), slice(zs, ze)))
             else:
                 for i in random_idx:
                     ys = np.max([ones_idx[0][i]-self.patch_2d[0], 0])
                     ye = ones_idx[0][i]+self.patch_2d[0]
                     xs = np.max([ones_idx[1][i]-self.patch_2d[1], 0])
                     xe = ones_idx[1][i]+self.patch_2d[1]
                     all_slices.append((slice(ys, ye), slice(xs, xe)))
        else:
            # The image is small enough, so, we can extract a number of
            # non-overlapping patches smaller than self.max_patches
            if len(y_true.shape) == 3:
                yss = list(range(0, y_true.shape[0], self.patch_3d[0]))
                xss = list(range(0, y_true.shape[1], self.patch_3d[1]))
                zss = list(range(0, y_true.shape[2], self.patch_3d[2]))
                mixthis = [yss, xss, zss]
                ps = self.patch_3d
            else:
                yss = list(range(0, y_true.shape[0], self.patch_2d[0]))
                xss = list(range(0, y_true.shape[1], self.patch_2d[1]))
                mixthis = [yss, xss]
                ps = self.patch_2d
            coors_list = list(itertools.product(*mixthis))
            for coors in coors_list:
                all_slices.append( tuple([slice(ss, ss+ps[i]) for i,ss in enumerate(coors)]) )

        return all_slices


    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return torch.cat(self._buffers[0], dim=0)

