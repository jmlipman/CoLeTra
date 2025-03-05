#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================

import time
import numpy as np
import gudhi as gd
import torch
from torch.nn.modules.loss import _Loss
import math
from lib.losses import MyDiceCELoss
import nibabel as nib


def compute_dgm_force(lh_dgm, gt_dgm, pers_thd=0.03, pers_thd_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thd: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thd_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if (gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = len(gt_pers)  # number of holes in gt

    if (gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = []
        idx_holes_to_remove = list(range(len(lh_pers)))
        idx_holes_perfect = []
    else:
        tmp = lh_pers > pers_thd_perfect  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
            lh_n_holes_perfect = tmp.sum()
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            idx_holes_perfect = []

        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];

        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if (do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove

def getCriticalPoints(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()

    lh_cubic = gd.CubicalComplex(
        #dimensions=[lh.shape[0], lh.shape[1]],
        dimensions=lh.shape, # allows 3D
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return 0, 0, 0, False
    if (len(pairs_lh[0][0])==0): return 0, 0, 0, False

    # return persistence diagram, birth/death critical points
    pd_lh = np.array([[lh_vector[pairs_lh[0][0][i][0]], lh_vector[pairs_lh[0][0][i][1]]] for i in range(len(pairs_lh[0][0]))])
    # Make it 3D compatible
    #bcp_lh = np.array([[pairs_lh[0][0][i][0]//lh.shape[1], pairs_lh[0][0][i][0]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    #dcp_lh = np.array([[pairs_lh[0][0][i][1]//lh.shape[1], pairs_lh[0][0][i][1]%lh.shape[1]] for i in range(len(pairs_lh[0][0]))])
    bcp_lh = np.array(np.unravel_index(pairs_lh[0][0][:, 0], likelihood.shape)).T
    dcp_lh = np.array(np.unravel_index(pairs_lh[0][0][:, 1], likelihood.shape)).T

    if len(pd_lh.shape) != 2 or pd_lh.shape[1] != 2:
        print("pd_lh.shape", pd_lh.shape)
        from IPython import embed; embed(); asd

    return pd_lh, bcp_lh, dcp_lh, True

class TopoNetLoss(_Loss):
    """
    Calculate the topology loss of the predicted image and ground truth image
    Warning: To make sure the topology loss is able to back-propagation, likelihood
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood_tensor:   The likelihood pytorch tensor.
        gt_tensor        :   The groundtruth of pytorch tensor.
        topo_size        :   The size of the patch is used. Default: 100

    Returns:
        loss_topo        :   The topology loss value (tensor)

    """
    def __init__(self, topo_size, lambd, activate_at_iteration, weights=None):
        super().__init__()
        self.topo_size = topo_size
        self.lambd = lambd # CE+lambda*topoloss
        self.activate_at_iteration = activate_at_iteration
        self.celossfun = MyDiceCELoss(ignore_class=2, lambda_ce=1, lambda_dice=0, weights=weights)
        self.activate = False

    def forward(self, likelihood_tensor: torch.Tensor,
                gt_tensor: torch.Tensor) -> torch.Tensor:

        celoss = self.celossfun(likelihood_tensor, gt_tensor)
        if not self.activate:
            return celoss

        if len(likelihood_tensor.shape) == 5:
            topoloss = self._topoloss_3D(likelihood_tensor, gt_tensor)
        else:
            topoloss = self._topoloss_2D(likelihood_tensor, gt_tensor)

        return celoss + self.lambd * topoloss

    def _topoloss_2D(self, likelihood_tensor, gt_tensor):

        likelihood_tensor = torch.softmax(likelihood_tensor, 1)[:, 1]
        likelihood = likelihood_tensor.clone().cpu().detach().numpy()
        gt = gt_tensor[:, 0].clone().cpu().detach().numpy()
        # likelihood \in [0,1]
        # gt \in {0,1}

        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)

        idxs = [(b, y, x) for b in range(likelihood.shape[0]) for y in range(0, likelihood.shape[1], self.topo_size) for x in range(0, likelihood.shape[2], self.topo_size)]

        #for y in range(0, likelihood.shape[0], self.topo_size):
        #    for x in range(0, likelihood.shape[1], self.topo_size):
        #        for z in range(0, likelihood.shape[2], self.topo_size):

        for (b, y, x) in idxs:

            # 3D compatible
            lh_patch = likelihood[b, y:min(y + self.topo_size, likelihood.shape[1]),
                         x:min(x + self.topo_size, likelihood.shape[2])]
            gt_patch = gt[b, y:min(y + self.topo_size, gt.shape[1]),
                         x:min(x + self.topo_size, gt.shape[2])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thd=0.03)

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    #from IPython import embed; embed(); asd
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map[b, iy, ix] = 1
                        topo_cp_ref_map[b, iy, ix] = 0

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(dcp_lh[hole_indx][0])
                        ix = x + int(dcp_lh[hole_indx][1])

                        # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map[b, iy, ix] = 1
                        topo_cp_ref_map[b, iy, ix] = 1

                for hole_indx in idx_holes_to_remove:
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():

                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        # push birth to death  # push to diagonal
                        topo_cp_weight_map[b, iy, ix] = 1

                        if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(dcp_lh[hole_indx][0])
                            iix = int(dcp_lh[hole_indx][1])

                            topo_cp_ref_map[b, iy, ix] = lh_patch[iiy, iix]
                        else:
                            topo_cp_ref_map[b, iy, ix] = 1

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = int(dcp_lh[hole_indx][0])
                        ix = int(dcp_lh[hole_indx][1])

                        # push death to birth # push to diagonal
                        topo_cp_weight_map[b, iy, ix] = 1

                        if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(bcp_lh[hole_indx][0])
                            iix = int(bcp_lh[hole_indx][1])

                            topo_cp_ref_map[b, iy, ix] = lh_patch[iiy, iix]
                        else:
                            topo_cp_ref_map[b, iy, ix] = 0

        ####
        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).mean()
        return loss_topo

    def _topoloss_3D(self, likelihood_tensor, gt_tensor):

        likelihood_tensor = torch.softmax(likelihood_tensor, 1)[:, 1]
        likelihood = likelihood_tensor.clone().cpu().detach().numpy()
        gt = gt_tensor[:, 0].clone().cpu().detach().numpy()
        # likelihood \in [0,1]
        # gt \in {0,1}

        topo_cp_weight_map = np.zeros(likelihood.shape)
        topo_cp_ref_map = np.zeros(likelihood.shape)

        idxs = [(b, y, x, z) for b in range(likelihood.shape[0]) for y in range(0, likelihood.shape[1], self.topo_size) for x in range(0, likelihood.shape[2], self.topo_size) for z in range(0, likelihood.shape[3], self.topo_size)]

        #for y in range(0, likelihood.shape[0], self.topo_size):
        #    for x in range(0, likelihood.shape[1], self.topo_size):
        #        for z in range(0, likelihood.shape[2], self.topo_size):

        for (b, y, x, z) in idxs:

            # 3D compatible
            lh_patch = likelihood[b, y:min(y + self.topo_size, likelihood.shape[1]),
                         x:min(x + self.topo_size, likelihood.shape[2]),
                         z:min(z + self.topo_size, likelihood.shape[3])]
            gt_patch = gt[b, y:min(y + self.topo_size, gt.shape[1]),
                         x:min(x + self.topo_size, gt.shape[2]),
                         z:min(z + self.topo_size, gt.shape[3])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thd=0.03)

            #from IPython import embed; embed(); asd

            if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                for hole_indx in idx_holes_to_fix:
                    #from IPython import embed; embed(); asd
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        iz = z + int(bcp_lh[hole_indx][2])
                        # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map[b, iy, ix, iz] = 1
                        topo_cp_ref_map[b, iy, ix, iz] = 0

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = y + int(dcp_lh[hole_indx][0])
                        ix = x + int(dcp_lh[hole_indx][1])
                        iz = z + int(dcp_lh[hole_indx][2])

                        # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map[b, iy, ix, iz] = 1
                        topo_cp_ref_map[b, iy, ix, iz] = 1

                for hole_indx in idx_holes_to_remove:
                    if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():

                        iy = y + int(bcp_lh[hole_indx][0])
                        ix = x + int(bcp_lh[hole_indx][1])
                        iz = z + int(bcp_lh[hole_indx][2])
                        # push birth to death  # push to diagonal
                        topo_cp_weight_map[b, iy, ix, iz] = 1

                        if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(dcp_lh[hole_indx][0])
                            iix = int(dcp_lh[hole_indx][1])
                            iiz = int(dcp_lh[hole_indx][2])

                            topo_cp_ref_map[b, iy, ix, iz] = lh_patch[iiy, iix, iiz]
                        else:
                            topo_cp_ref_map[b, iy, ix, iz] = 1

                    if ((dcp_lh[hole_indx] >= 0) & (dcp_lh[hole_indx] < likelihood.shape[1:])).all():
                        iy = int(dcp_lh[hole_indx][0])
                        ix = int(dcp_lh[hole_indx][1])
                        iz = int(dcp_lh[hole_indx][2])

                        # push death to birth # push to diagonal
                        topo_cp_weight_map[b, iy, ix, iz] = 1

                        if ((bcp_lh[hole_indx] >= 0) & (bcp_lh[hole_indx] < likelihood.shape[1:])).all():
                            iiy = int(bcp_lh[hole_indx][0])
                            iix = int(bcp_lh[hole_indx][1])
                            iiz = int(bcp_lh[hole_indx][2])

                            topo_cp_ref_map[b, iy, ix, iz] = lh_patch[iiy, iix, iiz]
                        else:
                            topo_cp_ref_map[b, iy, ix, iz] = 0

        ####
        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

        # Measuring the MSE loss between predicted critical points and reference critical points
        loss_topo = (((likelihood_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).mean()
        return loss_topo

