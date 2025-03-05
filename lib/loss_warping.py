import cv2, time
import numpy as np
from scipy import ndimage
import imageio
import torch
import cc3d
# loss functions
# apply softmax on network output for dice, not CE
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss
from monai.networks import one_hot


def decide_simple_point_2D(gt, x, y):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2]
    if patch.shape != (3,3):
        return gt

    ## check local topology
    number_fore, _ = cv2.connectedComponents(patch, 4)
    number_back, _ = cv2.connectedComponents(1-patch, 8)

    label = (number_fore-1) * (number_back-1)

    ## flip the simple point
    if (label == 1):
        gt[x,y] = 1 - gt[x,y]

    return gt

def decide_simple_point_3D(gt, x, y, z):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2, z-1:z+2]
    if patch.shape != (3,3,3):
        return gt

    ## check local topology
    if patch.shape[0] != 0 and patch.shape[1] != 0 and patch.shape[2] != 0:
        try:
            _, number_fore = cc3d.connected_components(patch, 6, return_N = True)
            _, number_back = cc3d.connected_components(1-patch, 26, return_N = True)
        except:
            number_fore = 0
            number_back = 0
            pass
        label = number_fore * number_back

        ## flip the simple point
        if (label == 1):
            gt[x,y,z] = 1 - gt[x,y,z]

    return gt

def update_simple_point(distance, gt):
    non_zero = np.nonzero(distance)
    # indice = np.argsort(-distance, axis=None)
    indice = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

    if len(gt.shape) == 2:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_2D(gt, x, y)
    else:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]
            z = indice[2][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_3D(gt, x, y, z)
    return gt


class WarpingLoss(_Loss):
    """
    Calculate the warping loss of the predicted image and ground truth image
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative
    def __init__(self, lambd, ignore_class, activate_at_iteration):
        super().__init__()
        self.lambd = lambd
        self.ignore_class = ignore_class
        self.activate_at_iteration = activate_at_iteration
        self.activate = False

    def forward(self, y_pred: torch.Tensor,
                y_gt: torch.Tensor) -> torch.Tensor:
        #self.c += 1

        if isinstance(self.ignore_class, int):
            consider_idx = y_gt!=self.ignore_class
            y_gt[~consider_idx] = 0
        else:
            consider_idx = torch.ones(y_gt.shape).to(y_gt.device)

        target = one_hot(y_gt, num_classes=y_pred.shape[1])
        y_pred = softmax_helper(y_pred)

        # Partial dice
        smooth = 1
        axis = list([i for i in range(2, len(target.shape))])
        num = 2 * torch.sum(y_pred * target * consider_idx, axis=axis) + smooth
        denom = torch.sum(target * consider_idx, axis=axis) + torch.sum(y_pred * consider_idx, axis=axis) + smooth
        dice_loss = torch.mean(1 - (num / denom))
        if not self.activate:
            return dice_loss

        ce_loss = RobustCrossEntropyLoss()

        if (len(y_pred.shape) == 4):
            B, C, H, W = y_pred.shape
            critical_points = np.zeros((B,H,W))
        else:
            B, C, H, W, Z = y_pred.shape
            critical_points = np.zeros((B,H,W,Z))

        pre = torch.argmax(y_pred, dim=1)
        y_gt = target[:, 1:2]
        gt = target[:, 1]

        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy() # BHW
        gt_copy = gt.copy() # BHW

        for i in range(B):
            false_positive = ((pre_copy[i] - gt_copy[i]) == 1).astype(int)
            false_negative = ((gt_copy[i] - pre_copy[i]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance_gt = ndimage.distance_transform_edt(gt_copy[i]) * false_negative  # shrink gt while keep connected
            false_positive_distance_gt = ndimage.distance_transform_edt(1 - gt_copy[i]) * false_positive  # grow gt while keep unconnected

            gt_warp = update_simple_point(false_negative_distance_gt, gt_copy[i])
            gt_warp = update_simple_point(false_positive_distance_gt, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy[i]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(1-pre_copy[i]) * false_negative # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i] = np.logical_or(np.not_equal(pre[i], gt_warp), np.not_equal(gt[i], pre_warp)).astype(int)

        warping_loss = ce_loss(y_pred * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda(), y_gt * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda())


        loss = dice_loss + self.lambd*warping_loss
        return loss


softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

        # print("batch_dice: {}\ndo_bg: {}\n".format(self.batch_dice, self.do_bg))

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        #print("[SAUMDEBUG]\naxes: {}\n".format(axes))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            #print("[SAUMDEBUG]\napply_nonlin called")

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        #print("[SAUMDEBUG]\ndc without manipulation: {}\n -dc: {}\n".format(dc, -dc))
        return -dc

class IOU(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(IOU, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

        # print("batch_dice: {}\ndo_bg: {}\n".format(self.batch_dice, self.do_bg))

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        #print("[SAUMDEBUG]\naxes: {}\n".format(axes))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
            #print("[SAUMDEBUG]\napply_nonlin called")

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + fp + fn + self.smooth

        iou = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                diouc = iou[:, 1:]
        iou = iou.mean()
        #print("[SAUMDEBUG]\ndc without manipulation: {}\n -dc: {}\n".format(dc, -dc))
        return iou

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            #print("[SAUMDEBUG]\nDC_and_CE_loss\nweight_ce: {}\nce_loss: {}\nweight_dice: {}\ndc_loss: {}\n".format(self.weight_ce, ce_loss, self.weight_dice, dc_loss))
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result



