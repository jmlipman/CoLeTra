from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import one_hot

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+smooth)/(torch.sum(skel_true[:,1:,...])+smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)

class soft_dice_cldice(nn.Module):
    def __init__(self, ignore_class: int | None, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.ignore_class = ignore_class
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):

        if isinstance(self.ignore_class, int):
            consider_idx = y_true!=self.ignore_class
            y_true[~consider_idx] = 0
        else:
            consider_idx = torch.ones(y_true.shape).to(y_true.device)

        y_pred = torch.softmax(y_pred, 1)
        y_true = one_hot(y_true, num_classes=y_pred.shape[1])

        # Dice loss
        smooth = 1
        axis = list([i for i in range(2, len(y_true.shape))])
        num = 2 * torch.sum(y_pred * y_true * consider_idx, axis=axis) + smooth
        denom = torch.sum(y_true * consider_idx, axis=axis) + torch.sum(y_pred * consider_idx, axis=axis) + smooth
        dice = torch.mean(1 - (num / denom))

        #from IPython import embed; embed(); asd
        # cl Dice
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum((skel_pred * y_true * consider_idx)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...] * consider_idx)+self.smooth)
        tsens = (torch.sum((skel_true * y_pred * consider_idx)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...] * consider_idx)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0-self.alpha)*dice+self.alpha*cl_dice

class soft_dice_cldice_asd(nn.Module):
    def __init__(self, ignore_class: int | bool, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.ignore_class = ignore_class
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        #if isinstance(self.ignore_class, int):
        #    consider_idx = y_true!=self.ignore_class
        #    y_true[~consider_idx] = 0
        #else:
        #    consider_idx = torch.ones(y_true.shape)
        #y_true[y_true==self.ignore_class] = 0
        #print(y_pred.max(), y_pred.min())
        y_pred = torch.softmax(y_pred, 1)
        #print(y_pred.max(), y_pred.min())
        #print(y_true.shape)
        y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        #print(y_true.shape)
        #print(y_pred.shape, y_true.shape)
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        #return cl_dice
        return (1.0-self.alpha)*dice+self.alpha*cl_dice

class soft_dice_cldice_orig(nn.Module):
    def __init__(self, ignore_class: int | bool, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.ignore_class = ignore_class
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel
