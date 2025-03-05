from __future__ import annotations

from typing import List
from collections.abc import Hashable, Mapping
from monai.config import NdarrayOrTensor
from monai.transforms import RandomizableTransform, MapTransform
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta
import random, torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import cv2, time

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel

def getRandomCoordinates(labelmap: NdarrayOrTensor, num: int,
        size: List[int]) -> List[Tuple]:
    allcoors = np.where(labelmap)
    if len(allcoors[0]) == 0:
        return []

    min_size = np.array(size)/2
    max_size = np.array(labelmap.shape) - min_size - 1

    coors = []
    while len(coors) < num:
        idx = random.randint(0, len(allcoors[0])-1)
        arr_coor = np.array([allcoors[i][idx] for i in range(len(allcoors))])

        if (arr_coor >= min_size).all() and (arr_coor <= max_size).all():
            coors.append( tuple(arr_coor) )

    return coors

def getNumberOfHoles(holes):
    if isinstance(holes, int):
        return holes
    return random.randint(min(holes), max(holes))

def getSize(size, dims):
    if isinstance(size, list):
        if len(size) == dims:
            return size
        msg = f"size specified to be {size} but the data has {dims} dimensions."
        raise ValueError(msg)
    else:
        return [size for _ in range(dims)]

def coor2slices(coors, size: List[int]):
    slices = []
    isodd = [s%2 for s in size]

    for coor in coors:
        sl = [slice(max(c-s//2, 0), c+s//2+o) for c,s,o in zip(coor, size, isodd)]
        # Adding the channel dimension
        slices.append(tuple([slice(0, None)] + sl))
    return slices

class CoLeTra(RandomizableTransform):
    def __init__(self, prob: float, mix_ratio: List, ws: int) -> None:
        RandomizableTransform.__init__(self, prob)
        self.mix_ratio = mix_ratio
        self.ws = ws

    def __call__(self, img: NdarrayOrTensor, inpainted: NdarrayOrTensor,
            slices: List) -> NdarrayOrTensor:

        if self.mix_ratio[0] == -1 or self.mix_ratio[1] == -1:
            mx1 = np.random.random()
            mx2 = 1-mx1
        elif self.mix_ratio[0] == -2 or self.mix_ratio[1] == -2:
            mx2 = torch.tensor(gkern(l=self.ws, sig=3))
            mx1 = 1 - mx2
        else:
            mx1 = self.mix_ratio[0]
            mx2 = self.mix_ratio[1]


        for sl in slices:
            img[sl] = img[sl]*mx1 + inpainted[sl]*mx2

        return img
        #from IPython import embed; embed(); asd

class MyNormalizeIntensityd(MapTransform):
    # It is important to normalize the inpainted image in the same way as
    # the original image
    def __init__(self, keys: List[str], ref: str|None=None) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.ref = ref

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.ref:
            mean = d[self.ref].mean()
            std = d[self.ref].std()

        for key in self.key_iterator(d):
            if self.ref:
                d[key] = (d[key]-mean)/(std)
            else:
                d[key] = (d[key]-d[key].mean())/(d[key].std())
        return d


class CoLeTraTransformd(RandomizableTransform, MapTransform):

    def __init__(self, key_images: List[str], key_label: str,
            key_label_class: int,
            holes: List[int] | int, size: List[int],
            fill: str="", fill_type: str="",
            prob: float=1) -> None:

        MapTransform.__init__(self, key_images)
        RandomizableTransform.__init__(self, prob=prob)

        if not isinstance(key_label, str):
            raise ValueError("`key_label` must be a string")
        if not isinstance(key_label_class, int):
            raise ValueError("`key_label_class` must be an int")
        if isinstance(holes, list):
            if len(holes) != 2:
                raise ValueError("`holes` must be of size 2 (min,max)")
            for holep in holes:
                if not isinstance(holep, int):
                    raise ValueError("`holes` must be an int or a list of ints")
        elif not isinstance(holes, int):
            raise ValueError("`holes` must be an int or a list of ints")
        if isinstance(size, list):
            if not len(size) in [2, 3]:
                msg = (f"`size` must have two or three elements, i.e.,"
                        "one per dimension")
                raise ValueError(msg)
            for i, el in enumerate(size):
                if not isinstance(el, int):
                    msg = f"The element {i} in `size` is not an int"
                    raise ValueError(msg)
        else:
            if not isinstance(size, int):
                raise ValueError("`size` must be an int or a list of ints")
        if not isinstance(fill, str):
            raise ValueError("`fill` must be a string")
        if not isinstance(fill_type, (str, list)):
            raise ValueError("`fill_type` must be a string or a list")

        err_msg = (f"`fill_type` must be a list "
                    "of floats or ints that sum up to 1. "
                    "Given `fill_type`={fill_type}")
        if not isinstance(fill_type, list) or len(fill_type) != 2:
            raise ValueError(err_msg)
        if not isinstance(fill_type[0], (int, float)) or not isinstance(fill_type[1], (int, float)):
            raise ValueError(err_msg)

        self.key_label = key_label
        self.key_label_class = key_label_class
        self.holes = holes
        self.size = size
        self.fill = fill
        self.fill_type = fill_type

        self.transform = CoLeTra(prob=1.0, mix_ratio=fill_type, ws=size[0])


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # size: CHW(d). At this point, C=1 because it's not onehot encoded
        labelmap = d[self.key_label] == self.key_label_class
        num_holes = getNumberOfHoles(self.holes)
        size = getSize(size=self.size, dims=len(d[self.key_label].shape[1:]))

        # Center coordinates where the boxes will be located
        coors = getRandomCoordinates(labelmap[0], num_holes, size)
        # Slices (with start:end) of those boxes
        slices = coor2slices(coors, size)


        for key in self.key_iterator(d):

            d[key] = self.transform(
                    convert_to_tensor(d[key], track_meta=get_track_meta()), # Image
                    d['inpainted'], slices)

        return d

