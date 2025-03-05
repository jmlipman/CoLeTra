from typing import Union, List
from pathlib import Path
import numpy as np
import yaml
from monai.transforms import Activations
from PIL import Image

class Crack500:

    classes = ["crack"]
    ext = "png"
    dim = 2

    label_folders = {
            "gt": "1st_manual",
            }


    def __init__(self, id: str) -> None:
        self.id = id

    def split(self, path_split: Path, fold: int) -> Union[List, List, List]:

        with open(path_split, 'r') as f:
            split = yaml.load(f, Loader=yaml.SafeLoader)[fold]

        trainFiles = [p for p in split['train'] if "/train/" in p]
        valFiles = [p for p in split['train'] if "/validation/" in p]
        testFiles = split['test'] if not split['test'] is None else []
        train, val, test = [], [], []

        for t in trainFiles:
            label_path = Path(t.replace(".png", "_mask.png"))
            inpainted = Path(t.replace("/train/", "/train_inpaintedv1/"))
            train.append({'image': t, 'label': label_path,
                'inpainted': inpainted})
        for t in valFiles:
            label_path = Path(t.replace(".png", "_mask.png"))
            val.append({'image': t, 'label': label_path, 'id': Path(t).stem})
        for t in testFiles:
            label_path = t.replace(".png", "_mask.png")
            test.append({'image': t, 'label': label_path, 'id': Path(t).stem})

        return train, val, test

    def savePrediction(self, pred: List[np.ndarray], outputPath: Path,
                       subjects_id: List[str]) -> None:
        # pred.shape = B,C,H,W(,D)

        for i in range(len(subjects_id)):
            outputFilepath = outputPath / f'{subjects_id[i]}.{self.ext}'

            #pred_tmp = np.fliplr(np.rot90(np.uint8(pred[i][1])))
            pred_tmp = np.flipud(np.rot90(np.uint8(pred[i][1])))
            Image.fromarray(pred_tmp[:, :]*255, 'L').save(outputFilepath)

