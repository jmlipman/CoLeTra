from typing import Union, List
from pathlib import Path
import numpy as np
import yaml
from monai.transforms import Activations
import nibabel as nib

class Pulmonary:

    classes = ["va"]
    ext = "nii.gz"
    dim = 3 # Dimensions, 3D

    def __init__(self, id: str) -> None:
        self.id = id

    def split(self, path_split: Path, fold: int) -> Union[List, List, List]:

        with open(path_split, 'r') as f:
            split = yaml.load(f, Loader=yaml.SafeLoader)[fold]


        trainFiles = split['train']
        valFiles = split['train']
        testFiles = split['test'] if not split['test'] is None else []
        train, val, test = [], [], []

        for t in trainFiles:
            label_path = str(Path(t).parents[1] / "labelsTr" / Path(t).name.replace("_0000", ""))
            inpainted = str(Path(t).parents[1] / "inpaintedTr" / f"{str(int(Path(t).name.split('_')[1]))}_image_inpainted_v1.nii.gz")

            train.append({'image': t, 'label': label_path,
                'inpainted_v1': inpainted})
        for t in valFiles:
            label_path = str(Path(t).parents[1] / "labelsTr" / Path(t).name.replace("_0000", ""))
            val.append({'image': t, 'label': label_path, 'id': Path(t).name.split(".")[0]})
        for t in testFiles:
            # This is hard-coded for evaluating the results
            label_path = str(Path(t).parents[1] / "labelsOneClass" / Path(t).name.replace("_0000", ""))
            test.append({'image': t, 'label': label_path, 'id': Path(t).name.split(".")[0]})

        return train, val, test

    def savePrediction(self, pred: np.ndarray, outputPath: Path,
                       subjects_id: List[str]) -> None:
        # pred.shape = B,C,H,W(,D)

        affine = np.eye(4)
        affine[0,0] = -1.0
        affine[1,1] = -1.0
        affine[2,2] = 1.0
        affine[:, -1] = 1.0

        for i in range(len(subjects_id)):
            outputFilepath = outputPath / f'{subjects_id[i]}.{self.ext}'

            pred_im = np.argmax(pred[i], axis=0).astype("uint8")

            im = nib.Nifti1Image(pred_im, affine)
            im.header.set_xyzt_units("micron")
            nib.save(im, outputFilepath)

