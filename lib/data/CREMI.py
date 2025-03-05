from typing import Union, List
from pathlib import Path
import numpy as np
import yaml
from monai.transforms import Activations
import nibabel as nib

class CREMI:

    classes = ["border"]
    ext = "nii.gz"
    dim = 3 # Dimensions, 3D

    def __init__(self, id: str) -> None:
        self.lab_thr = "4"

    def split(self, path_split: Path, fold: int) -> Union[List, List, List]:

        with open(path_split, 'r') as f:
            split = yaml.load(f, Loader=yaml.SafeLoader)[fold]

        trainFiles = split['train'][0:1]
        valFiles = split['train'][1:2]
        testFiles = split['test'] if not split['test'] is None else []
        train, val, test = [], [], []

        for t in trainFiles:
            label_path = (Path(t).parents[1] / "labels" / Path(t).name.replace(".nii.gz", f"_label_thr{self.lab_thr}.nii.gz"))
            inpainted = Path(t).parents[1] / "inpaintingv1" / Path(t).name.replace(".nii.gz", "_inpainted_v1.nii.gz")
            train.append({'image': t, 'label': label_path,
                'inpainted': inpainted})
        for t in valFiles:
            label_path = (Path(t).parents[1] / "labels" / Path(t).name.replace(".nii.gz", f"_label_thr{self.lab_thr}.nii.gz"))
            val.append({'image': t, 'label': label_path, 'id': Path(t).name.split(".")[0]})
        for t in testFiles:
            label_path = str(Path(t).parents[1] / "labels" / Path(t).name.replace(".nii.gz", f"_label_thr{self.lab_thr}.nii.gz"))
            test.append({'image': t, 'label': label_path, 'id': Path(t).name.split(".")[0]})


        return train, val, test

    def savePrediction(self, pred: np.ndarray, outputPath: Path,
                       subjects_id: List[str]) -> None:
        # pred.shape = B,C,H,W(,D)

        affine = np.eye(4)
        affine[0,0] = 0.04
        affine[1,1] = 0.004
        affine[2,2] = 0.004

        for i in range(len(subjects_id)):
            outputFilepath = outputPath / f'{subjects_id[i]}.{self.ext}'

            pred_im = np.argmax(pred[i], axis=0).astype("uint8")

            im = nib.Nifti1Image(pred_im, affine)
            im.header.set_xyzt_units("micron")
            nib.save(im, outputFilepath)

