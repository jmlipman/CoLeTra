import torch, os
import nibabel as nib
import numpy as np
import pathlib
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
from monai.data.meta_tensor import MetaTensor
from PIL import Image

def _end_train_iteration_save_last_model_before_val(
        model: torch.nn.Module,
        path_exp: Path,
        it: int,
        val_interval: int) -> None:
    """
    Saves the Pytorch model after each iteration.
    If the model in the previous iteration was saved, delete it.

    Args:
      `self`: model.
      `outputPath`: Path to the output (e.g., exp_name/21
      `it`: Current iteration.
    """

    if it % val_interval == 0:

        curr_model_path = path_exp / 'models' / f'model-{it}'
        torch.save(model.state_dict(), curr_model_path)

        prev_model_path = path_exp / 'models' / f'model-{it-val_interval}'
        if prev_model_path.exists():
            pathlib.os.remove(prev_model_path)

def _end_train_iteration_save_last_model(
        model: torch.nn.Module,
        path_exp: Path,
        it: int) -> None:
    """
    Saves the Pytorch model after each iteration.
    If the model in the previous iteration was saved, delete it.

    Args:
      `self`: model.
      `outputPath`: Path to the output (e.g., exp_name/21
      `it`: Current iteration.
    """

    curr_model_path = path_exp / 'models' / f'model-{it}'
    prev_model_path = path_exp / 'models' / f'model-{it-1}'

    torch.save(model.state_dict(), curr_model_path)
    if prev_model_path.exists():
        pathlib.os.remove(prev_model_path)


def _start_training_scheduler_init(
        scheduler: Union[_LRScheduler, None],
        iteration_start: int):
    """
    Decreases the learning rate by using the scheduler before training starts.
    This is useful when running on servers that allow you to run jobs for a
    limited time. Thus, when continue running those jobs, learning rate needs
    to be decreased before training.

    Args:
      `scheduler`: Custom name of the lr scheduling strategy.
      `iteration_start`: Number of times scheduler.step() will be executed.

    """
    if not scheduler is None and iteration_start > 1:
        for it in range(1, iteration_start):
            scheduler.step()
