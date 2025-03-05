import torch, monai, time
import lib.utils as utils
from monai.inferers import Inferer
from monai.data import decollate_batch
from typing import List
from types import ModuleType
from torch.nn.modules.loss import _Loss as Loss
from torch.cuda import amp
from pathlib import Path
import numpy as np
from datetime import datetime

from typing import List, Callable, Union, Optional
from types import ModuleType
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

def trainer(model: torch.nn.Module,
            tr_loader: monai.data.dataloader.DataLoader,
            loss: Loss,
            opt: Optimizer,
            scheduler: Union[_LRScheduler, None], # CONFIRM THIS
            iteration_start: int,
            iterations: int,
            val_loader: monai.data.dataloader.DataLoader,
            val_interval: int,
            val_inferer: Inferer,
            metrics: List[monai.metrics.CumulativeIterationMetric],
            dataobj: object,
            postprocessing: monai.transforms.Compose,
            path_exp: Path,
            device: str,
            callbacks: List[Callable]=[]) -> None:

    utils.log("Start training", path_exp)
    t0 = time.time()
    it = iteration_start
    scaler = amp.GradScaler()

    # As some servers only allow you to run jobs for max. 3 days, it
    # can be important to resume the training and re-execute certain
    # procedures, like scheduler.step()
    utils.callCallbacks(callbacks, '_start_training', locals())

    tr_loss = 0 # exponential moving average (alpha=0.99)
    model.train()
    while it <= iterations:

        utils.callCallbacks(callbacks, '_start_iteration', locals())
        for tr_i, batch_data in enumerate(tr_loader):
            # In the worst case, I can do an inner for loop here
            #X, Y, Y2 = (
            X, Y = (
                    batch_data['image'].to(device),
                    batch_data['label'].to(device),
                    #batch_data['label2'].to(device)
            )
            #from IPython import embed; embed(); asd
            utils.callCallbacks(callbacks, '_start_train_iteration', locals())

            #t0 = time.time()
            with amp.autocast():
                y_pred = model(X)

                if "TopoNet" in str(loss.__class__) or "Warping" in str(loss.__class__):
                    if it >= iterations*0.7:
                        loss.activate = True

                if model.__class__ == monai.networks.nets.attentionunet.AttentionUnet:
                    tr_loss_tmp = loss(y_pred, Y)
                else:
                    # Assumption: DynUNet and Deep supervision.
                    # So, output = torch.Size([B, DS, C, H, W, D])
                    tr_loss_tmp = loss(y_pred[:, 0], Y)
                    for i in range(1, y_pred.shape[1]):
                        tr_loss_tmp += loss(y_pred[:, i], Y)
                    tr_loss_tmp /= y_pred.shape[1]

            tr_loss = 0.99*tr_loss + 0.01*tr_loss_tmp.cpu().detach().numpy()

            # Optimization
            opt.zero_grad()
            scaler.scale(tr_loss_tmp).backward()
            utils.callCallbacks(callbacks, '_after_compute_grads', locals())
            scaler.step(opt)
            scaler.update()

            utils.callCallbacks(callbacks, '_end_train_iteration', locals())

            if it % val_interval == 0:
                val_str = ""
                if len(val_loader) > 0:
                    utils.log("Validation", path_exp)
                    model.eval()

                    val_str = evaluation(model=model,
                                         data_loader=val_loader,
                                         iteration=it,
                                         batch_size=tr_loader.bs,
                                         inferer=val_inferer,
                                         metrics=metrics,
                                         dataobj=dataobj,
                                         path_exp=path_exp,
                                         device=device,
                                         postprocessing=postprocessing,
                                         loss=None,
                                         callbacks=callbacks,
                                         save_preds=True)
                    model.train()

                eta = time.time() + (iterations-it)*(time.time()-t0)/it
                eta = datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')
                msg = f"Iteration: {it}. Loss: {tr_loss}. {val_str} ETA: {eta}"
                utils.log(msg, path_exp)

            if scheduler:
                scheduler.step()
            it += 1

            if it > iterations:
                break


    utils.callCallbacks(callbacks, "_end_training", locals())

def evaluation(model: torch.nn.Module,
               data_loader: monai.data.dataloader.DataLoader,
               iteration: int,
               batch_size: int,
               inferer: Inferer,
               metrics: List[monai.metrics.CumulativeIterationMetric],
               dataobj: ModuleType,
               path_exp: Path, # Folder
               device: str,
               postprocessing: monai.transforms.compose.Compose,
               loss: Loss=None,
               callbacks: List[str]=[],
               save_preds: bool=False) -> str:
    """
    Performs the prediction, computes the metrics, optionally saves the preds.
    """

    val_loss, all_subjects_id = 0, []
    with torch.no_grad():
        for val_i, val_data in enumerate(data_loader):
            utils.callCallbacks(callbacks, "_start_val_subject", locals())
            print(f"Predicting image {val_i+1}/{len(data_loader)}")

            X = val_data["image"].to(device) # cuda
            if loss or len(metrics) > 0:
                Y = val_data["label"] # cpu
            subjects_id = [val_data["id"][i] for i in range(len(val_data['id']))]
            all_subjects_id.extend(subjects_id)

            # Inference
            y_pred = inferer(inputs=X, network=model).to("cpu")

            # Note: I moved this up here because the "postprocessing"
            # done below will inevitably argmax my preds, which is necessary
            # for the metrics.
            if save_preds:

                path_preds = path_exp / 'preds' / str(iteration)
                path_preds.mkdir(parents=True, exist_ok=True)
                print("Save prediction in: ", path_preds)
                dataobj.savePrediction(pred=y_pred.cpu().detach().numpy(),
                                       outputPath=path_preds,
                                       subjects_id=subjects_id)


            if loss:
                val_loss += loss(y_pred, Y) / len(data_loader)
            y_pred = [postprocessing(i) for i in decollate_batch(y_pred)]

            for metric in metrics:
                metric(y_pred=y_pred, y=Y) # B,H,W(,D)

            utils.callCallbacks(callbacks, "_end_val_subject", locals())

    val_str = ""
    if loss:
        val_str = f"Val loss: {val_loss}. "

    if len(metrics) == 0:
        return val_str

    # Save metrics
    val_str += utils.saveMetrics(metrics, all_subjects_id, dataobj,
            path_exp / 'val_scores' / f'scores-{iteration}.csv')

    return val_str

