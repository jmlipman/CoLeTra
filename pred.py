from pydoc import locate
import lib.utils as utils
import time, importlib, torch, pathlib, yaml, copy, os
import lib.const as const
from monai.data import DataLoader, Dataset
from lib.core import evaluation
from pathlib import Path
import numpy as np
from monai.transforms import Compose
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

if __name__ == '__main__':
    t0 = time.time()

    cfg = utils.parseConfig(const.path_pred)
    cfg = utils.evaluateConfig(cfg)

    #cfg['path_datalib'] = Path('lib') / 'data' / f"{cfg['dataset']}.py"
    cfg['path_exp'] = cfg['path_exp'] / str(cfg['exp_run'])
    cfg['path_output_folder'] = cfg["path_exp"] / "preds"
    if not cfg['path_output_folder'].is_dir():
        cfg['path_output_folder'].mkdir()

    with open(const.paths_file, 'r') as f:
        paths = yaml.load(f, Loader=yaml.SafeLoader)
    cfg['path_split'] = paths['data'][cfg['dataset']['name']][utils.getComputerName()]

    path_pred = cfg['path_specific_conf']
    path_pred_started = Path(const.path_predstarted) / path_pred.name
    utils.safeCopy(source=path_pred,
            dest=path_pred_started)
    path_pred.unlink()

    # Loading the model
    model_state = cfg['path_exp'] / 'models' / f"model-{cfg['iterations']}"
    if not model_state.is_file():
        msg = (f"Expected to find the model in `{model_state}` but it was "
                "not found. If the path is different, specifcy "
                "with --model_state `path`")
        raise ValueError(msg)
    utils.log(f"Loading model...{model_state}")
    model = locate(cfg['model']['name'])(**cfg['model']['params']).to(cfg['device'])
    params = torch.load(model_state, map_location=cfg['device'])
    model.load_state_dict(params)

    # Loading the data
    tmp_path = str(cfg['path_datalib'].parents[0]).replace(pathlib.os.sep, '.')
    tmp_path += '.' + cfg['path_datalib'].stem
    dataclass = getattr(importlib.import_module(tmp_path), cfg['path_datalib'].stem)
    dataobj = dataclass(**cfg['dataset']['params'])
    _, _, testFiles = dataobj.split(cfg['path_split'], fold=cfg['fold'])
    if len(testFiles) == 0:
        exit()
    utils.log(f"Test set size: {len(testFiles)}")

    #print("Using batch_size=1 in the test dataloader")
    model.eval()

    test_dataset = Dataset(data=testFiles, transform=cfg['transform_test'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size_val'],
            shuffle=False, num_workers=4)

    if 'params' in cfg['val_inferer']:
        inferer = locate(cfg['val_inferer']['name'])(**cfg['val_inferer']['params'])
    else:
        inferer = locate(cfg['val_inferer']['name'])()

    with torch.no_grad():
        for test_i, test_data in enumerate(test_loader):
            X = test_data['image'].to(cfg['device'])
            subjects_id = test_data['id']

            # Inference
            y_pred = inferer(inputs=X, network=model).to("cpu")
            y_pred = [cfg["transform_post_pred"](i).cpu().detach().numpy()
                        for i in decollate_batch(y_pred)]

            #from IPython import embed; embed(); asd


            dataobj.savePrediction(pred=y_pred,
                    outputPath=cfg['path_output_folder'],
                    subjects_id=subjects_id)

    utils.safeCopy(source=path_pred_started,
            dest=Path(const.path_measure) / path_pred_started.name)
    path_pred_started.unlink()
    msg = f"Total prediction time - {np.round((time.time()-t0)/3600, 3)} hours"
    utils.log(msg, cfg['path_exp'])

    if 'run_measure' in cfg:
        cmd = cfg['run_measure'] + f' --config {Path(const.path_measure) / path_pred_started.name}"'
        os.system(cmd)

