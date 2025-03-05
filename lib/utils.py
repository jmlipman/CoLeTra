from typing import List, Tuple, Union, Dict, Callable
from pydoc import locate
from datetime import datetime
from yaml.loader import SafeLoader
import argparse, os, yaml, re, inspect, monai, shutil, itertools
from collections import abc
from monai.transforms import Compose, Randomizable
import lib.const as const
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from types import ModuleType
import numpy as np
import pandas as pd
from monai.data.meta_tensor import MetaTensor
import torch

def parseConfig(path_exps: str) -> Dict:
    """
    Reads the configuration files. First the paths, then the general
    configuration file (if it exists), and then the specific configuration
    file (if it exists).

    Args:
      path_exps: Path where the experiments are located, typically 'exp1_train'

    Returns:
      Configuration.
    """
    def parseArguments() -> Dict:
        """Parses and sanitizes the user's given input.
        """
        parser = argparse.ArgumentParser(description="New FEMAI")

        parser.add_argument("--config",
                            help="Location of the exp. conf. file",
                            required=False)
        parser.add_argument("--config_general",
                            help="Location of the general exp. conf. file",
                            required=False)
        parser.add_argument("--model_state",
                            help="Location of the saved model",
                            required=False)
        parser.add_argument("--path_output_folder",
                            help="Folder where the output patches will be saved",
                            required=False)
        parser.add_argument("--run_measure",
                            help="Submit a separate job for computing metrics",
                            required=False,
                            choices=["0", "1"],
                            )

        args = parser.parse_args()
        if args.config and not Path(args.config).is_file():
            msg = f"Configuration file `{args.config}` not found."
            raise ValueError(msg)
        if args.config_general and not Path(args.config_general).is_file():
            msg = f"Configuration file `{args.config_general}` not found."
            raise ValueError(msg)
        if args.model_state and not Path(args.model_state).is_file():
            msg = f("The specified --model_state `{args.model_state}` "
                    "does not exist.")
            raise ValueError(msg)
        return args

    def getNextSpecificConfigFile(path: str) -> str:
        """Returns the next configuration file that will be used.
        This makes sense when running on a local computer and not in a cluster.
        Files have the following format:
          XXX-name_experiment.yaml, where XXX is optional and indicates priority
          so that the smaller, the earliest it will be "served".
        If this function finds those "priority files", it will select the highest
        priority; otherwise, it will just take one of them.
        """
        if not Path(path).exists():
            msg = f"Specific config folder `{path}` does not exists."
            raise ValueError(msg)
        files = os.listdir(path)
        if len(files) == 0:
            msg = f"Folder `{path}` is empty."
            raise ValueError(msg)

        r = re.compile("^[0-9]+-")
        priority_files = list(filter(r.match, files))
        if len(priority_files) > 0:
            prioritized = [(int(f.split("-")[0]), f) for f in priority_files]
            prioritized = sorted(prioritized, key=lambda x: x[0])
            file = prioritized[0][1]
        else:
            file = files[0]

        return str(Path(path) / file)

    def update_or_add_config(cfg, specific_conf):

        append_config = ["transform_train", "transform_val",
                "transform_test", "transform_post", "transform_measure"]

        do_not_update = ['path_base', 'path_general_conf',
                'path_split', 'path_specific_conf', 'path_exp',
                'computer_name']

        if specific_conf.get('full_conf', False):
            # Delete certain things that will be regenerated
            for c in do_not_update:
                del specific_conf[c]
        else:
            for c in append_config:
                if c in cfg and c in specific_conf:
                    cfg[c].extend(specific_conf[c])
                    del specific_conf[c]
        cfg.update(specific_conf)

        return cfg

    # 1) Read paths
    cfg = {'computer_name': getComputerName()}
    with open(const.paths_file, 'r') as f:
        paths = yaml.load(f, Loader=SafeLoader)
    cfg['path_base'] = paths['output'][cfg['computer_name']]

    params = parseArguments()
    # 2) Read general configuration (if it exists)
    if path_exps == const.path_train:
        config_general = None
        if params.config_general:
            config_general = params.config_general
        elif Path(const.general_conf_file).is_file():
            config_general = const.general_conf_file
        if config_general:
            with open(config_general, 'r') as f:
                cfg_tmp = yaml.load(f, Loader=SafeLoader)
            cfg.update(cfg_tmp)
            cfg['path_general_conf'] = config_general

    # 3) Read specific configuration
    config_specific = None
    if params.config:
        config_specific = params.config
    else:
        config_specific = getNextSpecificConfigFile(path_exps)

    with open(config_specific, 'r') as f:
        cfg_tmp = yaml.load(f, Loader=SafeLoader)
    #from IPython import embed; embed(); asd
    cfg = update_or_add_config(cfg, cfg_tmp)
    cfg['path_specific_conf'] = config_specific

    #if not 'path_exp' in cfg:
    cfg['path_exp'] = str(Path(cfg['path_base']) / cfg['exp_name'])
    cfg['path_split'] = paths['data'][cfg['dataset']['name']][cfg['computer_name']]

    # 4) Sanitize, verification
    if not 'exp_name' in cfg:
        msg = "There should be an 'exp_name' in the config files."
        raise ValueError(msg)

    if not Path(f"{cfg['dataset']['name']}".replace(".", "/")+".py").is_file():
        msg = f"Expected lib file `{cfg['dataset']['name']}` does not exist."
        raise ValueError(msg)

    if params.model_state:
        cfg['model_state'] = params.model_state
    elif 'model_state' in cfg and not Path(cfg['model_state']).is_file():
        msg = (f"The model_state `{cfg['model_state']}` specified "
                "in the configuration files does not exist.")
        raise ValueError(msg)

    if not 'params' in cfg['loss'] or ('params' in cfg['loss'] and cfg['loss']['params'] is None):
        cfg['loss']['params'] = {}

    if not 'iteration_start' in cfg:
        cfg['iteration_start'] = 1
    if not 'batch_size_val' in cfg:
        cfg['batch_size_val'] = 1

    if not 'move_config_file' in cfg:
        cfg['move_config_file'] = True

    # For the new pred.py
    if 'path_output_folder' in params and not params.path_output_folder is None:
        cfg['path_output_folder'] = params.path_output_folder
    cfg['full_conf'] = True

    if 'run_measure' in params and params.run_measure == "1":
        if cfg['dataset']['name'] == 'lib.data.DRIVE':
            time, mem = "1:00", "1GB"
        elif cfg['dataset']['name'] == 'lib.data.CREMI':
            time, mem = "2:00", "20GB"
        elif cfg['dataset']['name'] == 'lib.data.Crack500':
            time, mem = "2:00", "8GB"
        elif cfg['dataset']['name'] == 'lib.data.NarwhalNoUOMs':
            time, mem = "5:00", "20GB"

        if cfg['computer_name'] == "cluster":
            cfg['run_measure'] = f'bsub -q compute -J measure -n 1 -W {time} -R "rusage[mem={mem}]" -o tmp_res/%J.out -e tmp_res/%J.err "module load python3/3.9.11 cuda/11.7 julia/1.9.2; source /PATH/bin/activate; python -u measure.py'
        else:
            msg = f"Trying to run a separate job to compute the metrics but this is an unknown server/computer `{cfg['computer_name']}`"
            raise ValueError(msg)



    # Check that the following configuration is in 'cfg' dictionary
    info = ['path_split', 'path_base', 'path_exp',
            'transform_train', 'transform_val', 'transform_test',
            'transform_post_val', 'transform_post_pred',
            'optimizer', 'random_seed', 'callbacks', 'iterations',
            'val_interval', 'metrics_val', 'metrics_pred',
            'computer_name', 'fold', 'move_config_file']
    required_not_given_config = set(info).difference(set(cfg.keys()))
    if len(required_not_given_config):
        msg = ("The following information is required and was not given to "
                f"the cfg dictionary: {required_not_given_config}")
        raise ValueError(msg)

    return cfg


def evaluateTransforms(cfg: dict, splits: List[str]):
    monai.utils.misc.set_determinism(seed=cfg['random_seed'])
    for data_split in splits:
        l = []
        for trans in cfg[f'transform_{data_split}']:
            t = locate(trans['name'])(**trans['params'])
            if isinstance(t, Randomizable):
                t.set_random_state(cfg['random_seed'])
            l.append(t)
        cfg[f'transform_{data_split}'] = Compose(l)
    return cfg

def evaluateConfig(cfg):
    """Convert the str corresponding to functions to actual functions.
    """
    def _evaluateLambdas(cfg: dict) -> dict:
        for k, v in cfg.copy().items():
            if isinstance(v, str) and v.startswith('lambda '):
                v = eval(v)
            cfg.pop(k)
            cfg[k] = v
            if isinstance(v, dict):
                _evaluateLambdas(v)
        return cfg

    def _evaluateMetrics(cfg: dict):
        for mm in ['metrics_val', 'metrics_pred']:
            metrics = []
            for m in cfg[mm]:
                if 'params' in m:
                    metrics.append( locate(m['name'])(**m['params']) )
                else:
                    metrics.append( locate(m['name'])() )
            cfg[mm] = metrics
        return cfg

    def _evaluatePaths(cfg: dict):
        for c in cfg.keys():
            if c.startswith('path'):
                cfg[c] = Path(cfg[c])
        #cfg['path_datalib'] = Path('lib') / 'data' / f"{cfg['dataset']}.py"
        cfg['path_datalib'] = Path(f"{cfg['dataset']['name'].replace('.', '/')}.py")
        if 'model_state' in cfg:
            cfg['model_state'] = Path(cfg['model_state'])
        return cfg

    cfg = _evaluateLambdas(cfg)
    splits = ['train', 'val', 'test', 'post_val', 'post_pred', 'measure']
    cfg = evaluateTransforms(cfg, splits)
    cfg = _evaluateMetrics(cfg)
    cfg = _evaluatePaths(cfg)
    if 'params' in cfg['loss'] and 'weight' in cfg['loss']['params']:
        cfg['loss']['params']['weight'] = torch.tensor(cfg['loss']['params']['weight'])
    return cfg

def log(text: str, path: str=None) -> None:

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text = date + ': ' + str(text)
    print(text)
    if path:
        with open(Path(path) / 'log.txt', 'a') as f:
            f.write(f'{text}\n')



def callCallbacks(callbacks: List[Callable], prefix: str,
        allvars: dict) -> None:
    """
    Call all callback functions starting with a given prefix.
    Check which inputs the callbacks need, and, from `allvars` (that contains
    locals()) pass those inputs.
    Read more about callback functions in lib.callback

    Args:
      `callbacks`: List of callback functions.
      `prefix`: Prefix of the functions to be called.
      `allvars`: locals() containing all variables in memory.
    """
    for c in callbacks:
        lib, cname = c.rsplit(".", 1)
        c = getattr(__import__(lib, fromlist=[cname]), cname)
        if cname.startswith(prefix):
            input_params = inspect.getfullargspec(c).args
            required_params = {k: allvars[k] for k in input_params}
            c(**required_params)

def safeCopy(source: Path, dest: Path) -> Path:
    """
    Copy the file `source` to `dest`. If the file already exists, rename it.
    """

    destFolder = dest.parents[0]
    if dest.exists():
        r = re.compile(dest.stem + "(_[0-9]+)")
        fi = list(filter(r.match, [x.name for x in destFolder.glob('*')]))
        c = len(fi)+1
        #from IPython import embed; embed(); asd
        destFile = destFolder / f'{dest.stem}_{c}{dest.suffix}'
    else:
        destFile = dest

    shutil.copy(source, destFile)
    return destFile

def saveMetrics(metrics: List[monai.metrics.CumulativeIterationMetric],
                subjects: List[str],
                dataobj: object,
                path_output: Path) -> str:
    # NOTE: This is prepared for one-class problems. For two-class problems
    # we probably need to edit the "aggregate" function in each metric class
    # in lib/metric.py
    all_metrics, metric_names = [], []
    val_str = ""
    for m in metrics:
        tmp_metric = m.aggregate().cpu().detach().numpy()
        if len(tmp_metric.shape) == 1:
            tmp_metric = np.expand_dims(tmp_metric, 1)
        all_metrics.append(tmp_metric)
        metric_names.append( m.__class__.__name__.replace("Metric", "") )
        if metric_names[-1] == "BettiError":
            metric_names[-1] = "Betti0Error"
            metric_names.append("Betti1Error")
            if dataobj.dim == 3:
                metric_names.append("Betti2Error")
        m.reset()
        val_str += f"Val {metric_names[-1]}: {all_metrics[-1].mean(axis=0)}. "

    #from IPython import embed; embed(); asd
    subjects = np.array([subjects]).T
    res = np.concatenate([subjects] + all_metrics, axis=1)
    combination = list(itertools.product(metric_names, dataobj.classes))
    cols = ["ID"] + ["_".join(t) for t in combination]
    df = pd.DataFrame(res, columns=cols)
    # Sort columns
    sorted_cols_names = sorted(cols)
    sorted_cols_names.remove("ID")
    sorted_cols_names = ["ID"] + sorted_cols_names
    df = df[sorted_cols_names]
    df.to_csv(path_output, index=False)
    print(f"Results saved in: {path_output}")
    return val_str

class DataLoaderWrapper:
    """
    For some reason, MONAI doesn't enforce the batch size when using
    num_samples. Read more here:
    https://github.com/Project-MONAI/tutorials/discussions/1244
    So, I created this wrapper to ensure that the batch size is correct.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.iter = self.dataset.__iter__()
        self.batch = None
        self.c = 0 # cursor
        self.bs = self.dataset.batch_size

    def __iter__(self):
        if not self.batch:
            try:
                self.batch = next(self.iter)
            except:
                self.iter = self.dataset.__iter__()
                self.batch = next(self.iter)
        d = {}
        for k in self.batch:
            if isinstance(self.batch[k], dict):
                d[k] = {}
                for t in self.batch[k].keys():
                    d[k][t] = self.batch[k][t][self.c:self.c+self.bs]
            elif isinstance(self.batch[k], MetaTensor): # Tensor
                d[k] = self.batch[k][self.c:self.c+self.bs]
                self.max = self.batch[k].shape[0]
            else:
                raise Exception(f"Unknown value `{type(self.batch[k])}`")

        self.c += self.bs
        if self.c >= self.max:
            self.c = 0
            self.batch = None
        yield d

def getComputerName() -> str:
    """Returns the computer's name.
    """
    pc_name = os.uname()[1]
    return pc_name

