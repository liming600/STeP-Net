import yaml
from easydict import EasyDict
import os
from pathlib import Path
import shutil
from .logger import print_log

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config, base_dir=None):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                base_path = Path(new_config['_base_'])
                if not base_path.is_absolute() and base_dir is not None:
                    relative_path = Path(base_dir) / base_path
                    # Existing configurations use repository-root-relative paths.
                    base_path = relative_path if relative_path.exists() else base_path
                with open(base_path, 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val, base_dir=base_path.parent)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, base_dir=base_dir)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    cfg_path = Path(cfg_file).resolve()
    with open(cfg_path, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config, base_dir=cfg_path.parent)
    return config

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    shutil.copy2(args.config, config_path)
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )
