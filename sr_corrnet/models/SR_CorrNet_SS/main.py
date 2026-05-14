import os
import torch
from loguru import logger
from .dataset import get_dataloaders, get_multi_dataloaders
from .model import Model
from .engine import Engine
from sr_corrnet.utils import util_system
from sr_corrnet.utils.decorators import logger_wraps


# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):

    ''' Build Setting '''
    # Call configuration file (configs.yaml)
    config_name = getattr(args, 'config', 'baseline.yaml')  # default to baseline.yaml if not specified
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_name)
    yaml_dict = util_system.parse_yaml(yaml_path)
    logger.info(f"Using config file: {config_name}")

    # Run wandb and get configuration
    config = yaml_dict["config"]

    partitions = config["partitions"][args.engine_mode]
    logger.info(f"mode partition : {partitions}")

    ''' Build Dataloader '''
    if config["is_var_spks"]:
        dataloaders = get_multi_dataloaders(partitions, config)
    else:
        dataloaders = get_dataloaders(partitions, config)

    ''' Build Model '''
    # Call network model
    model = Model(**config["model"])

    ''' Build Engine '''
    # Call gpu id & device
    gpuid = tuple(map(int, args.gpuid.split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')

    # Call & Run Engine
    engine = Engine(args, config, model, dataloaders, gpuid, device)
    if args.engine_mode == "train":
        engine.run()
    else:  # test
        engine.run_eval()
        
        