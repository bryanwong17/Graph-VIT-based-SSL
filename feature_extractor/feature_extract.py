from simclr import SimCLR
from mocov3 import MocoV3
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse
import torch
import random
import os
import numpy as np

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("train")
    torch.cuda.empty_cache()
    # for reproductibility
    seed_everything(1001)
    
    # os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('--magnification', type=str, default='20x')
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # return train_loader and valid_loader
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR(dataset, config)
    simclr.train()
    # mocov3 = MocoV3(dataset, config)
    # mocov3.train()


if __name__ == "__main__":
    main()
