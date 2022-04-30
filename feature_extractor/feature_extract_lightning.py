# from simclr import SimCLR
from mocov3_lightning import MocoV3
import yaml
import os
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
    torch.cuda.empty_cache()
    # for reproductibility
    seed_everything(1001)

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    mocov3 = MocoV3(config)
    mocov3.train()


if __name__ == "__main__":
    main()
