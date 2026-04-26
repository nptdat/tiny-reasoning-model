import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # already call torch.cuda.manual_seed_all and torch.mps.manual_seed
    # https://github.com/pytorch/pytorch/blob/bdd83c4c7f1ede5f528ffc49ee8eb348f20a26bd/torch/random.py#L32-L60
    torch.manual_seed(seed)

    # it's safe to call this even cuda is not available
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
