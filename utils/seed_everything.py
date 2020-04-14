import tensorflow as tf
import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """
    seed値を固定
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seed_everything_dl(seed=1234):
    """
    tensorflowやpytorchをつかうならこっち
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
