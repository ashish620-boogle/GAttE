import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
