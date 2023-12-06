import numpy as np
from dataclasses import dataclass

@dataclass
class PerspectiveTransform():
    matrix: np.ndarray
    width: float
    height: float
