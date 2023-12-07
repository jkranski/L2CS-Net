import numpy as np
from dataclasses import dataclass

@dataclass
class PerspectiveTransform():
    matrix: np.ndarray
    width: float
    height: float


@dataclass
class Point2D():
    x: int
    y: int

@dataclass
class BoundingBox():
    center: Point2D
    upper_left: Point2D
    lower_right: Point2D
    width: int
    height: int
