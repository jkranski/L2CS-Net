from typing import List
from pydantic import BaseModel


class Point2DF(BaseModel):
    x: float
    y: float


class Face(BaseModel):
    camera_centroid_norm: Point2DF
    gaze_vector: Point2DF
    gaze_screen_intersection_norm: Point2DF


class GazeData(BaseModel):
    faces: List[Face] = []
