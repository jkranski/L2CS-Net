from typing import List
from pydantic import BaseModel


class Point2DF(BaseModel):
    x: float
    y: float


class Point3DF(BaseModel):
    x: float
    y: float
    z: float


class Face(BaseModel):
    camera_centroid_norm: Point2DF
    gaze_vector: Point3DF
    gaze_screen_intersection_norm: Point2DF
    face_patch_jpeg_base64: str | None = None

class GazeData(BaseModel):
    faces: List[Face] = []
