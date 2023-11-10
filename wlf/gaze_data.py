from typing import List
from pydantic import BaseModel


class GazeData(BaseModel):
    column_counts: List[int] = [0, 0, 0, 0]
