class GazeData:
    def __init__(self, column_counts = [0, 0, 0, 0]):
        self.column_counts = list(column_counts)
    
    def __str__(self):
        return f'columns: {self.column_counts}'
    
    def __repr__(self):
        return f'GazeData(column_counts = {repr(self.column_counts)})'