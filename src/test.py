import polars as pl
from pymovements.gaze.gaze_dataframe import GazeDataFrame

df = pl.DataFrame({'t': [0, 1, 2], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]})

# Mauvais : on oublie pixel_columns
gaze = GazeDataFrame(data=df, time_column='t', time_unit='ms')  # --> devrait déclencher un warning

gaze.pix2deg()  # --> devrait déclencher une erreur explicite

