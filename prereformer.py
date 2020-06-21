import os
import pandas as pd
import pathlib
import numpy as np
from maxvolpy.maxvol import rect_maxvol
os.environ['OMP_NUM_THREADS'] = '6'
print (os.environ['OMP_NUM_THREADS'])

cur_pos = str(pathlib.Path().parent.absolute())
data = pd.read_csv(cur_pos + "/Fraunhofer/prereformer_data/Prereformer_training_matrix.csv", sep=',', header=None)
num_extracted_points = 50000
exp_name = "prereformer_pts_7"

appropriate_data = data.values[1:, :]
A = np.array(appropriate_data, dtype='float64')
print(A.shape, type(A))

pivs,_ = rect_maxvol(A, maxK = num_extracted_points, minK = num_extracted_points)
np.savez(os.path.join(cur_pos, "Fraunhofer/prereformer_data/" + exp_name), pts=pivs)
