import os
import pandas as pd
import pathlib
import numpy as np
from maxvolpy.maxvol import rect_maxvol


# os.environ['OMP_NUM_THREADS'] = '6'
# print(os.environ['OMP_NUM_THREADS'])


def prereformer_single_shot(A, num_extracted_points, exp_name):
    print(A.shape, type(A))
    pivs, _ = rect_maxvol(A, maxK=num_extracted_points, minK=num_extracted_points)
    np.savez(os.path.join(cur_pos, "Fraunhofer/prereformer_data/" + exp_name), pts=pivs)


def extract_pivots(piv, p):
    upd_pivs = np.delete(p, list(piv))
    return upd_pivs


def prereformer_k_run(filepath: str, data_name: str, ext_list, exp_name=None):
    data = pd.read_csv(filepath + data_name, sep=',', header=None)
    appropriate_data = data.values[1:, :]
    A = np.array(appropriate_data, dtype='float32')
    if exp_name is not None:
        taken_points = np.load(filepath + exp_name + ".npz")
        full_piv = taken_points['pts']
    else:
        full_piv = None

    cmplt_piv = maxvol_full_k_run(ext_list, A, start_piv=full_piv)
    return cmplt_piv


def maxvol_full_k_run(extraction_list: list, A, start_piv=None):
    if start_piv is not None:
        full_piv = start_piv
        upd = extract_pivots(full_piv, np.arange(A.shape[0]))
        A = A[upd]

    for i, num_pts_extract in enumerate(extraction_list):
        relative_pivs, _ = rect_maxvol(A, maxK=num_pts_extract, minK=num_pts_extract)
        print(relative_pivs, 'relative')
        if i == 0 and start_piv is None:
            full_piv = np.copy(relative_pivs)
            upd = extract_pivots(relative_pivs, np.arange(A.shape[0]))
        else:
            full_piv = np.concatenate((full_piv, upd[relative_pivs]), axis=0)
            upd = extract_pivots(relative_pivs, upd)
        kek = extract_pivots(relative_pivs, np.arange(A.shape[0]))
        A = A[kek]
    return full_piv


if __name__ == '__main__':
    ext_list = [19000]
    exp_name = "prereformer"
    final_name = "prereformer_full"
    cur_pos = str(pathlib.Path().parent.absolute())
    filepath = cur_pos + "/Fraunhofer/prereformer_data/"
    data_name = "Prereformer_training_matrix.csv"
    result = prereformer_k_run(filepath, data_name, ext_list, exp_name)
    np.savez(filepath + final_name, pts=result)
