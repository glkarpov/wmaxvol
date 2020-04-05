#! /bin/sh

EXE='python3 ./nmb_error.py'
ndim=2
experiment='designs_dim='$ndim'_aaa'
func=f_sincos
param_str="--ndim=$ndim --experiment=$experiment --func=$func"
$EXE $param_str