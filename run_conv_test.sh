#! /bin/sh

EXE='python3 ./convergence_test.py'
global_iters=2
wmxvl_iters=160
maxex=3
ndim=1
npts=70

param_str="--global_iters=$global_iters --wmxvl_iters=$wmxvl_iters --ndim=$ndim --npts=$npts --maxex=$maxex "
$EXE $param_str
