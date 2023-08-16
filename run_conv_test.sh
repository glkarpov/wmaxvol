#! /bin/sh

EXE='python3 ./convergence_test.py'
global_iters=10
wmxvl_iters=2000
maxex=5
ndim=3
npts=200
adder=''
param_str="--global_iters=$global_iters --wmxvl_iters=$wmxvl_iters --ndim=$ndim --npts=$npts --maxex=$maxex --add_name=$adder"
$EXE $param_str
