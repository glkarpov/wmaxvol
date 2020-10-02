#! /bin/sh

EXE='python3 ./convergence_test.py'
global_iters=2
wmxvl_iters=100
maxex=3
ndim=2
npts=70
adder='testexp'
param_str="--global_iters=$global_iters --wmxvl_iters=$wmxvl_iters --ndim=$ndim --npts=$npts --maxex=$maxex --add_name=$adder"
$EXE $param_str
