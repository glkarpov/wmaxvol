#! /bin/sh

EXE='python3 ./experiment_oop_based.py'
minex=3
maxex=5
ndim=4
domtype=None
npts=500

param_str="--minex=$minex --maxex=$maxex --ndim=$ndim --domtype=$domtype --npts=$npts"
$EXE $param_str
