#! /bin/sh

EXE='python3 ./experiment_script.py'
minex=3
maxex=4
ndim=2
domtype=None
maxpts=40

param_str="--minex=$minex --maxex=$maxex --maxpts=$maxpts  --domtype=$domtype"
$EXE $param_str