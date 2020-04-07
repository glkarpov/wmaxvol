#! /bin/sh

EXE='python3 ./process_oop.py'
ndim=2

for ndim in 2 4 7 
do 
    param_str="--ndim=$ndim"
    $EXE $param_str
done
