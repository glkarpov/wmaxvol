#! /bin/sh

EXE='python ./experiment_script.py'
minex=20
maxex=100
maxpts=200


for domtype in circle blob
do
    for rad in 0.005 0.01
    do
        param_str="--minex=$minex --maxex=$maxex --maxpts=$maxpts --domtype=$domtype --cutrad=$rad"
        echo $EXE $param_str
    done
done
