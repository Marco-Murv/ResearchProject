#!/usr/bin/env bash


options=(
    dragonnet
    tarnet
)

#PATH=c/Users/Marco/anaconda3

for i in ${options[@]}; do
    echo $i
    py -m experiment.ihdp_main --data_base_dir C:/Users/Marco/Documents/ResearchProject/dragonnet/dat/ihdp/csv/
                                 --knob $i\
                                 --output_base_dir C:/Users/Marco/Documents/ResearchProject/dragonnet/results/ihdp


done

