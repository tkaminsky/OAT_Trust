#!/bin/bash

H_max=100

# For each file in ./configs
for file in ./configs/*
do
    # Run the experiment with the yaml file
    echo "Running $file"
    # For every h from 1 to H_max
    for h in $(seq 1 $H_max)
    do
        # Run the experiment with the yaml file
        python3 calculate_optimal_attack.py -c $file -m $H_max -H $h
    done
done