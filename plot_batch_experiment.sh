#!/bin/bash
# Run python plot_attacks.py -f {file} for every file in ./block_experiments
for file in ./extra_interesting_experiments/*
do
    echo "Plotting $file"
    python3 plot_attacks.py -f $file
done