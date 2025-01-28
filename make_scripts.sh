#!/bin/bash

# List of all graphs (yaml param 'graph')
# graph_list=("line_5" "line_10" "line_20" "line 50" "ring_5" "ring_10" "ring_20" "barbell_6" "barbell_10" "barbell_20")
graph_list=()
n=(5 10 20)
p=(0.2 0.5 0.7)
# yaml param 'x_inits'
inits=("zeros" "hot" "cold-hot" "alternating")
# Add 'ER_{n}_{p}' to graph_list for all n and p
for i in "${n[@]}"
do
    for j in "${p[@]}"
    do
        graph_list+=("ER_${i}_${j}")
    done
done

# Print "considering len(graph_list) graphs"
echo "Considering ${#graph_list[@]} Graphs"
echo "Considering ${#inits[@]} Initial Parameters"

# Constant throughout experiments:
# m: 1
# eta: 1
# enforce_symmetry: False
# root_dir: "block_experiments"

# Create a yaml file with all the param combinations
for graph in "${graph_list[@]}"
do 
    for init in "${inits[@]}"
    do
        # Create yaml with the name {graph}_{init}.yaml
        yaml_file="./scripts/${graph}_${init}.yaml"
        echo "Creating ${yaml_file}"
        echo "graph: ${graph}" > $yaml_file
        echo "x_inits: ${init}" >> $yaml_file
        echo "m: 1" >> $yaml_file
        echo "eta: 1" >> $yaml_file
        echo "enforce_symmetry: False" >> $yaml_file
        echo "root_dir: 'block_experiments'" >> $yaml_file
    done
done