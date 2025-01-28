#!/bin/bash

H_max=50
file="finished_scripts/ER_5_0.2_alternating.yaml"

# Start time
start_time=$(($(date +%s)))

# For each file in ./scripts
echo "Running $file"
# For every h from 1 to H_max
for h in $(seq 1 $H_max)
do
    # Run the experiment with the yaml file
    python3 calculate_optimal_attack.py -c $file -m $H_max -H $h
done

# End time
end_time=$(($(date +%s)))

# Calculate and echo the runtime
runtime=$((end_time - start_time))
echo "Total Experiment Runtime: $runtime seconds"