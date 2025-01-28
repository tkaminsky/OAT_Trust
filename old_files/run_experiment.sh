#!/bin/bash

for i in {1..20}
do
    python3 calculate_optimal_attack.py -H $i -c scripts/barbell_6_cold-hot.yaml --max_H 20
done