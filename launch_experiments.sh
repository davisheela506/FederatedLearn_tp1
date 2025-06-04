#!/bin/bash

alphas=(0.1 1.0 10.0)
rounds=10
clients=10
batch_size=64
epochs=3

for alpha in "${alphas[@]}"
do
    echo "Running simulation for alpha=$alpha"
    python run_simulation.py --clients $clients --alpha $alpha --rounds $rounds --batch_size $batch_size --epochs $epochs
done
