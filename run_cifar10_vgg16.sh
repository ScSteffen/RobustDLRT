#!/bin/bash

# Define a list of tol values to iterate over
stat_runs="1 2 3"
tols="0.01"  # Default tol value
integrator="2"  # Default lr_integrator_choice

# Constants based on the argparse default values
local_iterations=20  # Default num_local_iter
epochs=200  # Number of aggregation rounds (used as epochs)

for run in $stat_runs; do
    for clients_i in $clients; do
        for integrator_i in $integrator; do

            # Run the training script with adjusted parameters
            python main_cifar10.py --benchmark 1 \
                                   --model 1 \
                                   --batch_size 128 \
                                   --lr_integrator_choice $integrator_i \
                                   --initial_cr 0.8 \
                                   --max_rank 300 \
                                   --tol 0.1 \
                                   --num_local_iter $n_loc_iterations \
                                   --learning_rate 5e-3 \
                                   --epochs $epochs \
                                   --momentum 0.3 \
                                   --weight_decay 1e-2 \
                                   --output "results/Cifar10/VGG16/run_$run.csv" \
                                   --wandb 0 \
                                   --load_checkpoint 0
        done
    done
done
