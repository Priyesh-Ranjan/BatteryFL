#!/bin/bash

commands=()

plant_params="--optimizer SGD  --lr 0.01  --momentum 0.0 --AR fednova --dataset plant"
mnist_params="--optimizer Adam --lr 0.001 --momentum 0.5 --AR fednova --dataset mnist"

for thresh in 0.2 0.4 0.6 0.8 ; do
    for iid in 1 0.5 0.1 0.05 ; do
        for selection in ours all NSGA AGE EAFL ; do
                commands+=("python main.py --num_clients 10 ${plant_params} --loader_type dirichlet --experiment_name \"plant_Battery_${selection}_Collection_cost=20_Round_budget=${round_budget}_thresh=${thresh}_iid=${iid}\" --client_selection ${selection} --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio 20 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold ${thresh} --dirichlet_value ${iid} --collection_success_chance 1.0 --sample_selection loss --round_budget 10 --starting_battery 2")
        done
    done
done

printf "%s\n" "${commands[@]}" > experiments3.txt

parallel --bar -a experiments3.txt -j 4 bash -c "{} > /dev/null && echo {} >> done3.txt"
