#!/bin/bash

if [ ! -f "experiments.txt" ]; then
    commands=()

    for selection in ours all NSGA AGE EAFL ; do
        for round_budget in 10 ; do 
            for cost in 20 50 ; do 
                for chance in 0.5 0.6 0.7 0.8 0.9 1 ; do
                    commands+=("python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name \"Battery_${selection}_Collection_cost=${cost}_chance=${chance}_Round_budget=${round_budget}\" --client_selection ${selection} --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio ${cost} --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance ${chance} --sample_selection loss --round_budget ${round_budget} --starting_battery 2")
                done
            done

            for cost in 1000 100 10 ; do
                commands+=("python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name \"Battery_${selection}_Collection_cost=${cost}_chance=1_Round_budget=${round_budget}\" --client_selection ${selection} --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio ${cost} --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 1 --sample_selection loss --round_budget ${round_budget} --starting_battery 2")
            done
        done
    done

    printf "%s\n" "${commands[@]}" > experiments.txt
fi

parallel -a experiments.txt -j 5 bash -c "{}"
