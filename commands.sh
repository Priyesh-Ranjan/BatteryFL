for selection in all ours NSGA AGE EAFL ; do
# Cost of collecting samples
    for round_budget in 5 10 ; do
        for cost in 1000 100 50 20 10 ; do
            python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_${selection}_Collection_cost=${cost}_Round_budget=${round_budget}" --client_selection ${selection} --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio ${cost} --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget ${round_budget} --starting_battery 2
        done
    done


    # Maybe we can try doing the different collection success chance from 0.5 to 1
    for cost in 20 50 ; do
        for chance in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 ; do
            python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_${selection}_Collection_cost=${cost}_chance=${chance}_Round_budget=${round_budget}" --client_selection ${selection} --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio ${cost} --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance ${chance} --sample_selection loss --round_budget ${round_budget} --starting_battery 2
        done
    done
done