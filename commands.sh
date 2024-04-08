# Changing collection size

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=100" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 100 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=200" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 200 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=300" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 300 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=400" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 400 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=500" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 500 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=600" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 600 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=700" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 700 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=800" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 800 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=900" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 900 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Collection=1000" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size 1000 --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2


# For the best one we can compare the different sample selection methods

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Method=Loss" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size previous_max --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection loss --round_budget 10 --starting_battery 2

!python main.py --num_clients 10 --optimizer SGD --lr 0.001 --momentum 0.0 --AR fednova --dataset plant --loader_type dirichlet --experiment_name "Battery_Method=TracIn" --client_selection ours --device cuda --batch_size 64 --test_batch_size 512 --collection_size previous_max --collection_battery_ratio 1 --alpha 0.25 --beta 0.25 --gamma 1 --mu 0.5 --training_size 500 --entropy_threshold 0.5 --collection_success_chance 0.95 --sample_selection tracin --round_budget 10 --starting_battery 2


# ...


# ...


# I think after we have done everything then we can compare with the genetic algorithms here
