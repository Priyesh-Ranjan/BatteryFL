# EIFFEL
Code for the paper - EIFFEL

!python main.py --num_clients 10 --optimizer Adam --lr 0.01 --momentum 0.5 --AR fedavg --dataset mnist --loader_type iid --experiment_name "Battery" --device cpu --epochs 5 --batch_size 64 --test_batch_size 64 --collection_size 400 --collection_battery 0.001 --training_battery 0.002 --training_size 1000 --alpha 0.5 --beta 0.5 --gamma 0.5 --mu 0.5 --training_size 400 --entropy_threshold 0.5 --upload_battery 3 download_battery 3 --collection_success_chance 0.95 --sample_selection loss
