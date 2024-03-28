import argparse
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of models")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=11)
    parser.add_argument("-n", "--num_clients", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default="experiments",
                        help="path to output folder, e.g. \"experiment\"")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar", "insect", "plant"], default="mnist")
    parser.add_argument("--loader_type", type=str, choices=["iid", "byLabel", "dirichlet"], default="iid")
    parser.add_argument("--loader_path", type=str, default="./data/loader.pk", help="where to save the data partitions")
    parser.add_argument("--AR", type=str, default='fednova')
    parser.add_argument("--save_model_weights", action="store_true")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--path_to_aggNet", type=str)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cuda')
    #parser.add_argument("--inner_epochs", type=int, default=1)
    #parser.add_argument("--upload_battery", type=float, default = 3)
    #parser.add_argument("--download_battery", type=float, default = 3)
    parser.add_argument("--collection_battery_ratio", type=float, default = 1)
    parser.add_argument("--collection_size", type=int, default = 1000)
    parser.add_argument("--collection_success_chance", type=float, default = 0.95)
    #parser.add_argument("--training_battery", type=float, default = 0.002)
    parser.add_argument("--sample_selection", type=str, choices=["loss","tracin"],default = 'loss')
    parser.add_argument("--client_selection", type=str, choices=["ours","NSGA","AGE","EAFL"],default='ours')
    parser.add_argument("--alpha", type=float, default = 0.5)
    parser.add_argument("--beta", type=float, default = 0.5)
    parser.add_argument("--gamma", type=float, default = 0.5)
    parser.add_argument("--mu", type=float, default = 0.5)
    parser.add_argument("--training_size", type=int, default = 400)
    parser.add_argument("--entropy_threshold", type=float, default = 0.5)
    parser.add_argument("--round_budget", type=float, default = 10)
    parser.add_argument("--starting_battery", type=float, default = 100)

    args = parser.parse_args()

    n = args.num_clients

    if args.experiment_name == None:
        args.experiment_name = f"{args.loader_type}/{args.attacks}/{args.AR}"
    
    
    return args


if __name__ == "__main__":

    import _main

    args = parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)
