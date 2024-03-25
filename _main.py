from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients import *
from server import Server
import numpy as np
import matplotlib.pyplot as plt

def writing_function(writer, text, client, testData, step):
    loss, accuracy, F1, conf = client.test(testData)
    writer.add_scalar("Client_"+str(text)+'_test/loss', loss, global_step = step)
    writer.add_scalar("Client_"+str(text)+'_test/accuracy', accuracy, global_step = step)
    writer.add_scalar("Client_"+str(text)+'_test/F1', F1, global_step = step)
    fig = plt.figure(); plt.imshow(conf, cmap='gray', vmin=0, vmax=255)
    writer.add_figure("Client_"+str(text)+'_test/conf', fig, global_step = step)
    writer.add_scalar("Client_"+str(text)+'_train_loss', client.losses[-1], global_step = step)
    writer.add_scalar("Client_"+str(text)+'_battery_level', client.report_battery(), global_step = step)

def main(args):
    print('#####################')
    print('#####################')
    print('#####################')
    print(f'Aggregation Rule:\t{args.AR}\nData distribution:\t{args.loader_type} ')
    print('#####################')
    print('#####################')
    print('#####################')

    torch.manual_seed(args.seed)

    device = args.device

    writer = SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')

    if args.dataset == 'mnist':
        from tasks import mnist
        trainData = mnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
        num_classes = 10
    elif args.dataset == 'cifar':
        from tasks import cifar
        trainData = cifar.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
        num_classes = 10
    elif args.dataset == 'agro':
        from tasks import agro
        trainData = agro.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                          store=False)
        testData = agro.test_dataloader(args.test_batch_size)
        Net = agro.Net
        criterion = F.cross_entropy
        num_classes = 15

    # create server instance
    model0 = Net()
    server = Server(model = model0, dataLoader = testData, criterion = criterion, device = device, client_selection = args.client_selection, num_classes = num_classes)
    server.set_AR(args.AR)
    server.path_to_aggNet = args.path_to_aggNet
    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
    # create clients instance
    method = args.sample_selection

    clients_list = []
    battery = 100
    for i in range(args.num_clients):
        model = Net()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        client_i = Client(cid = i, battery = battery, model = model, dataLoader = trainData[i], 
                          optimizer = optimizer, criterion = criterion, reputation_method = method, device = device, 
                          batch_size = args.batch_size, round_budget = args.round_budget,
                          collection_size = args.collection_size, collection_prob = args.collection_success_chance, training_size = args.training_size,
                          entropy_threshold = args.entropy_threshold, momentum = args.momentum, num_classes = num_classes)
        client_i.init_update_battery()
        client_i.init_collection_battery(args.collection_battery_ratio)
        server.attach(client_i)
        clients_list.append(client_i)
        print("Client",i,"initialized with", client_i.report_battery(), "battery")

    loss, accuracy, F1, conf = server.test()
    step = 0
    #writing_function(writer, "test", loss, accuracy, F1, conf, steps)
    
    #for i,client in enumerate(clients_list) :
    #        loss, accuracy, F1, conf = client.test
    #        writing_function(writer, "client"+str(i)+'test', loss, accuracy, F1, conf, steps)

    while True:
        step += 1

        print('\n\n########EPOCH %d ########' % step)
        print('###Model distribution###\n')
        if not(server.do()):
            print('No clients have any battery left')
            break
        writer.add_scalar('Server/loss', loss, global_step = step)
        writer.add_scalar('Server/accuracy', accuracy, global_step = step)
        writer.add_scalar('Server/F1', F1, global_step = step)
        fig = plt.figure(); plt.imshow(conf, cmap='gray', vmin=0, vmax=255)
        writer.add_figure('Server/conf', fig, global_step = step)
        writer.add_scalars("Server/selected_clients", {"Selected_Clients": server.get_selected_clients()}, global_step = step)

        #         server.train_concurrent(group)
        for i, client in enumerate(clients_list) :
            #loss, accuracy, F1, conf = client.train_checking()
            #writing_function(writer, "client"+str(i)+'train', loss, accuracy, F1, conf, steps)

            #loss, accuracy, F1, conf = client.test(testData)
            writing_function(writer, str(i), client, testData, step)
            
            print("Client", i, "now has", client.report_battery() ,"battery left \n")

        loss, accuracy, F1, conf = server.test()

    writer.close()
