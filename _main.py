from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients import *
from server import Server

def writing_function(writer, text, loss, accuracy, F1, conf, steps):
    writer.add_scalar(str(text)+'/loss', loss, steps)
    writer.add_scalar(str(text)+'/accuracy', accuracy, steps)
    writer.add_scalar(str(text)+'/F1', F1, steps)
    writer.add_scalars(str(text)+'/conf', {"True Pos" : conf[0,0], "False Pos" : conf[0,1],
                                         "False Neg" : conf[1,0], "True Neg" : conf[1,1]}, steps)

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
    elif args.dataset == 'cifar':
        from tasks import cifar
        trainData = cifar.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar100':
        from tasks import cifar100
        trainData = cifar100.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                              store=False)
        testData = cifar100.test_dataloader(args.test_batch_size)
        Net = cifar100.Net
        criterion = F.cross_entropy
    elif args.dataset == 'imdb':
        from tasks import imdb
        trainData = imdb.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                          store=False)
        testData = imdb.test_dataloader(args.test_batch_size)
        Net = imdb.Net
        criterion = F.cross_entropy

    # create server instance
    model0 = Net()
    server = Server(model0, testData, args.upload_battery, args.download_battery, args.collection_battery, args.training_battery, args.collection_size, criterion, device)
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
        client_i = Client(i, battery, model, trainData[i], 
                          optimizer, criterion, method, device, args.inner_epochs, args.batch_size, 
                          args.upload_battery, args.download_battery, args.collection_battery, args.training_battery, 
                          args.collection_size, args.collection_success_chance, training_size = args.training_size,
                          entropy_threshold = args.entropy_threshold)
        server.attach(client_i)
        #client_i.collect_data(collection = 0)
        clients_list.append(client_i)
        print("Client",i,"initialized with", client_i.report_battery(), "battery")

    loss, accuracy, F1, conf = server.test()
    steps = 0
    #writing_function(writer, "test", loss, accuracy, F1, conf, steps)
    
    #for i,client in enumerate(clients_list) :
    #        loss, accuracy, F1, conf = client.test
    #        writing_function(writer, "client"+str(i)+'test', loss, accuracy, F1, conf, steps)

    for j in range(args.epochs):
        steps = j + 1

        print('\n\n########EPOCH %d ########' % j)
        print('###Model distribution###\n')
        #server.distribute()
        #         group=Random().sample(range(5),1)
        #group = range(args.num_clients)
        #server.collection_function()
        server.do()
        #         server.train_concurrent(group)
        loss, accuracy, F1, conf = server.test()
        writing_function(writer, "test", loss, accuracy, F1, conf, steps)

        for i, client in enumerate(clients_list) :
            #loss, accuracy, F1, conf = client.train_checking()
            #writing_function(writer, "client"+str(i)+'train', loss, accuracy, F1, conf, steps)


            loss, accuracy, F1, conf = client.test(testData)
            writing_function(writer, "client"+str(i)+'test', loss, accuracy, F1, conf, steps)
            
            print("Client", i, "now has", client.report_battery() ,"battery left \n")


    writer.close()
