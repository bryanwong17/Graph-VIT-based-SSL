from __future__ import absolute_import, division, print_function

import random
import os
import numpy as np

import torch

import torch.nn as nn
from torch.cuda.amp import grad_scaler #https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html

from utils.dataset import GraphDataset #Do I need to use Graph data as well? Or only labels

from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

#from models.GraphTransformer import Classifier
from models.OnlyVisionTransformer import Classifier

from models.weight_init import weight_init

from datetime import datetime
from draw import get_loss_curve, get_accuracy_curve
from pytorchtools import EarlyStopping

from models.ViT import VisionTransformer

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        # seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms. See also torch.are_deterministic_algorithms_enabled() and torch.use_deterministic_algorithms().
    torch.backends.cudnn.benchmark = False

def main():
    # for reproductibility
    seed_everything(1001)

    args = Options().parse()
    n_class = args.n_class

    torch.cuda.synchronize() #Waits for all kernels in all streams on a CUDA device to complete.
    torch.backends.cudnn.deterministic = True #Why repeat? Because of synchronization?

    data_path = args.data_path
    model_path = args.model_path

    #Why do we need to make directories in two steps?
    if not os.path.isdir(model_path.split("/saved_models")[0]): os.mkdir(model_path.split("/saved_models")[0])
    if not os.path.isdir(model_path): os.mkdir(model_path)
    
    log_path = args.log_path
    if not os.path.isdir(log_path): os.mkdir(log_path)
    # task name for naming saved model files and log files
    task_name = args.task_name
    print(task_name)
    ###################################
    # default false for train, test, graphcam
    train = args.train
    test = args.test
    graphcam = args.graphcam
    print("train:", train, "test:", test, "graphcam:", graphcam) #Does GraphCAM mean visualization?

    ##### Load datasets
    print("preparing datasets and dataloaders......")
    # 8 for training validation and 1 for testing
    batch_size = args.batch_size

    # training
    if train:
        ids_train = open(args.train_set).readlines()
        # print(ids_train)
        # return sample dict contains label, id(name), features, adj
        dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train) #Why bother join path with "" ?
        #Details on pytorch dataloader https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=8, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
        # batch size: 8 # The default values is 4 right?
        total_train_num = len(dataloader_train) * batch_size

    # validation or testing
    # _val here means validation
    ids_val = open(args.val_set).readlines()
    dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=8, collate_fn=collate, shuffle=False, pin_memory=True)
    total_val_num = len(dataloader_val) * batch_size
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##### creating models #############
    print("creating models......")

    # args.num_epochs = 120
    num_epochs = args.num_epochs
    # args.lr = 1e-3 
    learning_rate = args.lr

    #model = Classifier(n_class)
    model = Classifier(n_class)
    #This is for parallel computing. So  we don't need to
    model = nn.DataParallel(model)

    # for load model (testing and GraphCAM visualization)
    if args.resume:
        print('load model{}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if torch.cuda.is_available():
        model = model.cuda()
    #model.apply(weight_init)

    #lr: 1e-3, weight decay: 5e-4
    #Adam optimizer involves a combination of two gradient descent methodologies
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)       # best:5e-4, 4e-3
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

    ##################################

    # criterion = nn.CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()

    if not test:
        # ../graph_transformer/runs/GraphVIT
        #SummaryWriter is for tensorboard visualization https://pytorch.org/docs/stable/tensorboard.html
        writer = SummaryWriter(log_dir=log_path + task_name)
        f_log = open(os.path.join(log_path, task_name + ".log"), 'w')
    #Use the model already defined
    trainer = Trainer(n_class)
    #seem to be related to evaluation using some metrics like confusion metrics
    evaluator = Evaluator(n_class)

    best_pred = 0.0

    start_time = datetime.now()
    print("Start Time: {start_time}")

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    #    """Early stops the training if validation loss doesn't improve after a given patience."""
    early_stopping = EarlyStopping(verbose=True)
    scaler = grad_scaler.GradScaler()

    # num_epochs 120 for training validation and 1 for testing
    for epoch in range(num_epochs):
        # optimizer.zero_grad()
        model.train()
        train_loss = 0.
        val_loss = 0
        total = 0.

        current_lr = optimizer.param_groups[0]['lr']
        print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch+1, current_lr, best_pred))
        #First train the model if not test but train mode in options
        if train:
            #Based on my understanding in https://pytorch.org/docs/stable/data.html, i_batch is the index and smaple_batched is the batched sample
            for i_batch, sample_batched in enumerate(dataloader_train):
                #scheduler(optimizer, i_batch, epoch, best_pred)
                # optimizer.zero_grad()
                # 1 batch
                # with autocast_mode.autocast():
                preds,labels,loss = trainer.train(sample_batched, model)

                # scaler.scale(loss / 8).backward()

                # if (i_batch + 1) % 8 == 0:
                #     scaler.step(optimizer)
                #     # scaler.step(scheduler)
                #     scaler.update()
                #     optimizer.zero_grad()

                #Sets the gradients of all optimized torch.Tensor s to zero.
                optimizer.zero_grad()
                #Backpropagate gradients https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                loss.backward()
                #All optimizers implement a step() method, that updates the parameters. https://pytorch.org/docs/stable/optim.html
                optimizer.step()
                scheduler.step()
                #Sum up all losses and count the traning samples
                train_loss += loss.detach()
                total += len(labels)

                #Update the confusion matrix metrics
                trainer.metrics.update(labels, preds)
                # trainer.plot_cm()

                # log_interval_local:6 (print every batch size x log_interval_local: 8 x 6 = 48)
                if (i_batch + 1) % args.log_interval_local == 0:
                    print("[%d/%d] train loss: %.3f; train acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                    trainer.plot_cm()#print confusion matrix
        
        # print the last one (total) [208/208]
        if not test: 
            print("[%d/%d] train loss: %.3f; train acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
            train_losses.append((train_loss / total).item())
            train_accs.append(trainer.get_scores())
            trainer.plot_cm()


        # applies to every epoch (validation) and testing one epoch
        if epoch % 1 == 0:
            with torch.no_grad(): #Context-manager that disabled gradient calculation.  https://pytorch.org/docs/stable/generated/torch.no_grad.html
                #https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
                model.eval()
                print("evaluating...")

                total = 0.
                batch_idx = 0

                # _val means validation
                for i_batch, sample_batched in enumerate(dataloader_val):
                    #pred, label, _ = evaluator.eval_test(sample_batched, model)
                    preds, labels, loss = evaluator.eval_test(sample_batched, model, graphcam)

                    total += len(labels)
                    val_loss += loss

                    evaluator.metrics.update(labels, preds)

                    # log_interval_local:6 (print every batch size x log_interval_local: 8 x 6 = 48)
                    if (i_batch + 1) % args.log_interval_local == 0:
                        print('[%d/%d] val loss: %.3f; val acc: %.3f' % (total, total_val_num, val_loss / total, evaluator.get_scores()))
                        evaluator.plot_cm()

                # print the last one [208/208]
                print('[%d/%d] val loss: %.3f; val acc: %.3f' % (total_val_num, total_val_num, val_loss / total, evaluator.get_scores()))
                val_losses.append((val_loss / total).item())
                val_accs.append(evaluator.get_scores())
                evaluator.plot_cm()

                # torch.cuda.empty_cache()
                #get scores in confusion matrix
                val_acc = evaluator.get_scores()
                if val_acc > best_pred: 
                    best_pred = val_acc
                    if not test:
                        print("saving model...")
                        # ../graph_transformer/saved_models/GraphVIT_{epoch}.pth
                        #WHAT IS A STATE_DICT IN PYTORCH https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html#
                        torch.save(model.state_dict(), os.path.join(model_path, task_name + ".pth"))

                log = ""
                log = log + 'epoch [{}/{}] ------ train acc = {:.4f}, val acc = {:.4f}'.format(epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"

                log += "================================\n"
                print(log)
                if test:
                    break

                f_log.write(log)
                f_log.flush()
                #tensorbaord visualization
                writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)

                # early stopping
                early_stopping((val_loss / total).item(), model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        trainer.reset_metrics()
        evaluator.reset_metrics()

    if not test: f_log.close()

    print(f"Training Execution time: {datetime.now() - start_time}")

    if train:
        #draw the results to file
        get_loss_curve(args.figure_path, train_losses, val_losses)
        get_accuracy_curve(args.figure_path, train_accs, val_accs)

if __name__ == "__main__":
    main()