import argparse
import importlib
import json
import os
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets as datasets_torch
from torchvision import transforms

from model import Expert, Discriminator
from utils import init_weights


def initialize_expert(epochs, expert, i, optimizer, loss, data_train, args, writer):
    print("Initializing expert [{}] as identity on preturbed data".format(i+1))
    expert.train()

    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        for batch in data_train:
            x_canonical, x_transf = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            x_transf = x_transf.view(x_transf.size(0), -1).to(args.device)
            x_transf = x_transf.type(torch.FloatTensor)
            print("testing")
            print(x_canonical.size())
            print(x_transf.size())
            x_hat = expert(x_transf)

            loss_rec = loss(x_hat, x_transf)
            total_loss += loss_rec.item()*batch_size
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # Loss
        mean_loss = total_loss/n_samples
        print("initialization epoch [{}] expert [{}] loss {:.4f}".format(
            epoch+1, i+1, mean_loss))
        writer.add_scalar('expert_{}_initialization_loss'.format(
            i+1), mean_loss, epoch+1)
        if mean_loss < 0.002:
            break

    torch.save(expert.state_dict(), checkpt_dir +
               '/{}_E_{}_init.pth'.format(args.name, i + 1))


def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args, writer):
    print("checkpoint 0")
    discriminator.train()
    print("checkpoint 1")
    for i, expert in enumerate(experts):
        expert.train()

    print("checkpoint 2")

    # Labels for canonical vs transformed samples
    canonical_label = 1
    transformed_label = 0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    expert_scores_D = [0 for i in range(len(experts))]
    expert_winning_samples_idx = [[] for i in range(len(experts))]

    # Iterate through data
    for idx, batch in enumerate(data_train):
        x_canon, x_transf = batch
        print(f'{x_canon.size() = } {x_transf.size()=}')
        # x_transf = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
        batch_size = x_canon.size(0)
        n_samples += batch_size
        x_canon = x_canon.view(batch_size, -1).to(args.device)
        x_transf = x_transf.view(batch_size, -1).to(args.device)
        print(x_canon.size())
        # Train Discriminator on canonical distribution
        scores_canon = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label,
                            device=args.device).unsqueeze(dim=1)
        loss_D_canon = criterion(scores_canon, labels)
        total_loss_D_canon += loss_D_canon.item() * batch_size
        optimizer_D.zero_grad()
        loss_D_canon.backward()

        # Train Discriminator on experts output
        labels.fill_(transformed_label)
        loss_D_transformed = 0
        exp_outputs = []
        expert_scores = []
        for i, expert in enumerate(experts):
            exp_output = expert(x_transf)
            exp_outputs.append(exp_output.view(batch_size, 1, args.input_size))
            exp_scores = discriminator(exp_output.detach())
            expert_scores.append(exp_scores)
            loss_D_transformed += criterion(exp_scores, labels)
        loss_D_transformed = loss_D_transformed / args.num_experts
        total_loss_D_transformed += loss_D_transformed.item() * batch_size
        loss_D_transformed.backward()
        optimizer_D.step()

        # Train experts
        exp_outputs = torch.cat(exp_outputs, dim=1)
        expert_scores = torch.cat(expert_scores, dim=1)
        mask_winners = expert_scores.argmax(dim=1)

        # Update each expert on samples it won
        for i, expert in enumerate(experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            accrue = 0 if idx == 0 else 1
            expert_winning_samples_idx[i] += (winning_indexes +
                                              accrue*n_samples).tolist()
            n_expert_samples = winning_indexes.size(0)
            if n_expert_samples > 0:
                total_samples_expert[i] += n_expert_samples
                exp_samples = exp_outputs[winning_indexes, i]
                D_E_x_transf = discriminator(exp_samples)
                labels = torch.full((n_expert_samples,), canonical_label,
                                    device=args.device).unsqueeze(dim=1)
                loss_E = criterion(D_E_x_transf, labels)
                total_loss_expert[i] += loss_E.item() * n_expert_samples
                optimizers_E[i].zero_grad()
                # TODO figure out why retain graph is necessary
                loss_E.backward(retain_graph=True)
                optimizers_E[i].step()
                expert_scores_D[i] += D_E_x_transf.squeeze().sum().item()

    # Logging
    mean_loss_D_generated = total_loss_D_transformed / n_samples
    mean_loss_D_canon = total_loss_D_canon / n_samples
    print("epoch [{}] loss_D_transformed {:.4f}".format(
        epoch + 1, mean_loss_D_generated))
    print("epoch [{}] loss_D_canon {:.4f}".format(
        epoch + 1, mean_loss_D_canon))
    writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch + 1)
    writer.add_scalar('loss_D_transformed', mean_loss_D_generated, epoch + 1)
    for i in range(len(experts)):
        print("epoch [{}] expert [{}] n_samples {}".format(
            epoch + 1, i + 1, total_samples_expert[i]))
        writer.add_scalar('expert_{}_n_samples'.format(
            i + 1), total_samples_expert[i], epoch + 1)
        writer.add_text('expert_{}_winning_samples'.format(i + 1),
                        ":".join([str(j) for j in expert_winning_samples_idx[i]]), epoch + 1)
        if total_samples_expert[i] > 0:
            mean_loss_expert = total_loss_expert[i] / total_samples_expert[i]
            mean_expert_scores = expert_scores_D[i] / total_samples_expert[i]
            print("epoch [{}] expert [{}] loss {:.4f}".format(
                epoch + 1, i + 1, mean_loss_expert))
            print("epoch [{}] expert [{}] scores {:.4f}".format(
                epoch + 1, i + 1, mean_expert_scores))
            writer.add_scalar('expert_{}_loss'.format(
                i + 1), mean_loss_expert, epoch + 1)
            writer.add_scalar('expert_{}_scores'.format(
                i + 1), mean_expert_scores, epoch + 1)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--outdir', default='.', type=str,
                        help='path to the output directory')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='name of the dataset')
    parser.add_argument('--optimizer_experts', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--optimizer_discriminator', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--optimizer_initialize', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--input_size', type=int, default=784, metavar='N',
                        help='input size of data (default: 784)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs_init', type=int, default=1, metavar='N',
                        help='number of epochs to initially train experts (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--learning_rate_initialize', type=float, default=1e-1,
                        help='size of expert learning rate')
    parser.add_argument('--learning_rate_expert', type=float, default=1e-3,
                        help='size of expert learning rate')
    parser.add_argument('--learning_rate_discriminator', type=float, default=1e-3,
                        help='size of discriminator learning rate')
    parser.add_argument('--name', type=str, default='',
                        help='name of experiment')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for optimizer')
    parser.add_argument('--num_experts', type=int, default=1, metavar='N',
                        help='number of experts (default: 5)')
    parser.add_argument('--load_initialized_experts', type=bool, default=False,
                        help='whether to load already pre-trained experts')
    parser.add_argument('--model_for_initialized_experts', type=str, default='',
                        help='path to pre-trained experts')

    # Get arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Random seed
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Experiment name
    timestamp = str(int(time.time()))
    if args.name == '':
        name = '{}_n_exp_{}_bs_{}_lri_{}_lre_{}_lrd_{}_ei_{}_e_{}_oi_{}_oe_{}_oe_{}_{}'.format(
            args.dataset, args.num_experts, args.batch_size, args.learning_rate_initialize,
            args.learning_rate_expert, args.learning_rate_discriminator, args.epochs_init,
            args.epochs, args.optimizer_initialize, args.optimizer_experts, args.optimizer_discriminator,
            timestamp)
        args.name = name
    else:
        args.name = '{}_{}'.format(args.name, timestamp)
    print('\nExperiment: {}\n'.format(args.name))

    # Logging. To run: tensorboard --logdir <args.outdir>/logs
    log_dir = os.path.join(args.outdir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir_exp = os.path.join(log_dir, args.name)
    os.mkdir(log_dir_exp)
    writer = SummaryWriter(log_dir=log_dir_exp)

    # Directory for checkpoints
    checkpt_dir = os.path.join(args.outdir, 'checkpoints')
    if not os.path.exists(checkpt_dir):
        os.mkdir(checkpt_dir)

    # Load dataset
    # print("tim")
    # print(dir(datasets_torch))
    if args.dataset in dir(datasets_torch):
        # Pytorch dataset
        dataset = getattr(datasets_torch, args.dataset)
        train_transform = transforms.Compose([transforms.ToTensor()])
        kwargs_train = {'download': True, 'transform': train_transform}
        dataset_train = dataset(
            root='{}/{}'.format(args.datadir, args.dataset), train=True, **kwargs_train)
    else:
        # Custom dataset
        dataset_train = getattr(importlib.import_module(
            '{}'.format(args.dataset)), 'PatientsDataset')(args)

    # Create Dataloader from dataset
    data_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.cuda), pin_memory=args.cuda
    )

    # Model
    experts = [Expert(args).to(args.device) for i in range(args.num_experts)]
    discriminator = Discriminator(args).to(args.device)
    # Losses
    loss_initial = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.BCELoss(reduction='mean')

    # Initialize Experts as approximately Identity on Transformed Data
    for i, expert in enumerate(experts):
        if args.load_initialized_experts:
            path = os.path.join(checkpt_dir,
                                args.model_for_initialized_experts + '_E_{}_init.pth'.format(i+1))
            init_weights(expert, path)
        else:
            if args.optimizer_initialize == 'adam':
                optimizer_E = torch.optim.Adam(expert.parameters(), lr=args.learning_rate_initialize,
                                               weight_decay=args.weight_decay)
            elif args.optimizer_initialize == 'sgd':
                optimizer_E = torch.optim.SGD(expert.parameters(), lr=args.learning_rate_initialize,
                                              weight_decay=args.weight_decay)
            else:
                raise NotImplementedError
            initialize_expert(args.epochs_init, expert, i,
                              optimizer_E, loss_initial, data_train, args, writer)

    # Optimizers
    optimizers_E = []
    for i in range(args.num_experts):
        if args.optimizer_experts == 'adam':
            optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate_expert,
                                           weight_decay=args.weight_decay)
        elif args.optimizer_experts == 'sgd':
            optimizer_E = torch.optim.SGD(experts[i].parameters(), lr=args.learning_rate_expert,
                                          weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        optimizers_E.append(optimizer_E)
    if args.optimizer_discriminator == 'adam':
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                       weight_decay=args.weight_decay)
    elif args.optimizer_discriminator == 'sgd':
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate_discriminator,
                                      weight_decay=args.weight_decay)

    # Training
    for epoch in range(args.epochs):
        train_system(epoch, experts, discriminator, optimizers_E,
                     optimizer_D, criterion, data_train, args, writer)

        if epoch % args.log_interval == 0 or epoch == args.epochs-1:
            torch.save(discriminator.state_dict(), checkpt_dir +
                       '/{}_D.pth'.format(args.name))
            for i in range(args.num_experts):
                torch.save(experts[i].state_dict(), checkpt_dir +
                           '/{}_E_{}.pth'.format(args.name, i+1))
