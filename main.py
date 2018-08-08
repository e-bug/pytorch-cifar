# -*- coding: utf-8 -*-
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import glob

import numpy as np

from models import *
from utils import progress_bar


# ==================================================================================================================== #
#                                                      Arguments                                                       #
# ==================================================================================================================== #
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

# Basic options
parser.add_argument('-m', '--model', default='resnet56',
                    help="""Name of model to run: 
                            densenet[121,161,169,201], dpn[26,92], googlenet, lenet, mobilenet, mobilenetv2, 
                            pnasnet[a,b], preact_resnet[18,34,50,101,152], resnet[18,20,32,34,44,50,56,101,110,152], 
                            resnext[29_2x64d,29_4x64d,29_8x64d,29_32x4d], senet18, shufflenet[g2,g3], 
                            vgg[11,13,16,19].""")
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help="""Size of the total minibatch.""")
parser.add_argument('-e', '--num_epochs', default=182, type=int,
                    help="""Number of epochs to run.""")
parser.add_argument('--data_dir', default='./data',
                    help="""Path to 'cifar-10-batches-py/' parent directory.""")
parser.add_argument('--num_classes', default=10, type=int,
                    help="""Number of classes: 
                            -  10: CIFAR-10.
                            - 100: CIFAR-100.""")
# Logging options
parser.add_argument('--log_dir', default='./checkpoints',
                    help="""Directory in which to write checkpoints.""")
parser.add_argument('--save_best', action='store_true',
                    help="""Whether to save the model's state at each epoch in which its test accuracy improved.""")
parser.add_argument('--save_every', default=0, type=int,
                    help="""Save model's state (checkpoint) every `save_every` epochs.""")
parser.add_argument('--num_ckpts', default=0, type=int,
                    help="""Number of checkpoints to store across `num_epochs` epochs. Overwrites `save_every`.""")
parser.add_argument('--resume', '-r', action='store_true',
                    help="""Whether to resume from checkpoint.""")
parser.add_argument('--progress_bar', action='store_true',
                    help="""Whether to display statistics at each epoch with an interactive progress bar or
                            by just printing the ones at the end of each epoch.""")
# Optimization options
parser.add_argument('--lr', default=0.1, type=float,
                    help="""Initial learning rate.""")
parser.add_argument('--momentum', default=0.9,
                    help="""Momentum factor.""")
parser.add_argument('--weight_decay', default=1e-4,
                    help="""Weight decay (L2 penalty).""")
parser.add_argument('--lr_decay_policy', default=None,
                    help="""Learning rate decay policy:
                            - None: constant learning rate.
                            - step: step decay (initial lr decayed by `lr_decay_rate` every `step_size` epochs).
                            - pconst: multi step decay (initial lr decayed by `lr_decay_rate` once the 
                                                        number of epoch reaches one of the milestones).
                            - exp: exponential decay (initial lr exponentially decayed by `lr_decay_rate` every epoch).
                         """)
parser.add_argument('--lr_decay_rate', default=0.1,
                    help="""Multiplicative factor of learning rate decay.""")
parser.add_argument('--lr_step_size', default=0.1,
                    help="""Period (in number of epochs) of learning rate decay in `step`.""")
parser.add_argument('--lr_milestones', nargs='*', default=[91, 136], type=int,
                    help="""List of increasing epoch indices when to decay the learning rate in `pconst`.""")

args = parser.parse_args()


# ==================================================================================================================== #
#                                                   Data preparation                                                   #
# ==================================================================================================================== #
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

nclasses = args.num_classes
if nclasses == 10:
    dataset_fn = torchvision.datasets.CIFAR10
elif nclasses == 100:
    dataset_fn = torchvision.datasets.CIFAR100
else:
    raise ValueError("Invalid CIFAR dataset: CIFAR-%d" % nclasses)

trainset = dataset_fn(root=args.data_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = dataset_fn(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ==================================================================================================================== #
#                                                   Model definition                                                   #
# ==================================================================================================================== #
model_name = args.model.lower()
print('==> Building %s model..' % model_name)
if model_name.startswith('densenet'):
    n = model_name[len('densenet'):]
    net = eval('DenseNet'+n)(num_classes=nclasses)
elif model_name.startswith('dpn'):
    n = model_name[len('dpn'):]
    net = eval('DPN' + n)()
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % DPN.__name__)
elif model_name == 'googlenet':
    net = GoogLeNet()
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % GoogLeNet.__name__)
elif model_name == 'lenet':
    net = LeNet()
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % LeNet.__name__)
elif model_name == 'mobilenet':
    net = MobileNet(num_classes=nclasses)
elif model_name == 'mobilenetv2':
    net = MobileNetV2(num_classes=nclasses)
elif model_name.startswith('pnasnet'):
    n = model_name[len('pnasnet'):]
    net = eval('PNASNet' + n.upper())()
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % PNASNet.__name__)
elif model_name.startswith('preact_resnet'):
    n = model_name[len('preact_resnet'):]
    net = eval('PreActResNet' + n)(num_classes=nclasses)
elif model_name.startswith('resnet'):
    n = model_name[len('resnet'):]
    net = eval('ResNet' + n)(num_classes=nclasses)
elif model_name.startswith('resnext'):
    n = model_name[len('resnext'):]
    net = eval('ResNeXt' + n)(num_classes=nclasses)
elif model_name == 'senet18':
    net = SENet18(num_classes=nclasses)
elif model_name.startswith('shufflenet'):
    n = model_name[len('shufflenet'):]
    net = eval('ShuffleNet' + n.upper())()
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % ShuffleNet.__name__)
elif model_name.startswith('vgg'):
    n = model_name[len('vgg'):]
    net = VGG('VGG' + n)
    if nclasses == 100:
        raise ValueError('%s not implemented for CIFAR-100' % VGG.__name__)
else:
    raise ValueError("Invalid model type: %s" % model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# ==================================================================================================================== #
#                                                       Logging                                                        #
# ==================================================================================================================== #
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
ckpt_time_elapsed = 0  # time elapsed training to last checkpoint epoch

ckpts = []
if args.num_ckpts:
    ckpts = np.linspace(0, args.num_epochs-1, args.num_ckpts, dtype=int)
elif args.save_every:
    ckpts = np.arange(0, args.num_epochs, args.save_every, dtype=int)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from last checkpoint..')
    assert os.path.isdir(args.log_dir), 'Error: no checkpoint directory found!'
    ckpt_files = glob.glob(os.path.join(args.log_dir, 'ckpt.*[0-9]'))
    ckpt_epochs = [int(fn.split('ckpt.')[-1]) for fn in ckpt_files]
    checkpoint = torch.load(ckpt_files[np.argmax(ckpt_epochs)])
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    ckpt_time_elapsed = checkpoint['time_elapsed']
elif os.path.isdir(args.log_dir) and len(ckpts):
    # Delete old checkpoints.
    ckpt_files = glob.glob(os.path.join(args.log_dir, 'ckpt.*'))
    for fn in ckpt_files:
        glob.os.remove(fn)


# ==================================================================================================================== #
#                                                     Optimization                                                     #
# ==================================================================================================================== #
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, 
                      weight_decay=args.weight_decay, nesterov=False)

scheduler_name = args.lr_decay_policy
if scheduler_name == 'step':
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay_rate)
elif scheduler_name == 'pconst':
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_rate)
elif scheduler_name == 'exp':
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay_rate)
elif scheduler_name is None:
    scheduler = MultiStepLR(optimizer, milestones=[0], gamma=1)
else:
    raise ValueError("Invalid learning rate decay policy: %s" % scheduler_name)


# ==================================================================================================================== #
#                                                       Training                                                       #
# ==================================================================================================================== #
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        msg = 'lr: %.3e | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            scheduler.get_lr()[0], train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        if args.progress_bar:
            progress_bar(batch_idx, len(trainloader), msg)
    if not args.progress_bar:
        print(' ' + msg)


# ==================================================================================================================== #
#                                                         Test                                                         #
# ==================================================================================================================== #
def test(epoch):
    global best_acc, since
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)
            if args.progress_bar:
                progress_bar(batch_idx, len(testloader), msg)
        if not args.progress_bar:
            print(' ' + msg)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        if args.save_best:
            print('Saving best..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.log_dir):
                os.makedirs(args.log_dir)
            torch.save(state, os.path.join(args.log_dir, 'ckpt.best'))
    if epoch in ckpts or epoch == args.num_epochs:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'best_acc': best_acc,
            'epoch': epoch,
            'time_elapsed': time.time() - since + ckpt_time_elapsed,
        }
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        torch.save(state, os.path.join(args.log_dir, 'ckpt.'+str(epoch)))


# ==================================================================================================================== #
#                                                         Run                                                          #
# ==================================================================================================================== #
since = time.time()
for epoch in range(start_epoch, args.num_epochs):
    scheduler.step()
    train(epoch)
    test(epoch)
time_elapsed = time.time() - since + ckpt_time_elapsed
print('Training complete in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: %4f' % best_acc)
