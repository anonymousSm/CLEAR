# from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import time
import math
import numpy as np

import tensorboard_logger as tb_logger
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import sinkhorn
from resnet_big import SupConResNet, SwavResNet, LinearClassifier
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='80,100,120',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')

    parser.add_argument('--cosine',  type=bool, default= True)

    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='2',
                        help='id for recording multiple runs')

    # New

    parser.add_argument('--num_proto', type=int, default='10',
                        help='number of prototypes')

    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='number of prototypes')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='number of prototypes')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon_swav/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon_swav/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt



# TODO
class Options():
    def __init__(self, save_freq = 50,
        batch_size = 1024,
        trial = 0,
        num_proto = 100,
        epsilon = 0.1,
        beta1 = 1,
        beta2 = 1,
        epochs=100,
        lr_decay_epochs = [60, 75, 90]):

        self.print_freq = 20
        # self.save_freq = 50
        # self.batch_size = 1024
        self.num_workers = 4
        self.epochs = epochs
        self.learning_rate = 0.1
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_rate = 0.1
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.model = 'resnet18'
        self.dataset = 'cifar10'
        self.mean = None
        self.std = None
        self.data_folder = None
        self.size = 32
        self.method = 'SimCLR' # 'SimCLR'
        self.temp = 0.1
        self.cosine = True
        self.syncBN = False
        self.warm = False
        # self.trial = 1
        # self.num_proto = 100
        # self.epsilon = 0.01
        # self.beta1 = 1
        # self.beta2 = 1

        self.save_freq = save_freq
        self.batch_size = batch_size
        self.trial = trial
        self.num_proto = num_proto
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2


        # check if dataset is path that passed required arguments
        if self.dataset == 'path':
            assert self.data_folder is not None \
                   and self.mean is not None \
                   and self.std is not None

        # set the path according to the environment
        if self.data_folder is None:
            self.data_folder = './datasets/'
        self.model_path = './save/SupCon_swav/{}_models'.format(self.dataset)
        self.tb_path = './save/SupCon_swav/{}_tensorboard'.format(self.dataset)

        # iterations = self.lr_decay_epochs.split(',')
        # self.lr_decay_epochs = list([])
        # for it in iterations:
        #     self.lr_decay_epochs.append(int(it))

        # self.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'. \
        #     format(self.method, self.dataset, self.model, self.learning_rate,
        #            self.weight_decay, self.batch_size, self.temp, self.trial)
        #
        self.model_name = '{}_{}_{}_lr_{}_{}_beta_{}_{}_bsz_{}_temp_{}_trial_{}_{}'. \
            format(self.method, self.dataset, self.model, self.learning_rate, self.num_proto,
                   self.beta1, self.beta2, self.batch_size, self.temp,  self.trial, self.epochs)

        if self.cosine:
            self.model_name = '{}_cosine'.format(self.model_name)

        # warm-up for large-batch training,
        if self.batch_size > 256:
            self.warm = True
        if self.warm:
            self.model_name = '{}_warm'.format(self.model_name)
            self.warmup_from = 0.01
            self.warm_epochs = 10
            if self.cosine:
                eta_min = self.learning_rate * (self.lr_decay_rate ** 3)
                self.warmup_to = eta_min + (self.learning_rate - eta_min) * (
                        1 + math.cos(math.pi * self.warm_epochs / self.epochs)) / 2
            else:
                self.warmup_to = self.learning_rate

        self.tb_folder = os.path.join(self.tb_path, self.model_name)
        if not os.path.isdir(self.tb_folder):
            os.makedirs(self.tb_folder)

        self.save_folder = os.path.join(self.model_path, self.model_name)
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

        # Linear

        if self.dataset == 'cifar10':
            self.n_cls = 10
        elif self.dataset == 'cifar100':
            self.n_cls = 100
        else:
            raise ValueError('dataset not supported: {}'.format(self.dataset))

        # Linear
        # self.linear_name = None
        self.ckpt = None


class Linear_Options(Options):
    def __init__(self):
        Options.__init__(self)
        # self.print_freq = 40
        # self.save_freq = 2
        # self.batch_size = 1024
        # self.num_workers = 4

        self.learning_rate = 0.1
        # self.epochs = 100
        # self.lr_decay_epochs = [60,75,90]

        self.epochs = 60
        self.lr_decay_epochs = [30,40,50]

        self.lr_decay_rate = 0.1
        self.weight_decay = 0
        # self.momentum = 0.9
        # self.model = 'resnet18'
        # self.dataset = 'cifar10'
        self.cosine = True
        # self.syncBN = False
        # self.num_proto = 100
        # self.epsilon = 0.01
        # self.beta = 1.0
        self.ckpt = None

        if self.dataset == 'cifar10':
            self.n_cls = 10
        elif self.dataset == 'cifar100':
            self.n_cls = 100
        else:
            raise ValueError('dataset not supported: {}'.format(self.dataset))




def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

def set_model(opt):
    model = SwavResNet(name='resnet18', num_proto=10)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features, output = model(images)

        # compute loss

        loss1 = 0
        if opt.beta1:

            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == 'SupCon':
                loss1 = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss1 = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))



        # if args.beta2:
        # ============ swav loss ... ============
        # features = features.detach()

        loss2 = 0
        if opt.beta2:

            for i, crop_id in enumerate([0, 1]):
                with torch.no_grad():
                    out = output[bsz * crop_id: bsz * (crop_id + 1)]

                    # get assignments
                    q = out / opt.epsilon
                    # if opt.improve_numerical_stability:
                    M = torch.max(q)
                        # dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q -= M
                    q = torch.exp(q).t()
                    q = sinkhorn(q)[-bsz:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum([2])), crop_id):
                    p = nn.Softmax(dim=1)(output[bsz * v: bsz * (v + 1)] / opt.temp)
                    subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                loss2 += subloss / (np.sum([2]) - 1)
            loss2 /= 2


        loss = opt.beta1 * loss1 + opt.beta2 * loss2


        # update metric
        losses1.update(loss1 if loss1 ==0 else loss1.item(), bsz)
        losses2.update(loss2 if loss2 ==0 else loss2.item(), bsz)
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1, loss2=losses2))
            sys.stdout.flush()

    return losses1.avg, losses2.avg, losses.avg


## linear  ##
def set_linear_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_linear_model(opt, input_model = None):

    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if input_model == None:
        model = SwavResNet(name=opt.model)
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
    else:
        model = input_model

    return model, classifier, criterion

def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()

    return losses.avg, top1.avg

def validate_linear(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #
            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         idx, len(val_loader), batch_time=batch_time,
            #         loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def train_eval_linear(train_loader, val_loader,model, classifier, criterion, optimizer, opt):


    # opt = parse_option()

    # build data loader

    # build model and criterion
    # model, classifier, criterion = set_linear_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, classifier)

    # training routine
    best_acc = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        _, acc = train_linear(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        _, val_acc = validate_linear(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
    return best_acc



def main(
        save_freq = 50,
        batch_size = 1024,
        trial = 1,
        num_proto = 100,
        epsilon = 0.01,
        beta1 = 1,
        beta2 = 1,
        epochs=100,
        lr_decay_epochs=[60, 75, 90]
):
    # opt = parse_option()

    opt = Options(save_freq = save_freq,
        batch_size = batch_size,
        trial = trial,
        num_proto = num_proto,
        epsilon = epsilon,
        beta1 = beta1,
        beta2 = beta2,
                  epochs=epochs,
                  lr_decay_epochs=lr_decay_epochs)

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)


    # ----------  Linear  ---------- #
    lopt = Linear_Options()

    train_linear_loader, val_linear_loader = set_linear_loader(lopt)

    # build model and criterion
    _, classifier, linear_criterion = set_linear_model(lopt, model)

    # build optimizer
    linear_optimizer = set_optimizer(lopt, classifier)


    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss1, loss2, loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss1', loss1, epoch)
        logger.log_value('loss2', loss2, epoch)
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            print(save_file)

            save_model(model, optimizer, opt, epoch, save_file)


        # Load model and train/test linear

        # if epoch % opt.save_freq == 0:
            # print(' ## Linear classifier ##')
            # lopt = Linear_Options()
            # lopt.ckpt = save_file
            # best_acc = train_eval_linear(lopt)
            # best_acc = train_eval_linear(train_linear_loader, val_linear_loader, model, classifier, linear_criterion, linear_optimizer, lopt)
            # logger.log_value('acc', best_acc, epoch)
        # # resnet_model, classifier, criterion = set_linear_model(opt)    # loss = torch.nn.CrossEntropyLoss()
        # _, classifier, = set_linear_model(opt)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



    best_acc = train_eval_linear(train_linear_loader, val_linear_loader, model, classifier, linear_criterion,
                                 linear_optimizer, lopt)
    logger.log_value('best_acc', best_acc, epoch)

    return

