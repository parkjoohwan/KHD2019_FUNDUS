"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model import EfficientNet
#from dataprocessing import TrainDatasetFromFolder

import torchvision
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', default='data/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b7',
                    help='model architecture (default: efficientnet-b7)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=254, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args() #하이퍼 파라미터값을 받아옴

    #결정모드는 모델에 따라 성능에 영향을 줌.
    #모델이 비 결정적일때 보다 처리속도가 느려질수 있음을 의미함.
    #seed = None -> 비결정모드
    #seed = not Nome -> 결정모드
    if args.seed is not None: #initalizing training
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #gpu는 무슨 gpu를 사용할지 지정 default값은 none이기때문에 gpu를 사용할 것이라면 번호로 지정
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #set up distributed training.
    #정확히 잘 모르겠는데 트레이닝할때 분할하여 하는것같음 분할 개수는 world_size이고 병렬처리를 하는것 같음
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() #gpu개수를 확인
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        #nprocs = 프로세스 수
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) #프로레스 기반 병렬처리
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args) #병렬처리 하지않음


#ngpus_per_node : 사용가능한 gpu 개수,
#gpu : 사용할 gpu 개수
#args : 하이퍼파라미터
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1 #초기값은 0
    args.gpu = gpu

    #몇개의 GPU를 사용할지 출력해줌
    #GPU를 사용하지않으면 출력하지 않음.
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    #병렬처리 할지말지말지를 결정
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    # args.arch는 현재 사용할 모델의 이름이 담겨있는 하이퍼파라미터
    if 'efficientnet' in args.arch:  # efficientnet을 사용할 경우
        if args.pretrained: #pretrained된 모델을 사용하겠다는 것. default값으로는 pretrained된 모델을 사용함.
            model = EfficientNet.from_pretrained(args.arch) #pretrained된 efficientnet의 weigth을 가져와 model에 넣음 이 함수는 model.py에 있음
            print("=> using pre-trained model '{}'".format(args.arch))
        else: #pre-trained된 모델을 사용하지 않는다면
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch) #pretrained되지않은 efficientnet을 가져와 model에 넣음

    else: #efficientnet이 아닌 다른 모델을 사용할 경우
        if args.pretrained: #pretrained된 모델을 사용할지 여부
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.distributed: #병렬처리를 사용할 경우
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None: #병렬 처리를 사용 안하고, gpu를 사용할 경우
        torch.cuda.set_device(args.gpu) #gpu의 개수만큼 cuda를 설정
        model = model.cuda(args.gpu) #설정한 gpu에 model을 올림
    else: #병렬 처리를 안하고 gpu도 사용안할 경우
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'): #모델이 alexnet이나 vgg기반 모델일 경우
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else: #모델이 alexnet이나 vgg기반 모델이 아닐 경우
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # CrossEntropyLoss를 사용 (softmax function)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu) #loss를 gpu에 올림

    #optimizer는 SGD를 사용하고 하이퍼 파라미터는 다음과 같이 설정.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    # resume은 최신 checkpoint의 파일 경로를 읽어오는것으로 default값은 None임
    if args.resume:
        if os.path.isfile(args.resume): #최신 checkpoint파일이 존재한다면
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) #최신 checkpoint를 로드함.
            args.start_epoch = checkpoint['epoch'] #최신 checkpoint의 epoch을 일거와서 학습으로 시작할 epoch에 맞춰줌
            best_acc1 = checkpoint['best_acc1'] #checkpoing의 성능이 best_acc1가됨. best_acc1는 현재 0인 상태임
            if args.gpu is not None: #gpu를 사용한다면
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu) #checkpoint의 best_acc1를 gpu에 올림
            model.load_state_dict(checkpoint['state_dict']) #checkpoint의 현재상태를 model에 올림
            optimizer.load_state_dict(checkpoint['optimizer']) #optimizer의 상태도 checkpoint의 상태로 변경
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])) #불러온 checkpoint의 정보를 출력
        else: #최신 checkpoint가 없다면 다음을 출력하고 진행
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn에서 benchmark모드를 활성화함
    # bechmark모드는 네트워크의 입력 크기가 다를때 사용하는것이 좋음
    # cudnn은 특정 구성에 대해 최적의 알고리즘 세트를 찾음(시간이 소요됨), 일반적으로 runtime이 빨라짐
    # 그러나 각 epoch에서 입력 크기가 변경되면 cudnn은 새로운 크기가 나타날 때마다 benchmark을 하기때문에 runtime 성능 저하 가능성이있음
    cudnn.benchmark = True


    """
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    """

    transform_train = transforms.Compose([
        transforms.Resize((600, 600), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)







    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if 'efficientnet' in args.arch: #모델이 efficientnet이라면
        image_size = EfficientNet.get_image_size(args.arch) #efficientnet의 이미지 사이즈를 받아옴 ex) 'efficientnet-b7'이면 size=600
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            #normalize,
        ])
        print('Using image size', image_size)
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #normalize,
        ])
        print('Using image size', 224)


    """
    #validation loader
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    """


    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transforms)

    # validation loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    #evaluate하는경우 weight을 업데이트 해줄필요가 없기때문에 구분지어놓음
    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, f)
        return

    #start_epoch부터 epochs까지 loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #learning rate 조정하는 함수 아래에 선언되어있음
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images) #output은 총 classes_num만큼 나옴
        loss = criterion(output, target) #output으로 나온 것과 target값과 비교하여 loss 측정

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

#learning rate decay하는 방식으로 조절해도됨
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
