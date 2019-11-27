import os
import argparse
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

#import nsml
#from nsml.constants import DATASET_PATH, GPU_NUM

from dataprocessing import TrainDatasetFromFolder

parser = argparse.ArgumentParser(description='Train Classification Models')
parser.add_argument('--Crop_h_size', default=1000, type=int, help='Center Crop height size')
parser.add_argument('--Crop_w_size', default=1000, type=int, help='Center Crop weight sizen')
parser.add_argument('--h_size', default=600, type=int, help='height resize')
parser.add_argument('--w_size', default=600, type=int, help='weight resize')
parser.add_argument('--pad', default=20, type=int, help='Crop image padding')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=10, type=int, help='batch size') #7x13x43=3913


opt = parser.parse_args()


CROP_H_SIZE = opt.Crop_h_size
CROP_W_SIZE = opt.Crop_w_size
H_SIZE = opt.h_size
W_SIZE = opt.w_size
PAD = opt.pad
NUM_EPOCHS = opt.num_epochs
BATCH_SIZE = opt.batch_size



train_set = TrainDatasetFromFolder('data/train', CROP_H_SIZE, CROP_W_SIZE, H_SIZE, W_SIZE, PAD)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_loader))
img = []
for _, (data, label) in enumerate(train_loader):
    print(_)
    print(data.shape, label)
    img.append(data[0])
    break

ToPILImage()(img[0])
