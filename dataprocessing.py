from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, ColorJitter, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop
import torch


#이미지 파일 읽기
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


#train_data
def train_transform(Crop_h_size, Crop_w_size, h_size, w_size, pad):
    return Compose([
        CenterCrop((Crop_h_size, Crop_w_size)), #이미지에 가운데를 h_size, w_size로 Crop함
        Resize((h_size, w_size), interpolation=Image.BICUBIC), #이미지 크기를 h_size, w_size로 통일함
        #ColorJitter( #밝기를 랜덤하게 조정
        #    brightness = abs(0.1 * float(torch.randn(1))),
        #    contrast = abs(0.1 * float(torch.randn(1))),
        #    saturation = abs(0.1 * float(torch.randn(1))),
        #    hue = abs(0.1 * float(torch.randn(1)))
        #),
        RandomHorizontalFlip(p=0.5), #수평으로 랜덤하게 flip
        RandomVerticalFlip(p=0.5), #수직으로 랜덤하게 flip
        RandomCrop((h_size, w_size), padding=pad), #이미지를 Crop한 뒤에 빈 곳을 pad함
        ToTensor(),
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, Crop_h_size, Crop_w_size, h_size, w_size, pad):
        super(TrainDatasetFromFolder, self).__init__()
        self.AMD_image_filenames = [[join(dataset_dir, "AMD", x), 0] for x in listdir(join(dataset_dir, "AMD")) if is_image_file(x)]
        self.DMR_image_filenames = [[join(dataset_dir, "DMR", x), 1] for x in listdir(join(dataset_dir, "DMR")) if is_image_file(x)]
        self.NORMAL_image_filenames = [[join(dataset_dir, "NORMAL", x), 2] for x in listdir(join(dataset_dir, "NORMAL")) if is_image_file(x)]
        self.RVO_image_filenames = [[join(dataset_dir, "RVO", x), 3] for x in listdir(join(dataset_dir, "RVO")) if is_image_file(x)]
        self.image_filenames = self.AMD_image_filenames + self.DMR_image_filenames + self.NORMAL_image_filenames + self.RVO_image_filenames

        #data agumentation
        self.transform = train_transform(Crop_h_size, Crop_w_size, h_size, w_size, pad)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.image_filenames[index][0]))
        label = self.image_filenames[index][1]
        return image, label

    def __len__(self):
        return len(self.image_filenames)

