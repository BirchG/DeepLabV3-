import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from math import floor

import custom_transforms as tr

"""
获取数据矩阵
"""


class Mydata(Dataset):
    def __init__(self, data_path, label_path, subset='train'):
        self.data = []
        self.label = []
        self.subset = subset
        dirs = os.listdir(data_path)
        for d in dirs:
            imgpath = data_path + '/' + d
            # img = cv.imread(data_path + '/' + d)
            # img_tensor = (torch.tensor(img, dtype=torch.float)[:, :, 0].unsqueeze(0)) / 255
            # img_tensor = img_tensor[:, :(int(floor(img_tensor.shape[1] / 16) * 16)),
            #              :(int(floor(img_tensor.shape[2] / 16) * 16))]
            # self.data.append(img_tensor)
            self.data.append(imgpath)
        for d in dirs:
            maskpath = label_path + '/' + d
            # img = cv.imread(label_path + '/' + d)
            # img_tensor = (torch.tensor(img, dtype=torch.float)[:, :, 0].unsqueeze(0)) / 255
            # img_tensor = img_tensor[:, :(int(floor(img_tensor.shape[1] / 16) * 16)),
            #              :(int(floor(img_tensor.shape[2] / 16) * 16))]
            # self.label.append(img_tensor)
            self.label.append(maskpath)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        mask = Image.open(self.label[index])

        sample = {'image': img, 'label': mask}
        if self.subset == 'train':
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=513, crop_size=513),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

        elif self.subset == 'test':
            composed_transforms = transforms.Compose([
                tr.FixScaleCrop(crop_size=513),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else:
            raise ValueError

        sample = composed_transforms(sample)
        # print(sample['label'])
        return sample['image'], sample['label']


    def __len__(self):
        return len(self.data)


def LoadData(train_data_path="./data/train_data", train_label_path="./data/train_label",
             test_data_path="./data/test_data", test_label_path="./data/test_label",
             batch_size=4, shuffle=True, **kwargs):
    train_dataset = Mydata(train_data_path, train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_dataset = Mydata(test_data_path, test_label_path, 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
