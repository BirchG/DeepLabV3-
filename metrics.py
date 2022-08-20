import cv2
import numpy as np

import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.inter = 0
        self.union = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.Boundary_Count(pre_image, gt_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Boundary_Count(self, pre_pic, real_pic):
        def get_boundary(pic, is_mask):
            # if not is_mask:
            #     pic = torch.argmax(pic, 1).cpu().numpy().astype('float64')
            # else:
            #     pic = pic.cpu().numpy()
            batch, width, height = pic.shape
            new_pic = np.zeros([batch, width + 2, height + 2])
            mask_erode = np.zeros([batch, width, height])
            dil = int(round(0.02 * np.sqrt(width ** 2 + height ** 2)))
            if dil < 1:
                dil = 1
            for i in range(batch):
                new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            kernel = np.ones((3, 3), dtype=np.uint8)
            for j in range(batch):
                pic_erode = cv2.erode(new_pic[j], kernel, iterations=dil)
                mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
            return torch.from_numpy(pic - mask_erode)

        inter = 0
        union = 0
        pre_pic = get_boundary(pre_pic, is_mask=False)
        real_pic = get_boundary(real_pic, is_mask=False)
        batch, width, height = pre_pic.shape
        for i in range(batch):
            predict = pre_pic[i]
            mask = real_pic[i]
            inter += ((predict * mask) > 0).sum()
            union += ((predict + mask) > 0).sum()
        self.inter += inter
        self.union += union

    def Boundary_IoU(self):
        if self.union < 1:
            return 0
        biou = (self.inter / self.union)
        return biou
