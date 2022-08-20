import os
import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import LoadData
from opts import build_args
from deeplab_xception import *
from metrics import Evaluator
import custom_transforms as tr


def train(epoch, step, model, loader, criterion, optimizer, writer, args):
    # 模型训练函数
    model.train()
    running_loss = 0.0

    print('[Epoch {} / lr {:.2e}]'.format(epoch, optimizer.param_groups[0]['lr']))
    tq = tqdm.tqdm(loader, ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')
    batches = len(tq)
    for idx, sample in enumerate(tq):
        images = sample[0].to(device=args.device, dtype=args.dtype)
        mask = sample[1].to(device=args.device).long()

        outputs = model(images)
        loss = criterion(outputs, mask)
        # print(outputs.shape,mask.shape)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        per_loss = loss.item()
        running_loss += per_loss
        mean_loss = running_loss / (idx + 1)
        # 损失
        writer.add_scalars('train/Loss', {'Loss': per_loss}, global_step=step)
        tq.set_description('Train Loss:{:.4f}'.format(mean_loss))
        if step == 1:  # 模型结构图
            writer.add_graph(model, images)
        # if step % batches == 0:  # 每个epoch采样结果可视化
        #     writer.add_figure('figures', show_img(images, mask, outputs), global_step=step)

        step += 1

    writer.add_scalars('train/Average Loss', {'Average Loss': mean_loss}, global_step=epoch + 1)
    return step


def evaluate(epoch, model, val_loader, criterion, writer, args):
    model.eval()
    running_loss = 0.0
    evaluator = Evaluator(num_class=2)

    tq = tqdm.tqdm(val_loader, ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}')
    for idx, sample in enumerate(tq):
        data = sample[0].to(device=args.device, dtype=args.dtype)
        mask = sample[1].to(device=args.device).long()

        with torch.no_grad():
            outputs = model(data)

        loss = criterion(outputs, mask)
        running_loss += loss.item()
        mean_loss = running_loss / (idx + 1)
        tq.set_description('Val Loss:{:.4f}'.format(mean_loss))
        pred = outputs.data.cpu().numpy()
        mask = mask.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(mask, pred)

    Acc = evaluator.Pixel_Accuracy()
    mIoU = evaluator.Mean_Intersection_over_Union()
    BIoU = evaluator.Boundary_IoU()
    writer.add_scalar('val/mean_loss', mean_loss, epoch)
    writer.add_scalar('val/mIoU', mIoU, epoch)
    writer.add_scalar('val/Acc', Acc, epoch)
    writer.add_scalar('val/BIoU', BIoU, epoch)
    print('Acc:{:.3f}, mIoU:{:.3f}, BIoU:{:.3f}'.format(mean_loss, mIoU, BIoU))
    return mIoU


def main(args):
    # device
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(args.device))

    # log
    writer = SummaryWriter(args.checkpoint + '/runs')

    # model
    if args.backbone == 'xception':
        model = DeepLabv3_plus_xception(nInputChannels=3, n_classes=2, os=args.out_stride, pretrained=True,
                                        _print=True).to(args.device)
    # elif args.backbone == 'resnet':
    #     model = DeepLabv3_plus_resnet(nInputChannels=3, n_classes=2, os=args.out_stride, pretrained=True,
    #                                   _print=True).to(args.device)
    else:
        raise ValueError

    # optimize
    train_params = [{'params': get_1x_lr_params(model), 'lr': args.lr},
                    {'params': get_10x_lr_params(model), 'lr': args.lr * 10}]
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)

    # data
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_loader, val_loader = LoadData(train_data_path="./data/train_data", train_label_path="./data/train_label",
                                        test_data_path="./data/test_data", test_label_path="./data/test_label",
                                        batch_size=4, shuffle=True, **kwargs)

    if args.mode == 'train':
        step = 1
        best = 0.0
        for epoch in range(args.epoch):
            step = train(epoch, step, model, train_loader, criterion, optimizer, writer, args)

            if epoch % args.interval == (args.interval - 1):
                # Save model
                torch.save(model.state_dict(), "%s/epoch_%s.pth" % (args.checkpoint, epoch))
                nowmiou = evaluate(epoch, model, val_loader, criterion, writer, args)
                if nowmiou > best:
                    best = nowmiou
                    torch.save(model.state_dict(), '%s/best.pth' % args.checkpoint)

    elif args.mode == 'test':
        testpath = './data/test_data'
        labelpath = "./data/test_label"
        dirs = os.listdir(testpath)
        for d in dirs:
            imgpath = testpath + '/' + d
            maskpath = labelpath + '/' + d
            img = Image.open(imgpath).convert('RGB')
            mask = Image.open(maskpath)
            sample = {'image': img, 'label': mask}
            composed_transforms = transforms.Compose([
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
            sample = composed_transforms(sample)

            img = sample['image'].to(device=args.device, dtype=args.dtype)
            mask = sample['label']

            model.load_state_dict(torch.load("%s/best.pth" % args.checkpoint))

            with torch.no_grad():
                outputs = model(img)

            pred = outputs.data.cpu().numpy()
            # mask = mask.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            fig, axes = plt.subplots(1, 2)
            axes[0, 0].imshow(pred, cmap='gray')
            axes[0, 1].imshow(mask, cmap='gray')
            plt.savefig("%s/%s_result.jpg" % (args.checkpoint, d[:-4]), dpi=600)


if __name__ == '__main__':
    main(build_args())
