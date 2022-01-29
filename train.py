from metrics import evaluate_epoch_iou,iou
from utils import utils as utils
from torch import nn
from models import bilinear_kernel
import torchvision
from data import load_data
from loss import DiceLoss
import torch


def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs
    """
    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_iou_sum = iou(pred, y)
    return train_loss_sum, train_iou_sum

def train(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=utils.try_all_gpus()):
    """Train a model with mutiple GPUs

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = utils.Timer(), len(train_iter)
    animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = utils.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, iou_sum = train_batch(
                net, features, labels, loss, trainer, devices)
            print(iou_sum)
            metric.add(l, iou_sum, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_iou = evaluate_epoch_iou(net, test_iter)
        animator.add(epoch + 1, (None, None, test_iou))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_iou:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
#copies all the pretrained layers in the ResNet-18 except for 
#the final global average pooling layer and the fully-connected layer that are closest to the output
net = nn.Sequential(*list(pretrained_net.children())[:-2])


num_classes = 1
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
batch_size = 4
train_iter, valid_iter,test_iter = load_data(batch_size)

loss = DiceLoss()

num_epochs, lr, wd, devices = 1, 0.001, 1e-3, utils.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
train(net, train_iter, valid_iter, loss, trainer, num_epochs, devices)
