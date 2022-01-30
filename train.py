from metrics import evaluate_epoch_iou,iou
from utils import utils as utils
from torch import nn
from models import bilinear_kernel
import torchvision
from data import load_data
from loss import DiceLoss
import torch
from data import reverse_label
from models import My_Model
import sys
import segmentation_models_pytorch as smp



def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs
    """
    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    # pr_mask = reverse_label(pred[0].squeeze(0))
    # gt_mask = reverse_label(y[0])
    
    # utils.visualize(pr_mask =pr_mask.cpu(),gt_mask=gt_mask.cpu())
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_iou_sum = iou(pred, y)
    return train_loss_sum, train_iou_sum

def train(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=utils.try_all_gpus()):
    """Train a model with mutiple GPUs
    """
    NAME = 'log_My_Model'
    mylog = open('./logs/'+NAME+'.log','w')

    timer, num_batches = utils.Timer(), len(train_iter)
    train_epoch_best_loss = 1.
    animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print(epoch)
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = utils.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, iou_sum = train_batch(
                net, features, labels, loss, trainer, devices)

            metric.add(l, iou_sum, 1, 1)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(metric[1] / metric[3])
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_iou = evaluate_epoch_iou(net, test_iter)
        train_epoch_loss = metric[0] / metric[2]
        animator.add(epoch + 1, (None, None, test_iou))
    print(f'loss {metric[0] / metric[2]:.3f}, train IoU '
          f'{metric[1] / metric[3]:.3f}, test IOU {test_iou:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    print >> mylog, '********'
    print >> mylog, 'epoch:',epoch,'    time:',timer.sum()
    print >> mylog, 'train_loss:',metric[0] / metric[2]

    if train_epoch_loss >= train_epoch_best_loss:
        pass
    else:
        train_epoch_best_loss = train_epoch_loss
        torch.save(net, './weights/best_model_My_Model.pth')

def main(argv):
    if argv and argv[0] == "My_Model":
        batch_size = 8
        train_iter, valid_iter,test_iter = load_data(batch_size)
        loss = DiceLoss()
        num_epochs, lr, wd, devices = 5, 0.0001, 1e-3, utils.try_all_gpus()
        net = My_Model()
        trainer  = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd)
        train(net, train_iter, valid_iter, loss, trainer, num_epochs, devices)

    
    #Concise implementation using open souce library
    elif argv and argv[0] == "Unet":

        EPOCHS = 3
        batch_size = 8
        train_iter, valid_iter,test_iter = load_data(batch_size)

        model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
            activation='sigmoid'             # model output channels (number of classes in your dataset)
        )
        # define loss function
        loss = smp.utils.losses.DiceLoss()

        # define metrics
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        # define optimizer
        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=0.00008),
        ])

        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=utils.try_all_gpus()[0],
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=utils.try_all_gpus()[0],
            verbose=True,
        )

        

        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_iter)
            valid_logs = valid_epoch.run(valid_iter)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './weights/best_model_Unet.pth')
                print('Model saved!')

if __name__ == "__main__":
   main(sys.argv[1:])
