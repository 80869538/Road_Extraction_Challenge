import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):

    def __init__(self, smooth=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0) #batch size
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        if dims is not None:
          intersection = torch.sum(y_pred * y_true, dim=dims)
          cardinality = torch.sum(y_true + y_pred, dim=dims)

     
        scores =  (2. * intersection + self.smooth) / (cardinality + self.smooth)
        losses = 1.0 - scores.mean()  #broadcast

        return losses.mean()