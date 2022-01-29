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

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
     
        scores =  (2. * intersection + self.smooth) / (cardinality + self.smooth)
        losses = 1.0 - scores  #broadcast

        return losses.mean()
