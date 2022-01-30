from data import load_data
import torch 
from metrics import evaluate_epoch_iou,iou
from data import reverse_label
from utils import utils as utils


batch_size = 2
train_iter, valid_iter,test_iter = load_data(batch_size)
My_Model = torch.load('weights/best_model_My_Model.pth')
Unet = torch.load('weights/best_model_Unet.pth')
test_iou_My_Model = evaluate_epoch_iou(My_Model, test_iter)
test_iou_Unet = evaluate_epoch_iou(Unet, test_iter)
print("My model test IOU: " + str(test_iou_My_Model))
print("Unet test IOU:" + str(test_iou_Unet))

for i, (features, labels) in enumerate(train_iter):
    pred_My_Model = My_Model(features)
    pred_Unet = Unet(features.cuda())
    pr_mask_My_Model = reverse_label(pred_My_Model[0].squeeze(0))
    pr_mask__Unet  = reverse_label(pred_Unet[0].squeeze(0))
    gt_mask = reverse_label(labels[0])
    
    utils.visualize(gt_mask = gt_mask.cpu(), pr_mask_My_Model =pr_mask_My_Model.cpu(),pr_mask__Unet=pr_mask__Unet.cpu())
