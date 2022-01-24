import os
import torch
import torchvision
from utils import utils as utils
import glob
from matplotlib import pyplot as plt

def read_RE_images(RE_dir, is_train=True):
    """Read all RE feature and label images."""

    data_dir = os.path.join(RE_dir)
    mode = torchvision.io.image.ImageReadMode.RGB
    JPGImages = glob.glob(data_dir + '/*.jpg')

    assert len(JPGImages) == 6226

    PNGImages = [JPGImage.replace('sat.jpg','mask.png') for JPGImage in JPGImages]
    features, labels = [], []
    for JPGPath,PNGPath in zip(JPGImages,PNGImages):
        features.append(torchvision.io.read_image(JPGPath))
        labels.append(torchvision.io.read_image(PNGPath, mode))
    return features, labels

train_features, train_labels = read_RE_images('data/Road_Extraction_Dataset')
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
utils.show_images(imgs, 2, n)
plt.show()
