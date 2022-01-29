import os
import torch
import torchvision
from utils import utils as utils
import glob
from matplotlib import pyplot as plt
import random


COLORS = [[0,0,0], [255,255,255]]
CLASSES = ['background', 'road']
RE_dir = 'data/Road_Extraction_Dataset'


def read_RE_images(RE_dir, is_train=True):
    """Read all RE feature and label images."""

    data_dir = os.path.join(RE_dir)
    mode = torchvision.io.image.ImageReadMode.RGB
    JPGImages = glob.glob(data_dir + '/*.jpg')[:300]
    num_examples = len(JPGImages)
    indices = list(range(num_examples))
    random.shuffle(indices)
    train_indices = torch.tensor(indices[0:int(0.9*num_examples)])
    print(train_indices)

    PNGImages = [JPGImage.replace('sat.jpg','mask.png') for JPGImage in JPGImages]
    
    train_features, train_labels, test_features, test_labels = [], [],[],[]
    idx = 0
    for JPGPath,PNGPath in zip(JPGImages,PNGImages):
        if idx in train_indices:
            train_features.append(torchvision.io.read_image(JPGPath))
            train_labels.append(torchvision.io.read_image(PNGPath, mode))
        else:
            test_features.append(torchvision.io.read_image(JPGPath))
            test_labels.append(torchvision.io.read_image(PNGPath, mode))
        idx += 1
    return train_features, train_labels,test_features,test_labels

def color2label():
    color2label = torch.zeros(2, dtype=torch.long)
    for i, color in enumerate(COLORS):
        if color[0] == 255:
            color2label[1] = 1
    return color2label

def label_indices(colormap, color2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = (0 + colormap[:, :, 0])/255
    return color2label[idx]

class RoadSegDetaset(torch.utils.data.Dataset):
    """A customized dataset to load the RE dataset."""
    def __init__(self, features, labels):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color2label = color2label()
        self.features = [self.normalize_image(feature)
                         for feature in features]
        self.labels = labels
        print('read ' + str(len(self.features)) + ' examples')
    
    def normalize_image(self, img):
        return self.transform(img.float() / 255)


    def __getitem__(self, idx):
        return (self.features[idx], label_indices(self.labels[idx], self.color2label))
    
    def __len__(self):
        return len(self.features)



def load_data(batch_size):
    #divide dataset into training set and testing set with ratio 9:1
    random.seed(1) 
    train_features, train_labels,test_features, test_labels = read_RE_images(RE_dir, True)
    
    print(len(train_features))
    train_dataset = RoadSegDetaset(train_features,train_labels)
    test_dataset = RoadSegDetaset(test_features,test_labels)
    # assert len(train_dataset) + len(test_dataset) == 6226

    num_workers = utils.get_dataloader_workers()

    train_iter = torch.utils.data.DataLoader(train_dataset,
                    batch_size,shuffle=True, drop_last=True, 
                    num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(test_dataset,
                batch_size,shuffle=True, drop_last=True, 
                num_workers=num_workers)



    return train_iter, test_iter
# if __name__ == "__main__":
#     train_iter, test_iter = load_data(64)
#     for X,Y in train_iter:
#         print(X.shape)
#         print(Y.shape)